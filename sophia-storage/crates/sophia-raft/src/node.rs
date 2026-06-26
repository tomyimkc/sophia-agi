// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! A single Raft node, driven by explicit `tick()` and `step(msg)` calls — no
//! threads, no wall-clock — so a cluster is fully deterministic and testable.
//!
//! Implements the safety-critical parts of the Raft paper: leader election with
//! the up-to-date-log vote restriction (§5.4.1), AppendEntries consistency check
//! with conflict truncation (§5.3), and the rule that a leader only advances the
//! commit index over an entry from its own term (§5.4.2).

use std::collections::{HashMap, HashSet};

use crate::log::RaftLog;
use crate::types::{Envelope, Index, Message, NodeId, Term};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Follower,
    Candidate,
    Leader,
}

pub struct RaftNode {
    pub id: NodeId,
    peers: Vec<NodeId>,
    pub role: Role,
    pub current_term: Term,
    voted_for: Option<NodeId>,
    pub log: RaftLog,

    // Election timing, in logical ticks. Distinct per node (caller sets the
    // timeout) so the cluster converges instead of livelocking on split votes.
    election_elapsed: u32,
    election_timeout: u32,
    heartbeat_elapsed: u32,
    heartbeat_timeout: u32,

    // Candidate state.
    votes: HashSet<NodeId>,
    // Leader state.
    next_index: HashMap<NodeId, Index>,
    match_index: HashMap<NodeId, Index>,
    leader_id: Option<NodeId>,
}

impl RaftNode {
    pub fn new(id: NodeId, peers: Vec<NodeId>, election_timeout: u32, heartbeat_timeout: u32) -> Self {
        RaftNode {
            id,
            peers,
            role: Role::Follower,
            current_term: 0,
            voted_for: None,
            log: RaftLog::new(),
            election_elapsed: 0,
            election_timeout,
            heartbeat_elapsed: 0,
            heartbeat_timeout,
            votes: HashSet::new(),
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            leader_id: None,
        }
    }

    pub fn is_leader(&self) -> bool {
        self.role == Role::Leader
    }

    pub fn leader_id(&self) -> Option<NodeId> {
        self.leader_id
    }

    fn quorum(&self) -> usize {
        // Strict majority of all nodes (peers + self).
        let total = self.peers.len() + 1;
        total / 2 + 1
    }

    /// Advance logical time by one tick, possibly starting an election (follower/
    /// candidate) or emitting heartbeats (leader).
    pub fn tick(&mut self) -> Vec<Envelope> {
        match self.role {
            Role::Leader => {
                self.heartbeat_elapsed += 1;
                if self.heartbeat_elapsed >= self.heartbeat_timeout {
                    self.heartbeat_elapsed = 0;
                    return self.broadcast_append();
                }
                Vec::new()
            }
            _ => {
                self.election_elapsed += 1;
                if self.election_elapsed >= self.election_timeout {
                    self.become_candidate()
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Client proposal: append to the leader's log and replicate. Returns the
    /// new entry's index, or `None` if this node is not the leader.
    pub fn propose(&mut self, command: Vec<u8>) -> Option<(Index, Vec<Envelope>)> {
        if self.role != Role::Leader {
            return None;
        }
        let index = self.log.append(crate::types::LogEntry { term: self.current_term, command });
        self.match_index.insert(self.id, index);
        Some((index, self.broadcast_append()))
    }

    /// Handle one incoming message, returning any messages to send in reply.
    pub fn step(&mut self, env: Envelope) -> Vec<Envelope> {
        // Universal rule: a higher term means we are stale — step down first.
        if env.msg.term() > self.current_term {
            self.become_follower(env.msg.term(), None);
        }
        match env.msg {
            Message::RequestVote { term, candidate, last_log_index, last_log_term } => {
                self.handle_request_vote(term, candidate, last_log_index, last_log_term)
            }
            Message::RequestVoteResp { term, voter, granted } => {
                self.handle_vote_resp(term, voter, granted)
            }
            Message::AppendEntries {
                term,
                leader,
                prev_log_index,
                prev_log_term,
                entries,
                leader_commit,
            } => self.handle_append(term, leader, prev_log_index, prev_log_term, entries, leader_commit),
            Message::AppendEntriesResp { term, follower, success, match_index } => {
                self.handle_append_resp(term, follower, success, match_index)
            }
        }
    }

    // --- role transitions ---

    fn become_follower(&mut self, term: Term, leader: Option<NodeId>) {
        self.role = Role::Follower;
        self.current_term = term;
        self.voted_for = None;
        self.leader_id = leader;
        self.election_elapsed = 0;
        self.votes.clear();
    }

    fn become_candidate(&mut self) -> Vec<Envelope> {
        self.role = Role::Candidate;
        self.current_term += 1;
        self.voted_for = Some(self.id);
        self.leader_id = None;
        self.election_elapsed = 0;
        self.votes.clear();
        self.votes.insert(self.id);
        // Single-node cluster: we already have a quorum.
        if self.votes.len() >= self.quorum() {
            return self.become_leader();
        }
        let (lli, llt) = (self.log.last_index(), self.log.last_term());
        self.peers
            .iter()
            .map(|&to| Envelope {
                from: self.id,
                to,
                msg: Message::RequestVote {
                    term: self.current_term,
                    candidate: self.id,
                    last_log_index: lli,
                    last_log_term: llt,
                },
            })
            .collect()
    }

    fn become_leader(&mut self) -> Vec<Envelope> {
        self.role = Role::Leader;
        self.leader_id = Some(self.id);
        self.heartbeat_elapsed = 0;
        let next = self.log.last_index() + 1;
        self.next_index.clear();
        self.match_index.clear();
        for &p in &self.peers {
            self.next_index.insert(p, next);
            self.match_index.insert(p, 0);
        }
        self.match_index.insert(self.id, self.log.last_index());
        // Establish authority immediately with heartbeats.
        self.broadcast_append()
    }

    // --- RequestVote ---

    fn handle_request_vote(
        &mut self,
        term: Term,
        candidate: NodeId,
        last_log_index: Index,
        last_log_term: Term,
    ) -> Vec<Envelope> {
        let mut granted = false;
        if term >= self.current_term {
            let can_vote = self.voted_for.is_none() || self.voted_for == Some(candidate);
            // §5.4.1: candidate's log must be at least as up-to-date as ours.
            let up_to_date = last_log_term > self.log.last_term()
                || (last_log_term == self.log.last_term()
                    && last_log_index >= self.log.last_index());
            if can_vote && up_to_date {
                granted = true;
                self.voted_for = Some(candidate);
                self.election_elapsed = 0; // granting a vote resets our timer
            }
        }
        vec![Envelope {
            from: self.id,
            to: candidate,
            msg: Message::RequestVoteResp { term: self.current_term, voter: self.id, granted },
        }]
    }

    fn handle_vote_resp(&mut self, term: Term, voter: NodeId, granted: bool) -> Vec<Envelope> {
        if self.role != Role::Candidate || term != self.current_term {
            return Vec::new();
        }
        if granted {
            self.votes.insert(voter);
            if self.votes.len() >= self.quorum() {
                return self.become_leader();
            }
        }
        Vec::new()
    }

    // --- AppendEntries ---

    fn handle_append(
        &mut self,
        term: Term,
        leader: NodeId,
        prev_log_index: Index,
        prev_log_term: Term,
        entries: Vec<crate::types::LogEntry>,
        leader_commit: Index,
    ) -> Vec<Envelope> {
        // Stale leader: reject.
        if term < self.current_term {
            return vec![self.append_resp(leader, false, 0)];
        }
        // Valid leader for our term: (re)become follower and reset election timer.
        self.become_follower(term, Some(leader));

        if !self.log.matches(prev_log_index, prev_log_term) {
            // Consistency check failed; hint our last index so the leader backs up.
            return vec![self.append_resp(leader, false, self.log.last_index())];
        }
        self.log.append_from(prev_log_index, &entries);
        let new_match = prev_log_index + entries.len() as Index;
        if leader_commit > self.log.commit_index {
            self.log.commit_index = leader_commit.min(self.log.last_index());
        }
        vec![self.append_resp(leader, true, new_match)]
    }

    fn handle_append_resp(
        &mut self,
        term: Term,
        follower: NodeId,
        success: bool,
        match_index: Index,
    ) -> Vec<Envelope> {
        if self.role != Role::Leader || term != self.current_term {
            return Vec::new();
        }
        if success {
            self.match_index.insert(follower, match_index);
            self.next_index.insert(follower, match_index + 1);
            self.maybe_advance_commit();
            Vec::new()
        } else {
            // Back up next_index and retry (hint-bounded so we don't underflow).
            let ni = self.next_index.get(&follower).copied().unwrap_or(1);
            let backed = ni.saturating_sub(1).max(1).min(match_index.max(1));
            self.next_index.insert(follower, backed);
            vec![self.append_to(follower)]
        }
    }

    /// §5.4.2: find the highest N > commit_index replicated on a majority whose
    /// entry is from the **current term**, and commit up to it.
    fn maybe_advance_commit(&mut self) {
        let last = self.log.last_index();
        let mut n = self.log.commit_index + 1;
        while n <= last {
            if self.log.term_at(n) == self.current_term {
                let replicas = self
                    .match_index
                    .values()
                    .filter(|&&m| m >= n)
                    .count();
                if replicas >= self.quorum() {
                    self.log.commit_index = n;
                }
            }
            n += 1;
        }
    }

    // --- message builders ---

    fn broadcast_append(&mut self) -> Vec<Envelope> {
        let peers = self.peers.clone();
        peers.into_iter().map(|p| self.append_to(p)).collect()
    }

    fn append_to(&self, follower: NodeId) -> Envelope {
        let next = self.next_index.get(&follower).copied().unwrap_or(self.log.last_index() + 1);
        let prev_log_index = next - 1;
        let prev_log_term = self.log.term_at(prev_log_index);
        let entries = self.log.entries_after(prev_log_index);
        Envelope {
            from: self.id,
            to: follower,
            msg: Message::AppendEntries {
                term: self.current_term,
                leader: self.id,
                prev_log_index,
                prev_log_term,
                entries,
                leader_commit: self.log.commit_index,
            },
        }
    }

    fn append_resp(&self, to: NodeId, success: bool, match_index: Index) -> Envelope {
        Envelope {
            from: self.id,
            to,
            msg: Message::AppendEntriesResp {
                term: self.current_term,
                follower: self.id,
                success,
                match_index,
            },
        }
    }
}
