// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! A single Raft node as a pure state machine.
//!
//! The node performs no I/O and owns no clock. The driver (a real runtime, or
//! the deterministic [`crate::sim::Sim`]) supplies the current logical time and
//! delivers messages; every method that can produce network traffic returns the
//! outgoing [`Envelope`]s for the driver to route. This makes the consensus
//! logic exhaustively testable: same inputs, same outputs, no threads, no time.
//!
//! Implements the safety-critical core of Raft (figure 2): leader election with
//! the up-to-date-log restriction, log replication with the consistency check,
//! and the commit rule that a leader only commits an entry from its *current*
//! term. Persistent state (`current_term`, `voted_for`, `log`) survives a
//! restart; volatile leader/commit state does not.

use std::collections::{HashMap, HashSet};

use crate::types::{Envelope, Index, LogEntry, Msg, NodeId, Role, Term};

/// Timing configuration (in the driver's logical time units).
#[derive(Clone, Copy)]
pub struct Timing {
    pub election_min: u64,
    pub election_max: u64,
    pub heartbeat: u64,
}

impl Default for Timing {
    fn default() -> Self {
        Timing { election_min: 150, election_max: 300, heartbeat: 30 }
    }
}

pub struct RaftNode {
    id: NodeId,
    peers: Vec<NodeId>,
    timing: Timing,

    // --- persistent state (would be fsynced before responding) ---
    current_term: Term,
    voted_for: Option<NodeId>,
    log: Vec<LogEntry>, // index i (1-based) == log[i-1]

    // --- volatile state (all nodes) ---
    role: Role,
    commit_index: Index,
    last_applied: Index,

    // --- volatile leader state ---
    next_index: HashMap<NodeId, Index>,
    match_index: HashMap<NodeId, Index>,

    // --- volatile candidate state ---
    votes_granted: HashSet<NodeId>,
    leader_id: Option<NodeId>,

    // --- timers (absolute deadlines) + RNG for randomized election timeout ---
    election_deadline: u64,
    heartbeat_deadline: u64,
    rng: u64,

    /// Commands applied to the state machine, in commit order — for inspection/tests.
    applied: Vec<(Index, Vec<u8>)>,
}

impl RaftNode {
    pub fn new(id: NodeId, peers: Vec<NodeId>, timing: Timing, now: u64) -> Self {
        let mut node = RaftNode {
            id,
            peers,
            timing,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            role: Role::Follower,
            commit_index: 0,
            last_applied: 0,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            votes_granted: HashSet::new(),
            leader_id: None,
            election_deadline: 0,
            heartbeat_deadline: 0,
            // Seed varies per node so randomized timeouts diverge (avoids lockstep split votes).
            rng: 0x9e37_79b9_7f4a_7c15 ^ id.wrapping_mul(0xbf58_476d_1ce4_e5b9),
            applied: Vec::new(),
        };
        node.reset_election_deadline(now);
        node
    }

    // --- accessors (for the driver/tests) ---
    pub fn id(&self) -> NodeId {
        self.id
    }
    pub fn role(&self) -> Role {
        self.role
    }
    pub fn term(&self) -> Term {
        self.current_term
    }
    pub fn commit_index(&self) -> Index {
        self.commit_index
    }
    pub fn leader_id(&self) -> Option<NodeId> {
        self.leader_id
    }
    pub fn last_log_index(&self) -> Index {
        self.log.len() as Index
    }
    /// Committed commands in order — the replicated state machine's input.
    pub fn applied(&self) -> &[(Index, Vec<u8>)] {
        &self.applied
    }
    pub fn log_terms(&self) -> Vec<Term> {
        self.log.iter().map(|e| e.term).collect()
    }

    /// Next moment this node needs servicing (election timeout, or heartbeat if leader).
    pub fn next_deadline(&self) -> u64 {
        match self.role {
            Role::Leader => self.heartbeat_deadline,
            _ => self.election_deadline,
        }
    }

    fn quorum(&self) -> usize {
        // Majority of the cluster (peers + self). Equivalent to floor(n/2)+1.
        self.peers.len().div_ceil(2) + 1
    }

    fn term_at(&self, index: Index) -> Term {
        if index == 0 || index > self.log.len() as Index {
            0
        } else {
            self.log[(index - 1) as usize].term
        }
    }

    fn last_log_term(&self) -> Term {
        self.term_at(self.last_log_index())
    }

    fn next_rand(&mut self) -> u64 {
        self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.rng >> 17
    }

    fn reset_election_deadline(&mut self, now: u64) {
        let span = self.timing.election_max - self.timing.election_min;
        let jitter = if span == 0 { 0 } else { self.next_rand() % span };
        self.election_deadline = now + self.timing.election_min + jitter;
    }

    /// On restart, volatile state is rebuilt; persistent state (term/vote/log) is kept.
    pub fn restart(&mut self, now: u64) {
        self.role = Role::Follower;
        self.leader_id = None;
        self.votes_granted.clear();
        self.next_index.clear();
        self.match_index.clear();
        self.commit_index = 0;
        self.last_applied = 0;
        self.applied.clear();
        self.reset_election_deadline(now);
    }

    // --- timer-driven actions ---

    /// Service expired timers at time `now`.
    pub fn tick(&mut self, now: u64) -> Vec<Envelope> {
        match self.role {
            Role::Leader if now >= self.heartbeat_deadline => {
                self.heartbeat_deadline = now + self.timing.heartbeat;
                self.peers.clone().iter().map(|&p| self.append_entries_to(p)).collect()
            }
            Role::Leader => Vec::new(),
            _ if now >= self.election_deadline => self.start_election(now),
            _ => Vec::new(),
        }
    }

    fn start_election(&mut self, now: u64) -> Vec<Envelope> {
        self.role = Role::Candidate;
        self.current_term += 1;
        self.voted_for = Some(self.id);
        self.votes_granted.clear();
        self.votes_granted.insert(self.id);
        self.leader_id = None;
        self.reset_election_deadline(now);

        let (lli, llt) = (self.last_log_index(), self.last_log_term());
        self.peers
            .iter()
            .map(|&p| Envelope {
                from: self.id,
                to: p,
                msg: Msg::RequestVote {
                    term: self.current_term,
                    candidate: self.id,
                    last_log_index: lli,
                    last_log_term: llt,
                },
            })
            .collect()
    }

    fn become_leader(&mut self, now: u64) -> Vec<Envelope> {
        self.role = Role::Leader;
        self.leader_id = Some(self.id);
        let next = self.last_log_index() + 1;
        for &p in &self.peers {
            self.next_index.insert(p, next);
            self.match_index.insert(p, 0);
        }
        self.heartbeat_deadline = now + self.timing.heartbeat;
        // Immediate authority-asserting heartbeats.
        self.peers.clone().iter().map(|&p| self.append_entries_to(p)).collect()
    }

    fn step_down(&mut self, term: Term) {
        self.current_term = term;
        self.role = Role::Follower;
        self.voted_for = None;
        self.votes_granted.clear();
        self.leader_id = None;
    }

    /// Build the AppendEntries RPC for `peer` from its `next_index`.
    fn append_entries_to(&self, peer: NodeId) -> Envelope {
        let next = *self.next_index.get(&peer).unwrap_or(&(self.last_log_index() + 1));
        let prev_log_index = next - 1;
        let prev_log_term = self.term_at(prev_log_index);
        let entries = if next <= self.last_log_index() {
            self.log[(next - 1) as usize..].to_vec()
        } else {
            Vec::new()
        };
        Envelope {
            from: self.id,
            to: peer,
            msg: Msg::AppendEntries {
                term: self.current_term,
                leader: self.id,
                prev_log_index,
                prev_log_term,
                entries,
                leader_commit: self.commit_index,
            },
        }
    }

    // --- client interface ---

    /// Propose a command. Returns the assigned index (and replication traffic)
    /// if this node is the leader, else `None`.
    pub fn propose(&mut self, command: Vec<u8>) -> (Option<Index>, Vec<Envelope>) {
        if self.role != Role::Leader {
            return (None, Vec::new());
        }
        self.log.push(LogEntry { term: self.current_term, command });
        let index = self.last_log_index();
        let msgs = self.peers.clone().iter().map(|&p| self.append_entries_to(p)).collect();
        (Some(index), msgs)
    }

    // --- message handling ---

    pub fn recv(&mut self, now: u64, from: NodeId, msg: Msg) -> Vec<Envelope> {
        match msg {
            Msg::RequestVote { term, candidate, last_log_index, last_log_term } => {
                self.handle_request_vote(now, term, candidate, last_log_index, last_log_term)
            }
            Msg::RequestVoteResp { term, granted } => self.handle_vote_resp(now, from, term, granted),
            Msg::AppendEntries { term, leader, prev_log_index, prev_log_term, entries, leader_commit } => {
                self.handle_append_entries(now, term, leader, prev_log_index, prev_log_term, entries, leader_commit)
            }
            Msg::AppendEntriesResp { term, success, match_index } => {
                self.handle_append_resp(now, from, term, success, match_index)
            }
        }
    }

    fn handle_request_vote(
        &mut self,
        now: u64,
        term: Term,
        candidate: NodeId,
        last_log_index: Index,
        last_log_term: Term,
    ) -> Vec<Envelope> {
        if term > self.current_term {
            self.step_down(term);
        }
        let up_to_date = (last_log_term, last_log_index) >= (self.last_log_term(), self.last_log_index());
        let granted = term == self.current_term
            && self.voted_for.is_none_or(|v| v == candidate)
            && up_to_date;
        if granted {
            self.voted_for = Some(candidate);
            self.reset_election_deadline(now);
        }
        vec![Envelope {
            from: self.id,
            to: candidate,
            msg: Msg::RequestVoteResp { term: self.current_term, granted },
        }]
    }

    fn handle_vote_resp(&mut self, now: u64, from: NodeId, term: Term, granted: bool) -> Vec<Envelope> {
        if term > self.current_term {
            self.step_down(term);
            return Vec::new();
        }
        if self.role != Role::Candidate || term != self.current_term || !granted {
            return Vec::new();
        }
        self.votes_granted.insert(from);
        if self.votes_granted.len() >= self.quorum() {
            self.become_leader(now)
        } else {
            Vec::new()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_append_entries(
        &mut self,
        now: u64,
        term: Term,
        leader: NodeId,
        prev_log_index: Index,
        prev_log_term: Term,
        entries: Vec<LogEntry>,
        leader_commit: Index,
    ) -> Vec<Envelope> {
        let reply = |node: &Self, success: bool, match_index: Index| {
            vec![Envelope {
                from: node.id,
                to: leader,
                msg: Msg::AppendEntriesResp { term: node.current_term, success, match_index },
            }]
        };

        if term < self.current_term {
            return reply(self, false, 0); // stale leader
        }
        if term > self.current_term {
            self.step_down(term);
        }
        // Valid current-term leader: (re)assert follower state and reset timeout.
        self.role = Role::Follower;
        self.leader_id = Some(leader);
        self.reset_election_deadline(now);

        // Log consistency check.
        if prev_log_index > self.last_log_index() || self.term_at(prev_log_index) != prev_log_term {
            return reply(self, false, self.last_log_index());
        }

        // Append, truncating on the first conflicting term.
        let mut idx = prev_log_index;
        for entry in entries {
            idx += 1;
            if idx <= self.last_log_index() {
                if self.term_at(idx) != entry.term {
                    self.log.truncate((idx - 1) as usize); // drop conflict + everything after
                    self.log.push(entry);
                }
                // else: already present and matching, skip
            } else {
                self.log.push(entry);
            }
        }

        if leader_commit > self.commit_index {
            self.commit_index = leader_commit.min(self.last_log_index());
            self.apply_committed();
        }
        reply(self, true, idx)
    }

    fn handle_append_resp(
        &mut self,
        now: u64,
        from: NodeId,
        term: Term,
        success: bool,
        match_index: Index,
    ) -> Vec<Envelope> {
        if term > self.current_term {
            self.step_down(term);
            return Vec::new();
        }
        if self.role != Role::Leader || term != self.current_term {
            return Vec::new();
        }
        if success {
            self.match_index.insert(from, match_index);
            self.next_index.insert(from, match_index + 1);
            self.maybe_commit();
            // Keep streaming if the follower is still behind.
            if *self.next_index.get(&from).unwrap_or(&1) <= self.last_log_index() {
                return vec![self.append_entries_to(from)];
            }
            Vec::new()
        } else {
            // Back up and retry (one step; hint could accelerate).
            let cur = *self.next_index.get(&from).unwrap_or(&1);
            let backed = cur.saturating_sub(1).max(1);
            self.next_index.insert(from, backed);
            let _ = now;
            vec![self.append_entries_to(from)]
        }
    }

    /// Advance commit_index to the highest N replicated on a quorum, but only if
    /// log[N] is from the current term (Raft's commit restriction).
    fn maybe_commit(&mut self) {
        let mut n = self.last_log_index();
        while n > self.commit_index {
            if self.term_at(n) == self.current_term {
                let replicated = 1 + self.peers.iter().filter(|p| *self.match_index.get(p).unwrap_or(&0) >= n).count();
                if replicated >= self.quorum() {
                    self.commit_index = n;
                    self.apply_committed();
                    break;
                }
            }
            n -= 1;
        }
    }

    fn apply_committed(&mut self) {
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            let cmd = self.log[(self.last_applied - 1) as usize].command.clone();
            self.applied.push((self.last_applied, cmd));
        }
    }
}
