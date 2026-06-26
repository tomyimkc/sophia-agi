// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Core Raft data types: ids, terms, log entries, and the RPC message set.

pub type NodeId = u64;
pub type Term = u64;
/// 1-based log position; index 0 is the empty sentinel (term 0, no command).
pub type Index = u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogEntry {
    pub term: Term,
    pub command: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Follower,
    Candidate,
    Leader,
}

/// The four Raft RPCs (request + response halves). Snapshots and membership
/// changes are out of scope for this core (documented in lib.rs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Msg {
    RequestVote {
        term: Term,
        candidate: NodeId,
        last_log_index: Index,
        last_log_term: Term,
    },
    RequestVoteResp {
        term: Term,
        granted: bool,
    },
    AppendEntries {
        term: Term,
        leader: NodeId,
        prev_log_index: Index,
        prev_log_term: Term,
        entries: Vec<LogEntry>,
        leader_commit: Index,
    },
    AppendEntriesResp {
        term: Term,
        success: bool,
        /// On success: highest index now matched on the follower. On failure: a
        /// hint (the follower's last index) so the leader can back up faster.
        match_index: Index,
    },
}

/// A message in flight from one node to another.
#[derive(Clone, Debug)]
pub struct Envelope {
    pub from: NodeId,
    pub to: NodeId,
    pub msg: Msg,
}
