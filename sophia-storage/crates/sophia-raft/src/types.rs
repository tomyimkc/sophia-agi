// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Core Raft types: ids, terms, log entries, and the RPC messages.

pub type NodeId = u64;
pub type Term = u64;
/// 1-based log index; `0` means "before the first entry" (empty log).
pub type Index = u64;

/// One replicated command plus the term it was created in. The `command` is an
/// opaque byte string — for Sophia it is a serialized decision-log record or
/// queue mutation; the state machine ([`crate::StateMachine`]) interprets it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogEntry {
    pub term: Term,
    pub command: Vec<u8>,
}

/// The Raft RPCs (request + response variants), as values the harness routes.
#[derive(Clone, Debug)]
pub enum Message {
    RequestVote {
        term: Term,
        candidate: NodeId,
        last_log_index: Index,
        last_log_term: Term,
    },
    RequestVoteResp {
        term: Term,
        voter: NodeId,
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
        follower: NodeId,
        success: bool,
        /// On success: the follower's last matching index. On failure: a hint
        /// (the follower's last index) so the leader can back `next_index` up.
        match_index: Index,
    },
}

/// A message in flight from one node to another.
#[derive(Clone, Debug)]
pub struct Envelope {
    pub from: NodeId,
    pub to: NodeId,
    pub msg: Message,
}

impl Message {
    /// The term carried by any message (used for the universal "step down if we
    /// see a higher term" rule).
    pub fn term(&self) -> Term {
        match self {
            Message::RequestVote { term, .. }
            | Message::RequestVoteResp { term, .. }
            | Message::AppendEntries { term, .. }
            | Message::AppendEntriesResp { term, .. } => *term,
        }
    }
}
