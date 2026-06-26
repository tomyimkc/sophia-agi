// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! # sophia-raft
//!
//! A deterministic **Raft** consensus core — the "分布式事务 / Paxos·Raft 共识"
//! requirement made demonstrable rather than asserted. It replicates an ordered
//! command log across N nodes so Sophia's decision log / task queue
//! (`sophia_contract/queue.py`, today single-process JSONL) can be made
//! highly-available: a committed entry survives any minority failure, and the
//! cluster keeps serving through a leader crash.
//!
//! ## Design
//! - [`node::RaftNode`] is the algorithm, driven by explicit `tick()` / `step()`
//!   calls — no threads, no wall-clock — implementing leader election with the
//!   up-to-date-log vote restriction, AppendEntries consistency + conflict
//!   truncation, and current-term-only commit advancement.
//! - [`cluster::Cluster`] is an in-memory harness that routes messages and
//!   drives logical time, so elections / replication / partitions / crashes are
//!   reproducible. A production deployment swaps it for a real transport and a
//!   durable per-node log (an [`sophia-lsm`](../sophia_lsm/index.html) engine),
//!   leaving the node core unchanged.
//! - [`state_machine::StateMachine`] turns the committed stream into state; the
//!   reference [`state_machine::KvStateMachine`] mirrors the queue/decision-log
//!   shape (idempotent, so leader-change retries are safe).
//!
//! ## Example
//! ```
//! use sophia_raft::{Cluster, KvStateMachine};
//! let mut cluster = Cluster::new(&[1, 2, 3], |_| KvStateMachine::new());
//! let leader = cluster.run_until_leader(50).expect("elects a leader");
//! cluster.propose("task:7=accepted");
//! cluster.settle(5);
//! // Every node applied the committed entry.
//! for id in [1, 2, 3] {
//!     assert_eq!(cluster.state_machine(id).get("task:7").map(String::as_str), Some("accepted"));
//! }
//! # let _ = leader;
//! ```

pub mod cluster;
pub mod log;
pub mod node;
pub mod state_machine;
pub mod types;

pub use cluster::Cluster;
pub use node::{RaftNode, Role};
pub use state_machine::{KvStateMachine, StateMachine};
pub use types::{Envelope, Index, LogEntry, Message, NodeId, Term};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elects_exactly_one_leader() {
        let mut c = Cluster::new(&[1, 2, 3], |_| KvStateMachine::new());
        let leader = c.run_until_leader(50).expect("a leader is elected");
        let leaders: Vec<NodeId> = [1, 2, 3].into_iter().filter(|&id| c.node(id).is_leader()).collect();
        assert_eq!(leaders, vec![leader], "exactly one leader");
    }

    #[test]
    fn replicates_committed_entries_to_all_nodes() {
        let mut c = Cluster::new(&[1, 2, 3], |_| KvStateMachine::new());
        c.run_until_leader(50).unwrap();
        c.propose("a=1");
        c.propose("b=2");
        c.settle(5);
        for id in [1, 2, 3] {
            let sm = c.state_machine(id);
            assert_eq!(sm.get("a").map(String::as_str), Some("1"), "node {id} missing a");
            assert_eq!(sm.get("b").map(String::as_str), Some("2"), "node {id} missing b");
        }
    }

    #[test]
    fn survives_leader_crash_and_keeps_committed_data() {
        let mut c = Cluster::new(&[1, 2, 3], |_| KvStateMachine::new());
        let leader = c.run_until_leader(50).unwrap();
        c.propose("before=crash");
        c.settle(5);

        // Crash the leader; the remaining two must elect a new one and serve.
        c.crash(leader);
        let new_leader = c.run_until_leader(50).expect("survivors elect a new leader");
        assert_ne!(new_leader, leader);

        c.propose("after=crash");
        c.settle(5);
        for id in [1, 2, 3] {
            if c.is_down(id) {
                continue;
            }
            assert_eq!(c.state_machine(id).get("before").map(String::as_str), Some("crash"));
            assert_eq!(c.state_machine(id).get("after").map(String::as_str), Some("crash"));
        }
    }

    #[test]
    fn minority_cannot_commit() {
        // 5 nodes (quorum 3); crash 3 → a 2-node minority survives. The surviving
        // leader stays leader (Raft leaders don't step down on unreachable peers),
        // but with no quorum it must NOT be able to commit a new entry.
        let mut c = Cluster::new(&[1, 2, 3, 4, 5], |_| KvStateMachine::new());
        c.run_until_leader(50).unwrap();
        c.crash(3);
        c.crash(4);
        c.crash(5);

        let idx = c.propose("nope=1"); // leader 1 survives, so this appends...
        c.settle(20); // ...but cannot reach a 3-node majority.
        for id in [1, 2] {
            assert!(c.commit_index(id) < idx.unwrap_or(1), "minority must not commit");
            assert_eq!(c.state_machine(id).get("nope"), None, "uncommitted entry applied");
        }
    }

    #[test]
    fn rejoining_follower_catches_up() {
        let mut c = Cluster::new(&[1, 2, 3], |_| KvStateMachine::new());
        c.run_until_leader(50).unwrap();
        // Partition node 3 out, commit entries with the 1+2 majority.
        c.crash(3);
        c.propose("x=10");
        c.propose("y=20");
        c.settle(5);
        // Node 3 rejoins and must be brought up to date by the leader.
        c.restart(3);
        c.settle(15);
        assert_eq!(c.state_machine(3).get("x").map(String::as_str), Some("10"));
        assert_eq!(c.state_machine(3).get("y").map(String::as_str), Some("20"));
    }
}
