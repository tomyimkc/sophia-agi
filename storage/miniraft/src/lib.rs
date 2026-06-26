// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! `miniraft` — a clean-room implementation of the Raft consensus core, with a
//! deterministic simulator for verifying its safety properties.
//!
//! Phase 3 of the Sophia distributed-storage roadmap
//! (`../../docs/storage/STORAGE_ROADMAP.md`). The roadmap's own advice was
//! "reproduce Raft — that one artifact beats ten read papers", and the job
//! description explicitly values solving from first principles over gluing
//! libraries together. So this is Raft built from the paper (Ongaro &
//! Ousterhout, *In Search of an Understandable Consensus Algorithm*, fig. 2),
//! not a wrapper over an existing crate.
//!
//! **In scope (and tested):** leader election with randomized timeouts and the
//! up-to-date-log voting restriction; log replication with the consistency
//! check and conflict truncation; the commit rule (a leader commits an entry
//! only once it is on a quorum *and* from the leader's current term); safety
//! under crashes and network partitions.
//!
//! **Out of scope (documented, not faked):** snapshotting/log compaction,
//! dynamic membership changes, and persistence to disk (the node marks which
//! state is persistent; wiring it to [`diskstore`](../diskstore) is the
//! productionization step, as is swapping this core under a real transport or
//! integrating `openraft`).
//!
//! ```
//! use miniraft::{Sim, Timing};
//! let mut sim = Sim::new(3, Timing::default(), 10);
//! // A leader emerges, a write replicates to a quorum and commits everywhere.
//! sim.run_until(10_000, |s| s.leader().is_some());
//! let leader = sim.leader().unwrap();
//! sim.propose(leader, b"set x=1".to_vec());
//! sim.run_until(10_000, |s| (0..3).all(|i| !s.node(i).applied().is_empty()));
//! ```

pub mod node;
pub mod sim;
pub mod types;

pub use node::{RaftNode, Timing};
pub use sim::Sim;
pub use types::{Envelope, Index, LogEntry, Msg, NodeId, PersistentState, Role, Term};
