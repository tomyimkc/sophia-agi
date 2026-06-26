// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! `raftkv` — durable Raft: the Phase-3 consensus core ([`miniraft`]) with its
//! persistent state backed by the Phase-2 storage engine ([`diskstore`]).
//!
//! Raft's correctness rests on a promise: *a completed write is durable*. A node
//! must record `current_term`, `voted_for`, and the log on stable storage before
//! acting on them, so that after a crash it recovers the exact state and never,
//! say, votes twice in one term or loses a committed entry. This crate fulfills
//! that promise by flushing each node's [`PersistentState`] to a per-node
//! `Bitcask` whenever it changes, and reloading it on restart.
//!
//! [`DurableCluster`] drives a [`miniraft::Sim`] and persists every node's dirty
//! state after each event, exposing `crash` (lose volatile state) and `restart`
//! (reload durable state from disk) so a full crash/recovery can be tested
//! end-to-end against real on-disk data.
//!
//! ```no_run
//! use raftkv::DurableCluster;
//! use miniraft::Timing;
//! # fn run() -> std::io::Result<()> {
//! let mut cluster = DurableCluster::new(3, "/tmp/raftkv-demo", Timing::default(), 10)?;
//! cluster.run_until(20_000, |s| s.leader().is_some())?;
//! let leader = cluster.sim().leader().unwrap();
//! cluster.propose(leader, b"set k=v".to_vec())?;          // appended + fsynced
//! cluster.crash(leader);                                  // volatile state lost
//! cluster.restart(leader)?;                               // term/vote/log reloaded
//! # Ok(()) }
//! ```

pub mod codec;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use diskstore::Bitcask;
use miniraft::{NodeId, PersistentState, Sim, Timing};

const STATE_KEY: &[u8] = b"raft/persistent-state";

pub struct DurableCluster {
    sim: Sim,
    stores: BTreeMap<NodeId, Bitcask>,
    n: usize,
    dir: PathBuf,
}

impl DurableCluster {
    /// Create an `n`-node cluster, each node's durable state in its own Bitcask
    /// under `dir/node-<id>`. fsync-on-write is enabled (Raft needs durability).
    pub fn new(n: usize, dir: impl AsRef<Path>, timing: Timing, latency: u64) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let mut stores = BTreeMap::new();
        for id in 0..n as NodeId {
            stores.insert(id, Bitcask::open(dir.join(format!("node-{id}")), true)?);
        }
        Ok(DurableCluster { sim: Sim::new(n, timing, latency), stores, n, dir })
    }

    pub fn sim(&self) -> &Sim {
        &self.sim
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Load a node's last-persisted state from disk, if any.
    pub fn load_state(&self, id: NodeId) -> std::io::Result<Option<PersistentState>> {
        Ok(self.stores[&id].get(STATE_KEY)?.and_then(|b| codec::decode(&b)))
    }

    /// Flush every node whose durable state changed since the last flush.
    fn persist_dirty(&mut self) -> std::io::Result<()> {
        for id in 0..self.n as NodeId {
            if self.sim.take_dirty(id) {
                let bytes = codec::encode(&self.sim.export(id));
                self.stores.get_mut(&id).unwrap().put(STATE_KEY, &bytes)?;
            }
        }
        Ok(())
    }

    /// Advance one event and persist any resulting durable-state changes.
    /// Returns false when the cluster is quiescent.
    pub fn step(&mut self) -> std::io::Result<bool> {
        let progressed = self.sim.step();
        self.persist_dirty()?;
        Ok(progressed)
    }

    /// Step until `pred(sim)` holds or `max_steps` is exhausted.
    pub fn run_until(&mut self, max_steps: usize, mut pred: impl FnMut(&Sim) -> bool) -> std::io::Result<bool> {
        for _ in 0..max_steps {
            if pred(&self.sim) {
                return Ok(true);
            }
            if !self.step()? {
                return Ok(pred(&self.sim));
            }
        }
        Ok(pred(&self.sim))
    }

    /// Propose on `id`; persists the resulting log append. Returns whether the
    /// node accepted it as leader.
    pub fn propose(&mut self, id: NodeId, command: Vec<u8>) -> std::io::Result<bool> {
        let accepted = self.sim.propose(id, command);
        self.persist_dirty()?;
        Ok(accepted)
    }

    /// Crash a node: it stops processing and loses all volatile state. Its
    /// durable state remains on disk (already persisted by prior steps).
    pub fn crash(&mut self, id: NodeId) {
        self.sim.crash(id);
    }

    /// Restart a crashed node, reloading its durable state from disk (term,
    /// vote, log) and resetting volatile state — a faithful process restart.
    pub fn restart(&mut self, id: NodeId) -> std::io::Result<()> {
        match self.load_state(id)? {
            Some(state) => self.sim.restart_from(id, state),
            None => self.sim.restart(id),
        }
        Ok(())
    }
}
