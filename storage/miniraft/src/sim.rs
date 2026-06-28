// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Deterministic, event-driven cluster simulator.
//!
//! Drives N [`RaftNode`]s in one thread with a logical clock. It advances time
//! to the next event (a message delivery or a node timeout), never wall-clock,
//! so runs are fully reproducible. It can drop a node (crash), restart it, and
//! install network partitions — the levers needed to test the hard safety
//! properties (no two leaders per term, majority-only commit, log convergence).

use std::collections::{BTreeMap, HashSet};

use crate::node::{RaftNode, Timing};
use crate::types::{Envelope, NodeId, PersistentState, Role};

struct Scheduled {
    deliver_at: u64,
    seq: u64,
    env: Envelope,
}

pub struct Sim {
    nodes: BTreeMap<NodeId, RaftNode>,
    now: u64,
    latency: u64,
    queue: Vec<Scheduled>,
    seq: u64,
    down: HashSet<NodeId>,
    /// Partition groups; empty means the network is fully connected.
    partition: Vec<HashSet<NodeId>>,
}

impl Sim {
    /// Build a fresh cluster of `n` nodes (ids `0..n`) with the given timing and
    /// one-way message `latency`.
    pub fn new(n: usize, timing: Timing, latency: u64) -> Self {
        let ids: Vec<NodeId> = (0..n as NodeId).collect();
        let mut nodes = BTreeMap::new();
        for &id in &ids {
            let peers = ids.iter().copied().filter(|&p| p != id).collect();
            nodes.insert(id, RaftNode::new(id, peers, timing, 0));
        }
        Sim { nodes, now: 0, latency, queue: Vec::new(), seq: 0, down: HashSet::new(), partition: Vec::new() }
    }

    pub fn now(&self) -> u64 {
        self.now
    }

    pub fn node(&self, id: NodeId) -> &RaftNode {
        &self.nodes[&id]
    }

    // --- fault injection ---

    pub fn crash(&mut self, id: NodeId) {
        self.down.insert(id);
    }

    pub fn restart(&mut self, id: NodeId) {
        self.down.remove(&id);
        let now = self.now;
        if let Some(n) = self.nodes.get_mut(&id) {
            n.restart(now);
        }
    }

    // --- persistence hooks (used by a durable driver, e.g. the `raftkv` crate) ---

    /// Whether node `id`'s durable state changed since last checked (clears flag).
    pub fn take_dirty(&mut self, id: NodeId) -> bool {
        self.nodes.get_mut(&id).is_some_and(|n| n.take_dirty())
    }

    /// Snapshot node `id`'s durable state, for writing to stable storage.
    pub fn export(&self, id: NodeId) -> PersistentState {
        self.nodes[&id].export()
    }

    /// Bring a crashed node back up with durable state reloaded from disk:
    /// overwrite term/vote/log, then reset volatile state (as a real restart does).
    pub fn restart_from(&mut self, id: NodeId, state: PersistentState) {
        self.down.remove(&id);
        let now = self.now;
        if let Some(n) = self.nodes.get_mut(&id) {
            n.restore_state(state);
            n.restart(now);
        }
    }

    /// Split the cluster into isolated groups. Nodes in different groups cannot
    /// exchange messages. `heal()` restores full connectivity.
    pub fn partition(&mut self, groups: Vec<Vec<NodeId>>) {
        self.partition = groups.into_iter().map(|g| g.into_iter().collect()).collect();
    }

    pub fn heal(&mut self) {
        self.partition.clear();
    }

    fn reachable(&self, a: NodeId, b: NodeId) -> bool {
        if self.partition.is_empty() {
            return true;
        }
        self.partition.iter().any(|g| g.contains(&a) && g.contains(&b))
    }

    fn deliverable(&self, env: &Envelope) -> bool {
        !self.down.contains(&env.from) && !self.down.contains(&env.to) && self.reachable(env.from, env.to)
    }

    fn schedule(&mut self, envs: Vec<Envelope>) {
        for env in envs {
            self.seq += 1;
            self.queue.push(Scheduled { deliver_at: self.now + self.latency, seq: self.seq, env });
        }
    }

    // --- client ---

    /// Propose on a specific node; returns true if it accepted as leader.
    pub fn propose(&mut self, id: NodeId, command: Vec<u8>) -> bool {
        if self.down.contains(&id) {
            return false;
        }
        let (idx, msgs) = self.nodes.get_mut(&id).unwrap().propose(command);
        self.schedule(msgs);
        idx.is_some()
    }

    /// The current leader, if exactly one node believes it leads the latest term.
    pub fn leader(&self) -> Option<NodeId> {
        let max_term = self.nodes.values().filter(|n| !self.down.contains(&n.id())).map(|n| n.term()).max()?;
        let leaders: Vec<NodeId> = self
            .nodes
            .values()
            .filter(|n| !self.down.contains(&n.id()) && n.role() == Role::Leader && n.term() == max_term)
            .map(|n| n.id())
            .collect();
        if leaders.len() == 1 {
            Some(leaders[0])
        } else {
            None
        }
    }

    /// Count of live nodes currently in the leader role (for split-brain assertions).
    pub fn leaders_in_term(&self, term: u64) -> usize {
        self.nodes
            .values()
            .filter(|n| !self.down.contains(&n.id()) && n.role() == Role::Leader && n.term() == term)
            .count()
    }

    // --- driving the clock ---

    /// Advance to the next event and process it. Returns false when the cluster
    /// is quiescent (no pending messages and no pending timeouts).
    pub fn step(&mut self) -> bool {
        let next_msg = self.queue.iter().map(|s| s.deliver_at).min();
        let next_timer = self
            .nodes
            .values()
            .filter(|n| !self.down.contains(&n.id()))
            .map(|n| n.next_deadline())
            .min();
        let next = match (next_msg, next_timer) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => return false,
        };
        // Don't get stuck if a timer is already in the past relative to `now`.
        self.now = next.max(self.now);

        // 1) Deliver all messages due now (stable order by seq).
        let mut due: Vec<Scheduled> = Vec::new();
        let mut rest: Vec<Scheduled> = Vec::new();
        for s in self.queue.drain(..) {
            if s.deliver_at <= self.now {
                due.push(s);
            } else {
                rest.push(s);
            }
        }
        self.queue = rest;
        due.sort_by_key(|s| s.seq);
        let now = self.now;
        for s in due {
            if !self.deliverable(&s.env) {
                continue; // dropped by crash/partition
            }
            let to = s.env.to;
            let out = self.nodes.get_mut(&to).unwrap().recv(now, s.env.from, s.env.msg);
            self.schedule(out);
        }

        // 2) Fire any due timers.
        let timer_ids: Vec<NodeId> = self
            .nodes
            .values()
            .filter(|n| !self.down.contains(&n.id()) && n.next_deadline() <= now)
            .map(|n| n.id())
            .collect();
        for id in timer_ids {
            let out = self.nodes.get_mut(&id).unwrap().tick(now);
            self.schedule(out);
        }
        true
    }

    /// Step until `pred` holds or `max_steps` is exhausted. Returns whether
    /// `pred` was satisfied.
    pub fn run_until(&mut self, max_steps: usize, mut pred: impl FnMut(&Sim) -> bool) -> bool {
        for _ in 0..max_steps {
            if pred(self) {
                return true;
            }
            if !self.step() {
                // Quiescent: a steady state. Check the predicate one last time.
                return pred(self);
            }
        }
        pred(self)
    }

    /// Run the cluster forward by stepping `n` times (or until quiescent).
    pub fn run_steps(&mut self, n: usize) {
        for _ in 0..n {
            if !self.step() {
                break;
            }
        }
    }
}
