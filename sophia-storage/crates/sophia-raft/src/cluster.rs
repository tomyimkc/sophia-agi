// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Deterministic in-memory cluster harness.
//!
//! Owns N [`RaftNode`]s and a state machine per node, routes messages between
//! them, and drives logical time. No threads, no real network — so elections,
//! replication, partitions, and leader crashes are all reproducible. This is the
//! test substrate; a production deployment swaps this for a real transport
//! (TCP/gRPC) and a durable log (an `sophia-lsm` engine per node), keeping the
//! `RaftNode` core unchanged.

use std::collections::{BTreeMap, HashSet, VecDeque};

use crate::node::RaftNode;
use crate::state_machine::StateMachine;
use crate::types::{Envelope, Index, NodeId};

pub struct Cluster<SM: StateMachine> {
    nodes: BTreeMap<NodeId, RaftNode>,
    sms: BTreeMap<NodeId, SM>,
    inflight: VecDeque<Envelope>,
    /// Crashed / partitioned nodes: they neither tick nor receive messages.
    down: HashSet<NodeId>,
}

impl<SM: StateMachine> Cluster<SM> {
    /// Build an `n`-node cluster. `make_sm` builds each node's state machine.
    /// Election timeouts are staggered by id so the cluster converges instead of
    /// livelocking on split votes (the deterministic stand-in for randomized
    /// timeouts).
    pub fn new(ids: &[NodeId], mut make_sm: impl FnMut(NodeId) -> SM) -> Self {
        let mut nodes = BTreeMap::new();
        let mut sms = BTreeMap::new();
        for (i, &id) in ids.iter().enumerate() {
            let peers: Vec<NodeId> = ids.iter().copied().filter(|&p| p != id).collect();
            // Distinct election timeouts (10, 13, 16, …); heartbeat well below.
            let election = 10 + (i as u32) * 3;
            nodes.insert(id, RaftNode::new(id, peers, election, 3));
            sms.insert(id, make_sm(id));
        }
        Cluster { nodes, sms, inflight: VecDeque::new(), down: HashSet::new() }
    }

    pub fn is_down(&self, id: NodeId) -> bool {
        self.down.contains(&id)
    }

    /// Crash a node: it stops ticking and drops messages until restarted.
    pub fn crash(&mut self, id: NodeId) {
        self.down.insert(id);
    }

    /// Bring a crashed node back (its log/term persist; volatile leader state
    /// resets via the normal follower path on the next valid AppendEntries).
    pub fn restart(&mut self, id: NodeId) {
        self.down.remove(&id);
    }

    /// One tick of logical time for every live node; queues any messages.
    pub fn tick(&mut self) {
        for (&id, node) in self.nodes.iter_mut() {
            if self.down.contains(&id) {
                continue;
            }
            for env in node.tick() {
                self.inflight.push_back(env);
            }
        }
    }

    /// Deliver every in-flight message (and the replies they generate) until the
    /// network is quiet, applying newly-committed entries after each step.
    /// Messages to/from down nodes are dropped.
    pub fn deliver_all(&mut self) {
        let mut budget = 100_000; // safety bound against a logic bug looping forever
        while let Some(env) = self.inflight.pop_front() {
            budget -= 1;
            if budget == 0 {
                panic!("deliver_all exceeded message budget — possible Raft livelock");
            }
            if self.down.contains(&env.to) || self.down.contains(&env.from) {
                continue;
            }
            if let Some(node) = self.nodes.get_mut(&env.to) {
                for reply in node.step(env) {
                    self.inflight.push_back(reply);
                }
            }
            self.apply_committed();
        }
    }

    /// Apply any newly-committed entries on every live node to its state machine.
    fn apply_committed(&mut self) {
        for (&id, node) in self.nodes.iter_mut() {
            if self.down.contains(&id) {
                continue;
            }
            while node.log.last_applied < node.log.commit_index {
                let next = node.log.last_applied + 1;
                if let Some(entry) = node.log.entry(next) {
                    self.sms.get_mut(&id).unwrap().apply(&entry.command);
                }
                node.log.last_applied = next;
            }
        }
    }

    /// Tick + deliver repeatedly until a leader exists or `max_ticks` elapse.
    pub fn run_until_leader(&mut self, max_ticks: u32) -> Option<NodeId> {
        for _ in 0..max_ticks {
            self.tick();
            self.deliver_all();
            if let Some(l) = self.leader() {
                return Some(l);
            }
        }
        None
    }

    /// Settle the cluster (deliver messages + a few heartbeat rounds) so commits
    /// propagate after a proposal.
    pub fn settle(&mut self, rounds: u32) {
        for _ in 0..rounds {
            self.tick();
            self.deliver_all();
        }
    }

    /// The current leader, if exactly one node believes it leads (highest term
    /// wins if a stale leader hasn't stepped down yet).
    pub fn leader(&self) -> Option<NodeId> {
        self.nodes
            .values()
            .filter(|n| n.is_leader() && !self.down.contains(&n.id))
            .max_by_key(|n| n.current_term)
            .map(|n| n.id)
    }

    /// Propose a command to the current leader. Returns the entry index, or
    /// `None` if there is no leader.
    pub fn propose(&mut self, command: impl Into<Vec<u8>>) -> Option<Index> {
        let leader = self.leader()?;
        let (index, out) = self.nodes.get_mut(&leader)?.propose(command.into())?;
        for env in out {
            self.inflight.push_back(env);
        }
        self.deliver_all();
        Some(index)
    }

    pub fn state_machine(&self, id: NodeId) -> &SM {
        &self.sms[&id]
    }

    pub fn commit_index(&self, id: NodeId) -> Index {
        self.nodes[&id].log.commit_index
    }

    pub fn node(&self, id: NodeId) -> &RaftNode {
        &self.nodes[&id]
    }
}
