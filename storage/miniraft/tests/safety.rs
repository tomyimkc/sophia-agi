// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Raft safety-property tests over the deterministic simulator. These are the
//! properties that distinguish a real consensus implementation from a sketch:
//! single leadership per term, quorum-only commit, durability across leader
//! crashes, no split-brain under partition, and log convergence after heal.

use miniraft::{NodeId, Role, Sim, Timing};

fn timing() -> Timing {
    Timing { election_min: 150, election_max: 300, heartbeat: 30 }
}

/// Invariant that must hold at every step: no two leaders share a term.
fn assert_no_split_brain(sim: &Sim, n: usize) {
    use std::collections::HashMap;
    let mut leader_term: HashMap<u64, NodeId> = HashMap::new();
    for id in 0..n as NodeId {
        let node = sim.node(id);
        if node.role() == Role::Leader {
            if let Some(&other) = leader_term.get(&node.term()) {
                panic!("two leaders in term {}: {} and {}", node.term(), other, id);
            }
            leader_term.insert(node.term(), id);
        }
    }
}

#[test]
fn elects_exactly_one_leader() {
    let mut sim = Sim::new(5, timing(), 10);
    assert!(sim.run_until(20_000, |s| s.leader().is_some()), "no leader elected");
    let leader = sim.leader().unwrap();
    // Run a while longer; leadership must remain unique per term throughout.
    for _ in 0..2000 {
        assert_no_split_brain(&sim, 5);
        if !sim.step() {
            break;
        }
    }
    assert_eq!(sim.node(leader).role(), Role::Leader);
}

#[test]
fn replicates_and_commits_on_quorum() {
    let mut sim = Sim::new(5, timing(), 10);
    assert!(sim.run_until(20_000, |s| s.leader().is_some()));
    let leader = sim.leader().unwrap();

    for i in 0..10 {
        assert!(sim.propose(leader, format!("set k{i}={i}").into_bytes()));
    }
    // Every node applies all 10 commands, in the same order.
    let done = sim.run_until(50_000, |s| (0..5).all(|id| s.node(id).applied().len() == 10));
    assert!(done, "commands did not fully replicate");

    let reference: Vec<Vec<u8>> = sim.node(leader).applied().iter().map(|(_, c)| c.clone()).collect();
    for id in 0..5u64 {
        let got: Vec<Vec<u8>> = sim.node(id).applied().iter().map(|(_, c)| c.clone()).collect();
        assert_eq!(got, reference, "node {id} diverged");
    }
}

#[test]
fn re_elects_after_leader_crash() {
    let mut sim = Sim::new(5, timing(), 10);
    assert!(sim.run_until(20_000, |s| s.leader().is_some()));
    let old_leader = sim.leader().unwrap();
    let old_term = sim.node(old_leader).term();

    sim.propose(old_leader, b"before-crash".to_vec());
    sim.run_until(50_000, |s| (0..5).filter(|&i| i != old_leader).all(|i| !s.node(i).applied().is_empty()));

    // Crash the leader. The remaining 4 (a quorum) must elect a new leader.
    sim.crash(old_leader);
    let elected = sim.run_until(50_000, |s| {
        s.leader().is_some_and(|l| l != old_leader && s.node(l).term() > old_term)
    });
    assert!(elected, "cluster failed to recover leadership after crash");

    let new_leader = sim.leader().unwrap();
    assert_ne!(new_leader, old_leader);
    // The new leader can still make progress.
    assert!(sim.propose(new_leader, b"after-crash".to_vec()));
    let committed = sim.run_until(50_000, |s| {
        (0..5).filter(|&i| i != old_leader).all(|i| s.node(i).applied().iter().any(|(_, c)| c == b"after-crash"))
    });
    assert!(committed, "new leader could not commit after recovery");
}

#[test]
fn minority_partition_cannot_commit() {
    let mut sim = Sim::new(5, timing(), 10);
    assert!(sim.run_until(20_000, |s| s.leader().is_some()));
    let leader = sim.leader().unwrap();

    // Partition so the leader is isolated with one follower (minority of 2),
    // leaving a 3-node majority on the other side.
    let others: Vec<NodeId> = (0..5).filter(|&i| i != leader).collect();
    let minority = vec![leader, others[0]];
    let majority = vec![others[1], others[2], others[3]];
    sim.partition(vec![minority.clone(), majority.clone()]);

    // The isolated old leader's writes must NOT commit (no quorum reachable).
    sim.propose(leader, b"doomed-write".to_vec());
    sim.run_steps(5000);
    for &id in &minority {
        assert!(
            !sim.node(id).applied().iter().any(|(_, c)| c == b"doomed-write"),
            "minority side committed without a quorum (split-brain write)"
        );
    }

    // The majority side elects its own leader and commits.
    let maj_leader = sim.run_until(50_000, |s| {
        majority.iter().any(|&id| s.node(id).role() == Role::Leader)
    });
    assert!(maj_leader, "majority failed to elect a leader");
    let new_leader = *majority.iter().find(|&&id| sim.node(id).role() == Role::Leader).unwrap();
    assert!(sim.propose(new_leader, b"majority-write".to_vec()));
    let maj_committed = sim.run_until(50_000, |s| {
        majority.iter().all(|&id| s.node(id).applied().iter().any(|(_, c)| c == b"majority-write"))
    });
    assert!(maj_committed, "majority side could not commit");

    // Heal. The old minority must converge to the majority's log and drop the
    // uncommitted doomed write.
    sim.heal();
    let converged = sim.run_until(100_000, |s| {
        let reference = s.node(new_leader).log_terms();
        (0..5).all(|id| s.node(id).log_terms() == reference)
    });
    assert!(converged, "logs did not converge after heal");
    for id in 0..5u64 {
        assert!(
            !sim.node(id).applied().iter().any(|(_, c)| c == b"doomed-write"),
            "doomed write survived on node {id} after convergence"
        );
        assert!(
            sim.node(id).applied().iter().any(|(_, c)| c == b"majority-write"),
            "node {id} missing the committed majority write"
        );
    }
}

#[test]
fn logs_never_disagree_on_committed_prefix() {
    // Property check from the paper: if two logs contain an entry at the same
    // index+term, the logs are identical in all preceding entries. We approximate
    // by asserting all applied prefixes are pairwise consistent throughout a run
    // with proposals interleaved.
    let mut sim = Sim::new(3, timing(), 8);
    assert!(sim.run_until(20_000, |s| s.leader().is_some()));
    for round in 0..15 {
        if let Some(l) = sim.leader() {
            sim.propose(l, format!("op-{round}").into_bytes());
        }
        sim.run_steps(400);

        // No two nodes may have conflicting commands at the same applied index.
        for a in 0..3u64 {
            for b in (a + 1)..3 {
                let la = sim.node(a).applied();
                let lb = sim.node(b).applied();
                let common = la.len().min(lb.len());
                assert_eq!(&la[..common], &lb[..common], "applied prefixes diverged at round {round}");
            }
        }
    }
}
