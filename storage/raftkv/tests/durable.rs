// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! End-to-end durability tests: Raft state survives crashes by round-tripping
//! through the diskstore engine, and a recovered node rejoins and converges.

use std::path::PathBuf;

use miniraft::{Role, Sim, Timing};
use raftkv::DurableCluster;

struct TmpDir(PathBuf);
impl TmpDir {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        let tid: String = format!("{:?}", std::thread::current().id()).chars().filter(|c| c.is_alphanumeric()).collect();
        p.push(format!("raftkv-test-{tag}-{tid}"));
        let _ = std::fs::remove_dir_all(&p);
        TmpDir(p)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn timing() -> Timing {
    Timing { election_min: 150, election_max: 300, heartbeat: 30 }
}

#[test]
fn committed_log_is_persisted_to_disk() {
    let dir = TmpDir::new("persist");
    let mut cluster = DurableCluster::new(3, dir.path(), timing(), 10).unwrap();
    assert!(cluster.run_until(20_000, |s| s.leader().is_some()).unwrap());
    let leader = cluster.sim().leader().unwrap();

    for i in 0..6 {
        assert!(cluster.propose(leader, format!("cmd-{i}").into_bytes()).unwrap());
    }
    cluster.run_until(50_000, |s| (0..3).all(|id| s.node(id).applied().len() == 6)).unwrap();

    // The leader's durable state on disk must reflect all 6 entries.
    let on_disk = cluster.load_state(leader).unwrap().expect("leader state persisted");
    assert_eq!(on_disk.log.len(), 6);
    assert_eq!(on_disk.log[0].command, b"cmd-0");
    assert_eq!(on_disk.log[5].command, b"cmd-5");
    assert!(on_disk.current_term >= 1);
    // A leader voted for itself in its election term.
    assert!(on_disk.voted_for.is_some());
}

#[test]
fn follower_recovers_log_from_disk_after_crash() {
    let dir = TmpDir::new("recover");
    let mut cluster = DurableCluster::new(5, dir.path(), timing(), 10).unwrap();
    assert!(cluster.run_until(20_000, |s| s.leader().is_some()).unwrap());
    let leader = cluster.sim().leader().unwrap();
    let follower = (0..5).find(|&i| i != leader).unwrap();

    for i in 0..5 {
        cluster.propose(leader, format!("v{i}").into_bytes()).unwrap();
    }
    cluster.run_until(50_000, |s| s.node(follower).applied().len() == 5).unwrap();
    let pre_crash = cluster.load_state(follower).unwrap().expect("follower persisted");
    assert_eq!(pre_crash.log.len(), 5);

    // Crash the follower (volatile state gone) then restart it from disk.
    cluster.crash(follower);
    cluster.restart(follower).unwrap();

    // After reload it holds its durable log and rejoins; everyone still agrees.
    let converged = cluster
        .run_until(50_000, |s| {
            let reference = s.node(leader).log_terms();
            (0..5).all(|id| s.node(id).log_terms() == reference)
        })
        .unwrap();
    assert!(converged, "recovered follower did not converge");
}

#[test]
fn restarted_leader_does_not_cause_split_brain() {
    let dir = TmpDir::new("leadercrash");
    let mut cluster = DurableCluster::new(5, dir.path(), timing(), 10).unwrap();
    assert!(cluster.run_until(20_000, |s| s.leader().is_some()).unwrap());
    let old_leader = cluster.sim().leader().unwrap();
    let old_term = cluster.sim().node(old_leader).term();

    cluster.propose(old_leader, b"pre".to_vec()).unwrap();
    cluster.run_until(50_000, |s| (0..5).filter(|&i| i != old_leader).all(|i| !s.node(i).applied().is_empty())).unwrap();

    // Crash the leader; the rest must elect a new one in a higher term.
    cluster.crash(old_leader);
    assert!(cluster
        .run_until(50_000, |s| s.leader().is_some_and(|l| l != old_leader && s.node(l).term() > old_term))
        .unwrap());

    // Restart the old leader from disk. With its persisted term/vote it must
    // recognize the newer leader and step down — never a second leader per term.
    cluster.restart(old_leader).unwrap();
    let new_leader = cluster.sim().leader().unwrap();
    cluster.propose(new_leader, b"post".to_vec()).unwrap();

    let converged = cluster
        .run_until(80_000, |s| {
            let reference = s.node(new_leader).log_terms();
            (0..5).all(|id| s.node(id).log_terms() == reference) && no_split_brain(s)
        })
        .unwrap();
    assert!(converged, "cluster failed to converge after leader recovery");
    // The recovered old leader must not still think it leads.
    assert!(cluster.sim().node(old_leader).role() != Role::Leader || cluster.sim().leader() == Some(old_leader));
}

fn no_split_brain(s: &Sim) -> bool {
    use std::collections::HashMap;
    let mut by_term: HashMap<u64, u64> = HashMap::new();
    for id in 0..5u64 {
        let node = s.node(id);
        if node.role() == Role::Leader && by_term.insert(node.term(), id).is_some() {
            return false;
        }
    }
    true
}
