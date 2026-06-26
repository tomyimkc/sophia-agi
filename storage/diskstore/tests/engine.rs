// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Durability, recovery, and compaction tests for the bitcask engine. Each test
//! uses a unique temp dir under the OS temp root and cleans up after itself.

use std::path::PathBuf;

use diskstore::{Bitcask, StdReader};

/// A throwaway directory; removed on drop. Name derived from a caller-supplied
/// tag plus the thread id to avoid collisions without needing `rand`.
struct TmpDir(PathBuf);
impl TmpDir {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        let tid = format!("{:?}", std::thread::current().id());
        let tid: String = tid.chars().filter(|c| c.is_alphanumeric()).collect();
        p.push(format!("diskstore-test-{tag}-{tid}"));
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

#[test]
fn put_get_delete() {
    let dir = TmpDir::new("basic");
    let mut db = Bitcask::open(dir.path(), true).unwrap();
    assert_eq!(db.get(b"missing").unwrap(), None);
    db.put(b"k", b"v1").unwrap();
    assert_eq!(db.get(b"k").unwrap(), Some(b"v1".to_vec()));
    db.put(b"k", b"v2-longer").unwrap(); // overwrite
    assert_eq!(db.get(b"k").unwrap(), Some(b"v2-longer".to_vec()));
    assert!(db.delete(b"k").unwrap());
    assert_eq!(db.get(b"k").unwrap(), None);
    assert!(!db.delete(b"k").unwrap());
}

#[test]
fn persists_across_reopen() {
    let dir = TmpDir::new("reopen");
    {
        let mut db = Bitcask::open(dir.path(), true).unwrap();
        for i in 0..500 {
            db.put(format!("k{i}").as_bytes(), format!("value-{i}").as_bytes()).unwrap();
        }
        db.delete(b"k7").unwrap();
        db.put(b"k3", b"updated").unwrap();
    }
    // Reopen: keydir must be rebuilt from the log alone.
    let db = Bitcask::open(dir.path(), true).unwrap();
    assert_eq!(db.len(), 499);
    assert_eq!(db.get(b"k3").unwrap(), Some(b"updated".to_vec()));
    assert_eq!(db.get(b"k7").unwrap(), None);
    assert_eq!(db.get(b"k499").unwrap(), Some(b"value-499".to_vec()));
}

#[test]
fn recovers_from_torn_tail() {
    let dir = TmpDir::new("torn");
    {
        let mut db = Bitcask::open(dir.path(), true).unwrap();
        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        db.put(b"c", b"3").unwrap();
    }
    // Simulate a crash mid-append: append garbage bytes to the log tail.
    let logp = dir.path().join("data.log");
    let good_len = std::fs::metadata(&logp).unwrap().len();
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&logp).unwrap();
        f.write_all(&[0xAB; 37]).unwrap(); // partial/garbage record
    }
    assert!(std::fs::metadata(&logp).unwrap().len() > good_len);

    // Recovery must drop the garbage and keep the three good records.
    let db = Bitcask::open(dir.path(), true).unwrap();
    assert_eq!(db.get(b"a").unwrap(), Some(b"1".to_vec()));
    assert_eq!(db.get(b"b").unwrap(), Some(b"2".to_vec()));
    assert_eq!(db.get(b"c").unwrap(), Some(b"3".to_vec()));
    assert_eq!(db.file_size(), good_len, "torn tail should have been truncated");
}

#[test]
fn compaction_reclaims_space_and_preserves_data() {
    let dir = TmpDir::new("compact");
    let mut db = Bitcask::open(dir.path(), false).unwrap();
    // Many overwrites of the same keys => lots of dead bytes.
    for _ in 0..50 {
        for i in 0..20 {
            db.put(format!("k{i}").as_bytes(), b"some-reasonably-sized-value-payload").unwrap();
        }
    }
    db.delete(b"k0").unwrap();
    db.sync().unwrap();
    let before = db.file_size();
    assert!(db.dead_bytes() > 0);

    db.compact().unwrap();
    let after = db.file_size();
    assert!(after < before, "compaction should shrink the log: {after} !< {before}");
    assert_eq!(db.dead_bytes(), 0);
    assert_eq!(db.len(), 19);
    assert_eq!(db.get(b"k0").unwrap(), None);
    assert_eq!(db.get(b"k19").unwrap(), Some(b"some-reasonably-sized-value-payload".to_vec()));

    // And it survives a reopen after compaction.
    drop(db);
    let db = Bitcask::open(dir.path(), false).unwrap();
    assert_eq!(db.len(), 19);
    assert_eq!(db.get(b"k10").unwrap(), Some(b"some-reasonably-sized-value-payload".to_vec()));
}

#[test]
fn multi_get_std_reader() {
    let dir = TmpDir::new("multiget");
    let mut db = Bitcask::open(dir.path(), false).unwrap();
    for i in 0..100 {
        db.put(format!("k{i}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
    }
    let keys: Vec<&[u8]> = vec![b"k0", b"k50", b"nope", b"k99"];
    let got = db.multi_get(&StdReader, &keys).unwrap();
    assert_eq!(got[0], Some(b"v0".to_vec()));
    assert_eq!(got[1], Some(b"v50".to_vec()));
    assert_eq!(got[2], None);
    assert_eq!(got[3], Some(b"v99".to_vec()));
}

#[cfg(feature = "io_uring")]
#[test]
fn multi_get_uring_matches_std() {
    use diskstore::UringReader;
    let dir = TmpDir::new("uring");
    let mut db = Bitcask::open(dir.path(), false).unwrap();
    for i in 0..1000 {
        db.put(format!("k{i}").as_bytes(), format!("value-payload-{i}").as_bytes()).unwrap();
    }
    let keys: Vec<String> = (0..1000).map(|i| format!("k{i}")).collect();
    let kref: Vec<&[u8]> = keys.iter().map(|s| s.as_bytes()).collect();

    let via_std = db.multi_get(&StdReader, &kref).unwrap();
    let uring = UringReader::new(256).unwrap();
    let via_uring = db.multi_get(&uring, &kref).unwrap();
    assert_eq!(via_std, via_uring, "io_uring reads must match pread reads exactly");
    assert_eq!(via_uring[42], Some(b"value-payload-42".to_vec()));
}
