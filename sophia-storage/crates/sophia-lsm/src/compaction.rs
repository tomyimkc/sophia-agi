// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Compaction: merge several SSTables into one, newest value wins, tombstones
//! reaped when no older table can shadow them.
//!
//! This is a simple full merge (all levels at once), which is correct and easy
//! to reason about. The leveled / size-tiered policy that bounds write
//! amplification is described in DESIGN.md as the next iteration — the merge
//! primitive here is what both policies are built from.

use std::collections::BTreeMap;

use crate::record::Record;

/// Merge ordered runs into a single sorted map. `runs` must be ordered
/// **newest first** so the first writer of a key wins.
pub fn merge(runs: Vec<Vec<Record>>) -> BTreeMap<Vec<u8>, Option<Vec<u8>>> {
    let mut merged: BTreeMap<Vec<u8>, Option<Vec<u8>>> = BTreeMap::new();
    for run in runs {
        for rec in run {
            // Only insert if no newer run already set this key.
            merged.entry(rec.key.clone()).or_insert_with(|| {
                if rec.is_tombstone() { None } else { Some(rec.value.clone()) }
            });
        }
    }
    merged
}

/// As [`merge`], but drops tombstones — safe only when these runs are the
/// oldest data in the engine (nothing below can resurrect a deleted key).
pub fn merge_and_reap(runs: Vec<Vec<Record>>) -> BTreeMap<Vec<u8>, Option<Vec<u8>>> {
    let mut merged = merge(runs);
    merged.retain(|_, v| v.is_some());
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newest_run_wins() {
        let newest = vec![Record::put(b"k".to_vec(), b"new".to_vec())];
        let oldest = vec![Record::put(b"k".to_vec(), b"old".to_vec())];
        let m = merge(vec![newest, oldest]);
        assert_eq!(m[b"k".as_slice()], Some(b"new".to_vec()));
    }

    #[test]
    fn reap_drops_tombstones() {
        let newest = vec![Record::delete(b"k".to_vec())];
        let oldest = vec![Record::put(b"k".to_vec(), b"v".to_vec())];
        assert!(!merge_and_reap(vec![newest, oldest]).contains_key(b"k".as_slice()));
    }
}
