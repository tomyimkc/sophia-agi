// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! In-memory write buffer: a sorted map of the most recent value per key.
//!
//! Tombstones are represented as `None` so a delete shadows an older on-disk
//! value until compaction reaps it. Kept ordered (BTreeMap) so a flush writes a
//! sorted SSTable in one pass with no extra sort.

use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct MemTable {
    map: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    bytes: usize,
}

impl MemTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.account(&key, Some(value.len()));
        self.map.insert(key, Some(value));
    }

    pub fn delete(&mut self, key: Vec<u8>) {
        self.account(&key, None);
        self.map.insert(key, None);
    }

    fn account(&mut self, key: &[u8], new_val_len: Option<usize>) {
        // Rough live-bytes estimate driving the flush threshold.
        if let Some(prev) = self.map.get(key) {
            self.bytes -= key.len() + prev.as_ref().map_or(0, |v| v.len());
        }
        self.bytes += key.len() + new_val_len.unwrap_or(0);
    }

    /// `Some(None)` = known tombstone, `Some(Some(v))` = value, `None` = absent here.
    pub fn get(&self, key: &[u8]) -> Option<&Option<Vec<u8>>> {
        self.map.get(key)
    }

    pub fn approx_bytes(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Drain in sorted order for a flush.
    pub fn drain_sorted(&mut self) -> BTreeMap<Vec<u8>, Option<Vec<u8>>> {
        self.bytes = 0;
        std::mem::take(&mut self.map)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &Option<Vec<u8>>)> {
        self.map.iter()
    }
}
