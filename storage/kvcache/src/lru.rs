// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! O(1) LRU map with per-entry TTL.
//!
//! Backed by a slab of nodes wired into an intrusive doubly-linked list, so
//! `get`, `insert`, and eviction are all O(1) (no scanning for the victim).
//! Not thread-safe by itself — `ShardedCache` owns one `Lru` per shard behind a
//! `Mutex`, which keeps lock contention proportional to 1/num_shards.

use std::collections::HashMap;
use std::time::Instant;

const NIL: usize = usize::MAX;

struct Node {
    key: Vec<u8>,
    val: Vec<u8>,
    expire_at: Option<Instant>,
    prev: usize, // toward head (MRU)
    next: usize, // toward tail (LRU)
}

/// Least-recently-used map. `head` is the most-recently-used end; `tail` is the
/// eviction victim. Freed slots are recycled via `free` to avoid reallocation.
pub struct Lru {
    cap: usize,
    map: HashMap<Vec<u8>, usize>,
    nodes: Vec<Node>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
    pub evictions: u64,
    pub expirations: u64,
}

impl Lru {
    pub fn new(cap: usize) -> Self {
        Lru {
            cap: cap.max(1),
            map: HashMap::new(),
            nodes: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            evictions: 0,
            expirations: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Fetch a key, refreshing its recency. Expired entries are evicted lazily
    /// on access and reported as a miss.
    pub fn get(&mut self, key: &[u8], now: Instant) -> Option<Vec<u8>> {
        let idx = match self.map.get(key) {
            Some(&i) => i,
            None => return None,
        };
        if let Some(exp) = self.nodes[idx].expire_at {
            if exp <= now {
                self.remove_idx(idx);
                self.expirations += 1;
                return None;
            }
        }
        self.detach(idx);
        self.push_front(idx);
        Some(self.nodes[idx].val.clone())
    }

    /// Insert or overwrite. `expire_at == None` means no TTL. Evicts the LRU
    /// entry first if inserting a new key would exceed capacity.
    pub fn insert(&mut self, key: Vec<u8>, val: Vec<u8>, expire_at: Option<Instant>) {
        if let Some(&idx) = self.map.get(&key) {
            self.nodes[idx].val = val;
            self.nodes[idx].expire_at = expire_at;
            self.detach(idx);
            self.push_front(idx);
            return;
        }
        if self.map.len() >= self.cap && self.tail != NIL {
            self.remove_idx(self.tail);
            self.evictions += 1;
        }
        let idx = self.alloc(Node {
            key: key.clone(),
            val,
            expire_at,
            prev: NIL,
            next: NIL,
        });
        self.map.insert(key, idx);
        self.push_front(idx);
    }

    pub fn remove(&mut self, key: &[u8]) -> bool {
        match self.map.get(key) {
            Some(&idx) => {
                self.remove_idx(idx);
                true
            }
            None => false,
        }
    }

    // --- internals ---

    fn alloc(&mut self, node: Node) -> usize {
        if let Some(i) = self.free.pop() {
            self.nodes[i] = node;
            i
        } else {
            self.nodes.push(node);
            self.nodes.len() - 1
        }
    }

    fn detach(&mut self, idx: usize) {
        let (prev, next) = (self.nodes[idx].prev, self.nodes[idx].next);
        if prev != NIL {
            self.nodes[prev].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.nodes[next].prev = prev;
        } else {
            self.tail = prev;
        }
        self.nodes[idx].prev = NIL;
        self.nodes[idx].next = NIL;
    }

    fn push_front(&mut self, idx: usize) {
        self.nodes[idx].prev = NIL;
        self.nodes[idx].next = self.head;
        if self.head != NIL {
            self.nodes[self.head].prev = idx;
        }
        self.head = idx;
        if self.tail == NIL {
            self.tail = idx;
        }
    }

    fn remove_idx(&mut self, idx: usize) {
        self.detach(idx);
        // Take the key out to drop the map entry; clear the slot for reuse.
        let key = std::mem::take(&mut self.nodes[idx].key);
        self.nodes[idx].val = Vec::new();
        self.nodes[idx].expire_at = None;
        self.map.remove(&key);
        self.free.push(idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn evicts_least_recently_used() {
        let now = Instant::now();
        let mut lru = Lru::new(2);
        lru.insert(b"a".to_vec(), b"1".to_vec(), None);
        lru.insert(b"b".to_vec(), b"2".to_vec(), None);
        // Touch "a" so "b" becomes the victim.
        assert_eq!(lru.get(b"a", now), Some(b"1".to_vec()));
        lru.insert(b"c".to_vec(), b"3".to_vec(), None);
        assert_eq!(lru.get(b"b", now), None);
        assert_eq!(lru.get(b"a", now), Some(b"1".to_vec()));
        assert_eq!(lru.get(b"c", now), Some(b"3".to_vec()));
        assert_eq!(lru.evictions, 1);
    }

    #[test]
    fn ttl_expires_lazily() {
        let now = Instant::now();
        let mut lru = Lru::new(8);
        lru.insert(b"k".to_vec(), b"v".to_vec(), Some(now + Duration::from_millis(10)));
        assert_eq!(lru.get(b"k", now), Some(b"v".to_vec()));
        assert_eq!(lru.get(b"k", now + Duration::from_millis(11)), None);
        assert_eq!(lru.expirations, 1);
        assert_eq!(lru.len(), 0);
    }

    #[test]
    fn overwrite_keeps_one_entry() {
        let mut lru = Lru::new(4);
        lru.insert(b"k".to_vec(), b"1".to_vec(), None);
        lru.insert(b"k".to_vec(), b"2".to_vec(), None);
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.get(b"k", Instant::now()), Some(b"2".to_vec()));
    }
}
