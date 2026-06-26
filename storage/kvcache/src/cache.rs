// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Sharded in-memory cache: the data plane behind the server.
//!
//! Keys are routed to one of `num_shards` independent LRU maps by an FNV-1a
//! hash, each guarded by its own `std::sync::Mutex`. The lock is held only for
//! the (non-async, microsecond) map operation and never across `.await`, so
//! many connection tasks contend on 1/num_shards of the keyspace rather than a
//! single global lock.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::lru::Lru;
use crate::protocol::StatsSnapshot;

/// FNV-1a, 64-bit. Chosen over `DefaultHasher` because it is deterministic
/// across runs/processes — a prerequisite for the consistent-hash routing that
/// the multi-node phase (Raft) will build on.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub struct ShardedCache {
    shards: Vec<Mutex<Lru>>,
    hits: AtomicU64,
    misses: AtomicU64,
    sets: AtomicU64,
    dels: AtomicU64,
}

impl ShardedCache {
    /// Build a cache with `num_shards` shards sharing `total_capacity` entries
    /// (split evenly, at least one per shard).
    pub fn new(num_shards: usize, total_capacity: usize) -> Self {
        let num_shards = num_shards.max(1);
        let per_shard = (total_capacity / num_shards).max(1);
        let shards = (0..num_shards).map(|_| Mutex::new(Lru::new(per_shard))).collect();
        ShardedCache {
            shards,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            sets: AtomicU64::new(0),
            dels: AtomicU64::new(0),
        }
    }

    fn shard(&self, key: &[u8]) -> &Mutex<Lru> {
        let idx = (fnv1a(key) % self.shards.len() as u64) as usize;
        &self.shards[idx]
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let out = self.shard(key).lock().unwrap().get(key, Instant::now());
        if out.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    pub fn set(&self, key: &[u8], val: Vec<u8>, ttl_ms: u64) {
        let expire_at = (ttl_ms > 0).then(|| Instant::now() + Duration::from_millis(ttl_ms));
        self.shard(key).lock().unwrap().insert(key.to_vec(), val, expire_at);
        self.sets.fetch_add(1, Ordering::Relaxed);
    }

    pub fn del(&self, key: &[u8]) -> bool {
        let found = self.shard(key).lock().unwrap().remove(key);
        self.dels.fetch_add(1, Ordering::Relaxed);
        found
    }

    pub fn stats(&self) -> StatsSnapshot {
        let (mut evictions, mut expirations, mut entries) = (0u64, 0u64, 0u64);
        for s in &self.shards {
            let g = s.lock().unwrap();
            evictions += g.evictions;
            expirations += g.expirations;
            entries += g.len() as u64;
        }
        StatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            sets: self.sets.load(Ordering::Relaxed),
            dels: self.dels.load(Ordering::Relaxed),
            evictions,
            expirations,
            entries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_spreads_keys_across_shards() {
        let cache = ShardedCache::new(8, 1024);
        // Distinct keys should not all land on one shard.
        let mut seen = std::collections::HashSet::new();
        for i in 0..64 {
            let k = format!("key-{i}");
            let idx = (fnv1a(k.as_bytes()) % 8) as usize;
            seen.insert(idx);
        }
        assert!(seen.len() > 1, "hash collapsed keys onto one shard");
        let _ = cache;
    }

    #[test]
    fn get_set_del_and_stats() {
        let cache = ShardedCache::new(4, 1024);
        assert_eq!(cache.get(b"missing"), None);
        cache.set(b"k", b"v".to_vec(), 0);
        assert_eq!(cache.get(b"k"), Some(b"v".to_vec()));
        assert!(cache.del(b"k"));
        assert!(!cache.del(b"k"));
        let s = cache.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert_eq!(s.sets, 1);
        assert_eq!(s.dels, 2);
        assert_eq!(s.entries, 0);
    }
}
