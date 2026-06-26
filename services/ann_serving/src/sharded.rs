// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//
//! Sharded HNSW — horizontal scale-out for the dense recall view.
//!
//! A single graph is bounded by one machine's memory and one core's walk. `ShardedHnsw` splits
//! the vector set across `N` independent [`HnswIndex`](crate::HnswIndex) shards, hash-routing
//! each id to a shard (stable SplitMix64, so the same id always lands on the same shard and the
//! split is even). A query fans out to **all shards in parallel** (`std::thread::scope`, no
//! dependency) and the per-shard top-k lists are merged into a global top-k.
//!
//! This is the architecture-track primitive the JD calls for ("千亿级数据" — hundred-billion-scale
//! retrieval): build/search throughput scales with shards, each shard fits a memory budget, and
//! the merge is cheap. Determinism is preserved despite parallelism — results are merged then
//! sorted by `(-similarity, id)`, so the output is identical to the sequential merge regardless
//! of thread completion order (verified against [`ShardedHnsw::search_seq`]).
//!
//! Persistence ([`to_bytes`](ShardedHnsw::to_bytes) / [`from_bytes`](ShardedHnsw::from_bytes))
//! lets a built index be saved and reloaded without re-running graph construction — the
//! expensive step — so a server starts in milliseconds instead of rebuilding.

use std::cmp::Ordering;

use crate::{splitmix64, ByteReader, HnswIndex};

const SHARD_MAGIC: &[u8; 8] = b"SOPHSHD1";

/// A set of HNSW shards over one vector space, with id-hash routing and parallel search.
pub struct ShardedHnsw {
    dim: usize,
    shards: Vec<HnswIndex>,
}

impl ShardedHnsw {
    /// `num_shards` independent graphs over `dim`-vectors. `m` / `ef_construction` are the HNSW
    /// build params applied to every shard. At least one shard is always created.
    pub fn new(num_shards: usize, dim: usize, m: usize, ef_construction: usize) -> Self {
        let n = num_shards.max(1);
        Self {
            dim,
            shards: (0..n).map(|_| HnswIndex::new(dim, m, ef_construction)).collect(),
        }
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Per-shard vector counts — exposes the balance of the hash routing.
    pub fn shard_sizes(&self) -> Vec<usize> {
        self.shards.iter().map(|s| s.len()).collect()
    }

    /// The shard an id routes to (stable across runs/processes — same id, same shard).
    #[inline]
    fn shard_for(&self, id: u32) -> usize {
        (splitmix64(id as u64) % self.shards.len() as u64) as usize
    }

    /// Route `id` to its shard and insert. Panics on a dimensionality mismatch (as HnswIndex).
    pub fn add(&mut self, id: u32, vec: &[f32]) {
        let s = self.shard_for(id);
        self.shards[s].add(id, vec);
    }

    /// Global approximate top-`k`. Each shard is searched concurrently with beam width `ef`,
    /// then the union is merged to the global top-`k`. Deterministic (see module docs).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        // Fan out one thread per shard; scope guarantees the borrows outlive the threads.
        let partials: Vec<Vec<(u32, f32)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| scope.spawn(move || shard.search(query, k, ef)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
        });
        merge_topk(partials, k)
    }

    /// Sequential equivalent of [`search`](ShardedHnsw::search) — same result, no threads.
    /// Used as the determinism oracle in tests and as a fallback on single-core hosts.
    pub fn search_seq(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let partials = self.shards.iter().map(|s| s.search(query, k, ef)).collect();
        merge_topk(partials, k)
    }

    /// Search a single shard's top-`k` — the unit of work a [`SearchPool`](crate::SearchPool)
    /// worker runs. Out-of-range shard → empty (graceful).
    pub fn search_shard(&self, shard: usize, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        self.shards.get(shard).map(|s| s.search(query, k, ef)).unwrap_or_default()
    }

    /// Serialize all shards to one portable blob (build-once, load-fast).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(SHARD_MAGIC);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.shards.len() as u32).to_le_bytes());
        for shard in &self.shards {
            let blob = shard.to_bytes();
            out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
            out.extend_from_slice(&blob);
        }
        out
    }

    /// Reconstruct from [`to_bytes`](ShardedHnsw::to_bytes). `None` on bad magic / truncation.
    pub fn from_bytes(bytes: &[u8]) -> Option<ShardedHnsw> {
        let mut r = ByteReader::new(bytes);
        if r.take(8)? != SHARD_MAGIC {
            return None;
        }
        let dim = r.u32()? as usize;
        let n = r.u32()? as usize;
        let mut shards = Vec::with_capacity(n);
        for _ in 0..n {
            let len = u64::from_le_bytes(r.take(8)?.try_into().ok()?) as usize;
            let blob = r.take(len)?;
            shards.push(HnswIndex::from_bytes(blob)?);
        }
        Some(ShardedHnsw { dim, shards })
    }
}

/// Merge per-shard `(id, similarity)` lists into a global top-`k`, sorted by descending
/// similarity with id as a deterministic tie-break. Shared with the persistent pool.
pub(crate) fn merge_topk(partials: Vec<Vec<(u32, f32)>>, k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = partials.into_iter().flatten().collect();
    all.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0))
    });
    all.truncate(k);
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{normalize, FlatIndex};

    fn unit(mut v: Vec<f32>) -> Vec<f32> {
        normalize(&mut v);
        v
    }

    fn lcg_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        (0..n).map(|_| unit((0..dim).map(|_| next()).collect())).collect()
    }

    #[test]
    fn routing_is_stable_and_balanced() {
        let s = ShardedHnsw::new(4, 8, 16, 64);
        for id in [1u32, 2, 3, 1000, 999999] {
            assert_eq!(s.shard_for(id), s.shard_for(id)); // stable
        }
        let mut s = ShardedHnsw::new(4, 4, 8, 32);
        for id in 0..4000u32 {
            s.add(id, &unit(vec![id as f32 % 7.0, 1.0, 2.0, 3.0]));
        }
        assert_eq!(s.len(), 4000);
        // Even-ish split: no shard wildly off from n/shards.
        for sz in s.shard_sizes() {
            assert!(sz > 700 && sz < 1300, "uneven shard: {sz}");
        }
    }

    #[test]
    fn parallel_search_equals_sequential() {
        let dim = 16;
        let data = lcg_dataset(600, dim, 0xABCDEF);
        let mut s = ShardedHnsw::new(5, dim, 16, 64);
        for (i, v) in data.iter().enumerate() {
            s.add(i as u32, v);
        }
        for q in lcg_dataset(20, dim, 0x13579) {
            assert_eq!(s.search(&q, 10, 64), s.search_seq(&q, 10, 64));
        }
    }

    #[test]
    fn sharded_recall_is_high_vs_flat() {
        // Sharding must not wreck recall: each shard searched at ef and merged should still
        // recover most of the true global top-10.
        let dim = 24;
        let data = lcg_dataset(1200, dim, 0xC0FFEE);
        let mut flat = FlatIndex::new(dim);
        let mut sharded = ShardedHnsw::new(4, dim, 16, 96);
        for (i, v) in data.iter().enumerate() {
            flat.add(i as u32, v);
            sharded.add(i as u32, v);
        }
        let queries = lcg_dataset(40, dim, 0xBEEF);
        let mut hit = 0usize;
        for q in &queries {
            let exact: std::collections::HashSet<u32> =
                flat.search(q, 10).into_iter().map(|(id, _)| id).collect();
            hit += sharded.search(q, 10, 96).iter().filter(|(id, _)| exact.contains(id)).count();
        }
        let recall = hit as f32 / (queries.len() * 10) as f32;
        assert!(recall >= 0.85, "sharded recall@10 too low: {recall}");
    }

    #[test]
    fn persistence_roundtrips_identically() {
        let dim = 20;
        let data = lcg_dataset(500, dim, 0x55AA);
        let mut s = ShardedHnsw::new(3, dim, 16, 64);
        for (i, v) in data.iter().enumerate() {
            s.add(i as u32, v);
        }
        let blob = s.to_bytes();
        let loaded = ShardedHnsw::from_bytes(&blob).expect("reload");
        assert_eq!(loaded.num_shards(), s.num_shards());
        assert_eq!(loaded.len(), s.len());
        // Identical search results before/after a save+load — graph fully preserved.
        for q in lcg_dataset(15, dim, 0x9090) {
            assert_eq!(s.search(&q, 10, 64), loaded.search(&q, 10, 64));
        }
    }

    #[test]
    fn from_bytes_rejects_garbage() {
        assert!(ShardedHnsw::from_bytes(b"not a real blob").is_none());
        assert!(ShardedHnsw::from_bytes(b"").is_none());
    }
}
