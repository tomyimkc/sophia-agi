// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//
//! Persistent search pool — amortize the parallel fan-out's thread cost across queries.
//!
//! [`ShardedHnsw::search`](crate::ShardedHnsw::search) spawns fresh threads **per query**
//! (`std::thread::scope`): correct and dependency-free, but the spawn cost can exceed the
//! per-shard saving at modest scale (see the bench). `SearchPool` fixes that: it spins up one
//! long-lived worker thread per shard at construction; each query is *dispatched* to the already-
//! running workers over channels instead of paying thread creation. This is where the parallel
//! latency win actually lands.
//!
//! Still dependency-free (`std::sync::mpsc` + `Arc`). Deterministic and identical to
//! `ShardedHnsw::search` (the partials are merged then sorted by `(-similarity, id)`, so worker
//! completion order is irrelevant) — verified in tests. Concurrency-safe: each `search` carries
//! its own result channel, so multiple threads may query one pool at once.

use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::sharded::merge_topk;
use crate::ShardedHnsw;

enum Job {
    Search {
        query: Arc<Vec<f32>>,
        k: usize,
        ef: usize,
        out: Sender<Vec<(u32, f32)>>,
    },
    Stop,
}

/// A sharded index wrapped in a persistent worker pool (one worker per shard).
pub struct SearchPool {
    index: Arc<ShardedHnsw>,
    workers: Vec<Sender<Job>>,
    handles: Vec<Option<JoinHandle<()>>>,
}

impl SearchPool {
    /// Take ownership of a built [`ShardedHnsw`] and start one worker thread per shard.
    pub fn new(index: ShardedHnsw) -> Self {
        let index = Arc::new(index);
        let n = index.num_shards();
        let mut workers = Vec::with_capacity(n);
        let mut handles = Vec::with_capacity(n);
        for shard in 0..n {
            let (tx, rx) = channel::<Job>();
            let idx = Arc::clone(&index);
            let handle = std::thread::spawn(move || {
                for job in rx {
                    match job {
                        Job::Search { query, k, ef, out } => {
                            let _ = out.send(idx.search_shard(shard, &query, k, ef));
                        }
                        Job::Stop => break,
                    }
                }
            });
            workers.push(tx);
            handles.push(Some(handle));
        }
        Self { index, workers, handles }
    }

    pub fn num_shards(&self) -> usize {
        self.index.num_shards()
    }

    pub fn dim(&self) -> usize {
        self.index.dim()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Global approximate top-`k`, dispatched to the persistent workers. Identical result to
    /// `ShardedHnsw::search`. Thread-safe: each call uses a private result channel.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let q = Arc::new(query.to_vec());
        let (tx, rx) = channel::<Vec<(u32, f32)>>();
        let mut dispatched = 0usize;
        for worker in &self.workers {
            let job = Job::Search { query: Arc::clone(&q), k, ef, out: tx.clone() };
            if worker.send(job).is_ok() {
                dispatched += 1;
            }
        }
        drop(tx); // so the collect below terminates once every worker has replied
        // Collect exactly the partials we dispatched (a dead worker simply contributes none).
        let partials: Vec<Vec<(u32, f32)>> = rx.iter().take(dispatched).collect();
        merge_topk(partials, k)
    }

    /// Hand the underlying index back (stops the workers). Useful for persistence after serving.
    pub fn into_inner(mut self) -> Arc<ShardedHnsw> {
        self.stop();
        Arc::clone(&self.index)
    }

    fn stop(&mut self) {
        for worker in &self.workers {
            let _ = worker.send(Job::Stop);
        }
        for handle in &mut self.handles {
            if let Some(h) = handle.take() {
                let _ = h.join();
            }
        }
    }
}

impl Drop for SearchPool {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize;

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

    fn build(n_shards: usize, dim: usize, n: usize, seed: u64) -> (ShardedHnsw, Vec<Vec<f32>>) {
        let data = lcg_dataset(n, dim, seed);
        let mut s = ShardedHnsw::new(n_shards, dim, 16, 64);
        for (i, v) in data.iter().enumerate() {
            s.add(i as u32, v);
        }
        (s, data)
    }

    #[test]
    fn pool_search_equals_direct_search() {
        let dim = 16;
        let (sharded, _) = build(5, dim, 600, 0xABCDEF);
        let queries = lcg_dataset(25, dim, 0x2468);
        // Capture the oracle BEFORE moving the index into the pool.
        let oracle: Vec<_> = queries.iter().map(|q| sharded.search(q, 10, 64)).collect();
        let pool = SearchPool::new(sharded);
        for (q, want) in queries.iter().zip(&oracle) {
            assert_eq!(&pool.search(q, 10, 64), want);
        }
    }

    #[test]
    fn pool_is_concurrency_safe() {
        use std::sync::Arc as StdArc;
        let dim = 12;
        let (sharded, _) = build(4, dim, 400, 0x99);
        let queries = lcg_dataset(8, dim, 0x77);
        let oracle: Vec<_> = queries.iter().map(|q| sharded.search(q, 5, 32)).collect();
        let pool = StdArc::new(SearchPool::new(sharded));

        // Hammer the pool from several threads at once; every result must match the oracle.
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let pool = StdArc::clone(&pool);
                let queries = &queries;
                let oracle = &oracle;
                scope.spawn(move || {
                    for (q, want) in queries.iter().zip(oracle) {
                        assert_eq!(&pool.search(q, 5, 32), want);
                    }
                });
            }
        });
    }

    #[test]
    fn pool_into_inner_recovers_index() {
        let (sharded, _) = build(3, 8, 120, 0x1234);
        let len = sharded.len();
        let pool = SearchPool::new(sharded);
        let inner = pool.into_inner();
        assert_eq!(inner.len(), len);
    }
}
