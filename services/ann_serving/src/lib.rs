// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//
//! Flat (exact) + NSW (approximate) nearest-neighbour search over cosine similarity.
//!
//! This is the architecture-track counterpart to Sophia's Python recall layer: the same
//! dense-vector search, but as a dependency-free Rust core built for the cost/latency budget
//! the JD names ("在成本、延迟与效果之间寻找最优平衡"). It ships two indexes over the same
//! vectors so the trade-off is measurable, not asserted:
//!
//!   * [`FlatIndex`]  — brute-force exact search. O(n·d) per query; the recall ground truth.
//!   * [`NswIndex`]   — a navigable small-world graph (the core idea behind HNSW). Greedy
//!     beam search visits a small neighbourhood instead of the whole set, trading a sliver
//!     of recall for a large latency win as `n` grows.
//!
//! Vectors are assumed L2-normalised, so cosine similarity is a plain dot product (the
//! Python `local-hash-v1` embedder already emits unit vectors — see `agent/rag_local_embed.py`).
//! Higher score = closer. Everything is deterministic: NSW construction depends only on
//! insertion order, so `cargo test` and the `bench` binary reproduce bit-for-bit.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Cosine similarity for L2-normalised vectors (== dot product). Panics on length mismatch.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimensionality mismatch");
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// L2-normalise in place; a zero vector is left untouched. Mirrors the Python embedder so a
/// caller can feed raw vectors and still get cosine semantics.
pub fn normalize(v: &mut [f32]) {
    let mut norm = 0.0f32;
    for &x in v.iter() {
        norm += x * x;
    }
    let norm = norm.sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// A (similarity, id) pair ordered by similarity. Total order over f32 (NaN sorts lowest) so
/// it can live in a `BinaryHeap`.
#[derive(Clone, Copy, Debug)]
struct Scored {
    sim: f32,
    idx: usize,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.sim == other.sim && self.idx == other.idx
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        // NaN-safe: treat NaN as the smallest similarity. Tie-break on idx for determinism.
        match self
            .sim
            .partial_cmp(&other.sim)
            .unwrap_or_else(|| match (self.sim.is_nan(), other.sim.is_nan()) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => Ordering::Equal,
            }) {
            Ordering::Equal => other.idx.cmp(&self.idx),
            ord => ord,
        }
    }
}

/// Brute-force exact cosine index. The recall ground truth and the right choice below a few
/// thousand vectors, where a graph's overhead doesn't pay off.
pub struct FlatIndex {
    dim: usize,
    ids: Vec<u32>,
    data: Vec<f32>, // row-major, ids.len() * dim
}

impl FlatIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, ids: Vec::new(), data: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Append a vector under `id`. Panics if `vec.len() != dim`.
    pub fn add(&mut self, id: u32, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim, "vector dimensionality mismatch");
        self.ids.push(id);
        self.data.extend_from_slice(vec);
    }

    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Top-`k` ids by descending cosine similarity. Exact.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dimensionality mismatch");
        let mut scored: Vec<Scored> = (0..self.ids.len())
            .map(|i| Scored { sim: cosine(query, self.row(i)), idx: i })
            .collect();
        scored.sort_unstable_by(|a, b| b.cmp(a)); // descending
        scored.into_iter().take(k).map(|s| (self.ids[s.idx], s.sim)).collect()
    }
}

struct Node {
    id: u32,
    vec: Vec<f32>,
    neighbors: Vec<usize>,
}

/// A single-layer navigable small-world graph with greedy beam search — the search/insert
/// core of HNSW without the layer hierarchy. Each node keeps up to `m` neighbours; queries
/// walk the graph greedily, visiting only a small frontier (`ef`) instead of every vector.
pub struct NswIndex {
    dim: usize,
    m: usize,
    ef_construction: usize,
    nodes: Vec<Node>,
    entry: Option<usize>,
}

impl NswIndex {
    /// `m` = max neighbours per node (graph degree); `ef_construction` = beam width while
    /// inserting (higher = better-connected graph, slower build). Typical: m=16, ef=64.
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        Self { dim, m: m.max(1), ef_construction: ef_construction.max(1), nodes: Vec::new(), entry: None }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    fn sim(&self, a: usize, query: &[f32]) -> f32 {
        cosine(&self.nodes[a].vec, query)
    }

    /// Greedy beam search from `entry`, returning up to `ef` visited nodes closest to
    /// `query` (as Scored), best last is not guaranteed — caller sorts. This is the shared
    /// primitive for both insertion (ef_construction) and query (ef).
    fn search_layer(&self, query: &[f32], entry: usize, ef: usize) -> Vec<Scored> {
        let mut visited = vec![false; self.nodes.len()];
        // `candidates` is a max-heap on similarity → always expand the most promising node.
        let mut candidates: BinaryHeap<Scored> = BinaryHeap::new();
        // `result` is a min-heap (Reverse) of the ef best so far → cheap eviction of the worst.
        let mut result: BinaryHeap<std::cmp::Reverse<Scored>> = BinaryHeap::new();

        let e = Scored { sim: self.sim(entry, query), idx: entry };
        visited[entry] = true;
        candidates.push(e);
        result.push(std::cmp::Reverse(e));

        while let Some(c) = candidates.pop() {
            // Stop when the best remaining candidate is worse than the worst kept result.
            if let Some(std::cmp::Reverse(worst)) = result.peek() {
                if c.sim < worst.sim && result.len() >= ef {
                    break;
                }
            }
            for &nb in &self.nodes[c.idx].neighbors {
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;
                let s = Scored { sim: self.sim(nb, query), idx: nb };
                let worst_sim = result.peek().map(|std::cmp::Reverse(w)| w.sim);
                if result.len() < ef || worst_sim.map_or(true, |w| s.sim > w) {
                    candidates.push(s);
                    result.push(std::cmp::Reverse(s));
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }
        result.into_iter().map(|std::cmp::Reverse(s)| s).collect()
    }

    /// Insert a vector under `id`. Connects it to its `m` nearest existing nodes (found via a
    /// greedy beam) and adds the back-edges, pruning each touched node back to degree `m`.
    pub fn add(&mut self, id: u32, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim, "vector dimensionality mismatch");
        let new_idx = self.nodes.len();
        self.nodes.push(Node { id, vec: vec.to_vec(), neighbors: Vec::new() });

        let entry = match self.entry {
            None => {
                self.entry = Some(new_idx);
                return;
            }
            Some(e) => e,
        };

        let mut found = self.search_layer(vec, entry, self.ef_construction);
        found.sort_unstable_by(|a, b| b.cmp(a)); // best first
        found.retain(|s| s.idx != new_idx);

        for s in found.iter().take(self.m) {
            self.nodes[new_idx].neighbors.push(s.idx);
            self.nodes[s.idx].neighbors.push(new_idx);
            self.prune(s.idx);
        }
        self.prune(new_idx);
    }

    /// Keep only the `m` most-similar neighbours of `node`, dropping the rest. Bounds degree
    /// so search stays fast and memory stays linear.
    fn prune(&mut self, node: usize) {
        if self.nodes[node].neighbors.len() <= self.m {
            return;
        }
        let base = self.nodes[node].vec.clone();
        let mut scored: Vec<Scored> = self.nodes[node]
            .neighbors
            .iter()
            .map(|&nb| Scored { sim: cosine(&base, &self.nodes[nb].vec), idx: nb })
            .collect();
        scored.sort_unstable_by(|a, b| b.cmp(a));
        scored.truncate(self.m);
        self.nodes[node].neighbors = scored.into_iter().map(|s| s.idx).collect();
    }

    /// Approximate top-`k` ids by cosine. `ef` (>= k) is the search beam width: larger trades
    /// latency for recall. Returns fewer than `k` only if the index holds fewer vectors.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dimensionality mismatch");
        let entry = match self.entry {
            None => return Vec::new(),
            Some(e) => e,
        };
        let mut found = self.search_layer(query, entry, ef.max(k));
        found.sort_unstable_by(|a, b| b.cmp(a));
        found.into_iter().take(k).map(|s| (self.nodes[s.idx].id, s.sim)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(mut v: Vec<f32>) -> Vec<f32> {
        normalize(&mut v);
        v
    }

    #[test]
    fn cosine_of_identical_unit_vectors_is_one() {
        let a = unit(vec![1.0, 2.0, 3.0]);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn flat_search_returns_nearest_first() {
        let mut idx = FlatIndex::new(2);
        idx.add(10, &unit(vec![1.0, 0.0]));
        idx.add(20, &unit(vec![0.0, 1.0]));
        idx.add(30, &unit(vec![0.9, 0.1]));
        let hits = idx.search(&unit(vec![1.0, 0.05]), 2);
        assert_eq!(hits[0].0, 10);
        assert_eq!(hits[1].0, 30);
    }

    #[test]
    fn nsw_matches_flat_on_top1_for_separable_clusters() {
        // Two well-separated clusters → NSW top-1 must equal exact top-1.
        let mut flat = FlatIndex::new(3);
        let mut nsw = NswIndex::new(3, 8, 32);
        let pts = [
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.95, 0.05, 0.0]),
            (3, vec![0.9, 0.1, 0.0]),
            (4, vec![0.0, 0.0, 1.0]),
            (5, vec![0.05, 0.0, 0.95]),
            (6, vec![0.0, 0.1, 0.9]),
        ];
        for (id, v) in pts.iter() {
            let u = unit(v.clone());
            flat.add(*id, &u);
            nsw.add(*id, &u);
        }
        let q = unit(vec![1.0, 0.02, 0.0]);
        assert_eq!(flat.search(&q, 1)[0].0, nsw.search(&q, 1, 16)[0].0);
    }

    // Deterministic pseudo-random unit vectors via a tiny LCG (no rand dependency).
    fn lcg_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        (0..n).map(|_| unit((0..dim).map(|_| next()).collect())).collect()
    }

    #[test]
    fn nsw_recall_at_10_is_high_vs_flat() {
        let dim = 32;
        let data = lcg_dataset(800, dim, 0xC0FFEE);
        let mut flat = FlatIndex::new(dim);
        let mut nsw = NswIndex::new(dim, 16, 64);
        for (i, v) in data.iter().enumerate() {
            flat.add(i as u32, v);
            nsw.add(i as u32, v);
        }
        let queries = lcg_dataset(40, dim, 0xBEEF);
        let mut hit = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let exact: std::collections::HashSet<u32> =
                flat.search(q, 10).into_iter().map(|(id, _)| id).collect();
            let approx = nsw.search(q, 10, 64);
            for (id, _) in approx {
                if exact.contains(&id) {
                    hit += 1;
                }
            }
            total += 10;
        }
        let recall = hit as f32 / total as f32;
        assert!(recall >= 0.85, "NSW recall@10 too low: {recall}");
    }
}
