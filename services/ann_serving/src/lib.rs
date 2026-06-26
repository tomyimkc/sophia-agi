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
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub mod sharded;
pub use sharded::ShardedHnsw;

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

// ---------------------------------------------------------------------------------------
// Multi-layer HNSW — the hierarchical upgrade over the single-layer NSW above.
//
// Hierarchical Navigable Small World adds a tower of sparse upper layers over the dense base
// layer 0. A query descends greedily through the upper layers (each a coarse "express lane")
// to land near the answer, then does the wide `ef` beam only on layer 0. The result is higher
// recall at the *same* `ef` than a flat NSW graph — the recall/latency curve shifts up.
//
// Determinism: a node's top level is drawn from the standard exp(-l/ml) distribution, but the
// "randomness" is a SplitMix64 hash of the insertion index — so builds are reproducible with
// no `rand` dependency.
// ---------------------------------------------------------------------------------------

struct HnswNode {
    id: u32,
    vec: Vec<f32>,
    /// `levels[l]` = neighbour indices at layer `l`. Length = this node's top level + 1.
    levels: Vec<Vec<usize>>,
}

/// Parse one exported index line — whitespace-separated `id f0 f1 …` — into `(id, vector)`.
/// Returns `None` on a malformed/empty line so a loader can skip it. This is the wire format
/// the Python bridge (`tools/export_rag_index.py`) writes and the `serve` binary reads.
pub fn parse_index_line(line: &str) -> Option<(u32, Vec<f32>)> {
    let mut it = line.split_whitespace();
    let id: u32 = it.next()?.parse().ok()?;
    let vec: Vec<f32> = it.map(|t| t.parse::<f32>().ok()).collect::<Option<_>>()?;
    if vec.is_empty() {
        return None;
    }
    Some((id, vec))
}

/// Deterministic SplitMix64 — turns an insertion index into a stable pseudo-random u64.
/// `pub(crate)` so the sharded index can hash-route ids with the same stable mixer.
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A hierarchical NSW index. `m` = neighbours per node on upper layers (layer 0 uses `2*m`),
/// `ef_construction` = build-time beam width, higher = better graph + slower build.
pub struct HnswIndex {
    dim: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ml: f64,
    nodes: Vec<HnswNode>,
    entry: Option<usize>,
    max_level: usize,
}

impl HnswIndex {
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        let m = m.max(1);
        Self {
            dim,
            m,
            m0: m * 2,
            ef_construction: ef_construction.max(1),
            ml: 1.0 / (m as f64).ln().max(1e-9),
            nodes: Vec::new(),
            entry: None,
            max_level: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    fn sim(&self, node: usize, query: &[f32]) -> f32 {
        cosine(&self.nodes[node].vec, query)
    }

    /// Deterministic top level for the node at insertion index `idx`.
    fn assign_level(&self, idx: usize) -> usize {
        let h = splitmix64(idx as u64 + 1);
        // Uniform in (0, 1] from the top 53 bits.
        let mut u = (h >> 11) as f64 / ((1u64 << 53) as f64);
        if u <= 0.0 {
            u = f64::MIN_POSITIVE;
        }
        (-u.ln() * self.ml).floor() as usize
    }

    /// Greedy beam search confined to `layer`, seeded from `entries`. Returns up to `ef`
    /// closest visited nodes (unsorted; caller sorts).
    fn search_layer(&self, query: &[f32], entries: &[usize], ef: usize, layer: usize) -> Vec<Scored> {
        let mut visited = vec![false; self.nodes.len()];
        let mut candidates: BinaryHeap<Scored> = BinaryHeap::new();
        let mut result: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();

        for &e in entries {
            if visited[e] {
                continue;
            }
            visited[e] = true;
            let s = Scored { sim: self.sim(e, query), idx: e };
            candidates.push(s);
            result.push(Reverse(s));
        }

        while let Some(c) = candidates.pop() {
            if let Some(Reverse(worst)) = result.peek() {
                if c.sim < worst.sim && result.len() >= ef {
                    break;
                }
            }
            for &nb in &self.nodes[c.idx].levels[layer] {
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;
                let s = Scored { sim: self.sim(nb, query), idx: nb };
                let worst_sim = result.peek().map(|Reverse(w)| w.sim);
                if result.len() < ef || worst_sim.map_or(true, |w| s.sim > w) {
                    candidates.push(s);
                    result.push(Reverse(s));
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }
        result.into_iter().map(|Reverse(s)| s).collect()
    }

    /// Keep only the `max_deg` closest neighbours of `node` at `layer`.
    fn prune(&mut self, node: usize, layer: usize, max_deg: usize) {
        if self.nodes[node].levels[layer].len() <= max_deg {
            return;
        }
        let base = self.nodes[node].vec.clone();
        let mut scored: Vec<Scored> = self.nodes[node].levels[layer]
            .iter()
            .map(|&nb| Scored { sim: cosine(&base, &self.nodes[nb].vec), idx: nb })
            .collect();
        scored.sort_unstable_by(|a, b| b.cmp(a));
        scored.truncate(max_deg);
        self.nodes[node].levels[layer] = scored.into_iter().map(|s| s.idx).collect();
    }

    /// Insert a vector under `id`.
    pub fn add(&mut self, id: u32, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim, "vector dimensionality mismatch");
        let idx = self.nodes.len();
        let level = self.assign_level(idx);
        self.nodes.push(HnswNode {
            id,
            vec: vec.to_vec(),
            levels: vec![Vec::new(); level + 1],
        });

        let entry = match self.entry {
            None => {
                self.entry = Some(idx);
                self.max_level = level;
                return;
            }
            Some(e) => e,
        };

        // Descend the layers above this node's level with a width-1 beam to find an entry.
        let mut cur = entry;
        let mut l = self.max_level;
        while l > level {
            let found = self.search_layer(vec, &[cur], 1, l);
            if let Some(best) = found.into_iter().max_by(|a, b| a.cmp(b)) {
                cur = best.idx;
            }
            l -= 1;
        }

        // From this node's level down to 0, connect to the M (or M0 at layer 0) nearest.
        let mut entries = vec![cur];
        let top = level.min(self.max_level);
        for layer in (0..=top).rev() {
            let mut w = self.search_layer(vec, &entries, self.ef_construction, layer);
            w.sort_unstable_by(|a, b| b.cmp(a));
            w.retain(|s| s.idx != idx);
            let max_deg = if layer == 0 { self.m0 } else { self.m };
            for s in w.iter().take(max_deg) {
                self.nodes[idx].levels[layer].push(s.idx);
                self.nodes[s.idx].levels[layer].push(idx);
                self.prune(s.idx, layer, max_deg);
            }
            self.prune(idx, layer, max_deg);
            entries = w.into_iter().take(self.ef_construction).map(|s| s.idx).collect();
            if entries.is_empty() {
                entries = vec![cur];
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(idx);
        }
    }

    /// Approximate top-`k` by cosine. `ef` (>= k) is the layer-0 beam width.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dimensionality mismatch");
        let entry = match self.entry {
            None => return Vec::new(),
            Some(e) => e,
        };
        let mut cur = entry;
        let mut l = self.max_level;
        while l > 0 {
            let found = self.search_layer(query, &[cur], 1, l);
            if let Some(best) = found.into_iter().max_by(|a, b| a.cmp(b)) {
                cur = best.idx;
            }
            l -= 1;
        }
        let mut w = self.search_layer(query, &[cur], ef.max(k), 0);
        w.sort_unstable_by(|a, b| b.cmp(a));
        w.into_iter().take(k).map(|s| (self.nodes[s.idx].id, s.sim)).collect()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Serialize the built graph (vectors + per-layer adjacency + params) to a portable,
    /// little-endian byte blob. Loading this with [`HnswIndex::from_bytes`] skips the expensive
    /// graph construction — the "build once, serve many" path for the architecture track.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(HNSW_MAGIC);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.m as u32).to_le_bytes());
        out.extend_from_slice(&(self.ef_construction as u32).to_le_bytes());
        out.extend_from_slice(&(self.max_level as u32).to_le_bytes());
        out.extend_from_slice(&self.entry.map_or(-1i64, |e| e as i64).to_le_bytes());
        out.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        for node in &self.nodes {
            out.extend_from_slice(&node.id.to_le_bytes());
            out.extend_from_slice(&(node.levels.len() as u32).to_le_bytes());
            for level in &node.levels {
                out.extend_from_slice(&(level.len() as u32).to_le_bytes());
                for &nb in level {
                    out.extend_from_slice(&(nb as u32).to_le_bytes());
                }
            }
            for &x in &node.vec {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// Rebuild an index from [`HnswIndex::to_bytes`] output. Returns `None` on a bad magic or
    /// truncated/corrupt blob (never panics), so a loader can fall back to rebuilding.
    pub fn from_bytes(bytes: &[u8]) -> Option<HnswIndex> {
        let mut r = ByteReader::new(bytes);
        if r.take(8)? != HNSW_MAGIC {
            return None;
        }
        let dim = r.u32()? as usize;
        let m = (r.u32()? as usize).max(1);
        let ef_construction = (r.u32()? as usize).max(1);
        let max_level = r.u32()? as usize;
        let entry_raw = r.i64()?;
        let entry = if entry_raw < 0 { None } else { Some(entry_raw as usize) };
        let n = r.u32()? as usize;
        let mut nodes = Vec::with_capacity(n);
        for _ in 0..n {
            let id = r.u32()?;
            let nlevels = r.u32()? as usize;
            let mut levels = Vec::with_capacity(nlevels);
            for _ in 0..nlevels {
                let cnt = r.u32()? as usize;
                let mut nbrs = Vec::with_capacity(cnt);
                for _ in 0..cnt {
                    nbrs.push(r.u32()? as usize);
                }
                levels.push(nbrs);
            }
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                vec.push(r.f32()?);
            }
            nodes.push(HnswNode { id, vec, levels });
        }
        Some(HnswIndex {
            dim,
            m,
            m0: m * 2,
            ef_construction,
            ml: 1.0 / (m as f64).ln().max(1e-9),
            nodes,
            entry,
            max_level,
        })
    }
}

/// Magic header for the persisted HNSW format (version 1).
const HNSW_MAGIC: &[u8; 8] = b"SOPHNSW1";

/// Minimal little-endian byte cursor for the persistence format — None on overrun, no panic.
pub(crate) struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.pos + n > self.bytes.len() {
            return None;
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Some(s)
    }
    pub(crate) fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    pub(crate) fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    pub(crate) fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
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

    fn recall_at_10(
        queries: &[Vec<f32>],
        flat: &FlatIndex,
        search: impl Fn(&[f32]) -> Vec<(u32, f32)>,
    ) -> f32 {
        let mut hit = 0usize;
        for q in queries {
            let exact: std::collections::HashSet<u32> =
                flat.search(q, 10).into_iter().map(|(id, _)| id).collect();
            hit += search(q).iter().filter(|(id, _)| exact.contains(id)).count();
        }
        hit as f32 / (queries.len() * 10) as f32
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
        let recall = recall_at_10(&queries, &flat, |q| nsw.search(q, 10, 64));
        assert!(recall >= 0.85, "NSW recall@10 too low: {recall}");
    }

    #[test]
    fn hnsw_top1_matches_flat_on_separable_clusters() {
        let mut flat = FlatIndex::new(3);
        let mut hnsw = HnswIndex::new(3, 8, 32);
        let pts = [
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.95, 0.05, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
            (4, vec![0.05, 0.0, 0.95]),
        ];
        for (id, v) in pts.iter() {
            let u = unit(v.clone());
            flat.add(*id, &u);
            hnsw.add(*id, &u);
        }
        let q = unit(vec![1.0, 0.02, 0.0]);
        assert_eq!(flat.search(&q, 1)[0].0, hnsw.search(&q, 1, 16)[0].0);
    }

    #[test]
    fn parse_index_line_roundtrips_and_rejects_garbage() {
        let (id, v) = parse_index_line("7 0.1 -0.2 0.3").unwrap();
        assert_eq!(id, 7);
        assert_eq!(v, vec![0.1, -0.2, 0.3]);
        assert!(parse_index_line("").is_none());
        assert!(parse_index_line("42").is_none()); // id but no vector
        assert!(parse_index_line("x 0.1 0.2").is_none()); // bad id
    }

    #[test]
    fn hnsw_beats_single_layer_nsw_at_equal_ef() {
        // The whole point of the hierarchy: higher recall at the SAME ef.
        let dim = 32;
        let data = lcg_dataset(1500, dim, 0xC0FFEE);
        let mut flat = FlatIndex::new(dim);
        let mut nsw = NswIndex::new(dim, 16, 100);
        let mut hnsw = HnswIndex::new(dim, 16, 100);
        for (i, v) in data.iter().enumerate() {
            flat.add(i as u32, v);
            nsw.add(i as u32, v);
            hnsw.add(i as u32, v);
        }
        let queries = lcg_dataset(60, dim, 0xBEEF);
        let ef = 24; // deliberately small so the hierarchy's advantage shows
        let nsw_r = recall_at_10(&queries, &flat, |q| nsw.search(q, 10, ef));
        let hnsw_r = recall_at_10(&queries, &flat, |q| hnsw.search(q, 10, ef));
        assert!(hnsw_r >= nsw_r, "HNSW {hnsw_r} should be >= NSW {nsw_r} at ef={ef}");
    }
}
