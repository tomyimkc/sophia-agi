// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//
//! Benchmark: NSW approximate vs flat exact search — recall and per-query latency.
//!
//! Quantifies the cost/latency/effect trade-off the JD asks for, on deterministic synthetic
//! data (seeded LCG, no dependencies). Run:  `cargo run --release --bin bench`.

use sophia_ann::{FlatIndex, HnswIndex, NswIndex};
use std::time::Instant;

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    v
}

/// Deterministic pseudo-random unit vectors via a small LCG (reproducible across machines).
fn dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    (0..n).map(|_| normalize((0..dim).map(|_| next()).collect())).collect()
}

fn main() {
    let dim = 64;
    let n = 20_000;
    let nq = 200;
    let k = 10;
    let m = 16;

    let data = dataset(n, dim, 0xC0FFEE);
    let queries = dataset(nq, dim, 0xBEEF);

    let mut flat = FlatIndex::new(dim);
    for (i, v) in data.iter().enumerate() {
        flat.add(i as u32, v);
    }

    // Precompute exact top-k (ground truth) once, and flat latency.
    let mut flat_ns = 0u128;
    let exact: Vec<std::collections::HashSet<u32>> = queries
        .iter()
        .map(|q| {
            let t0 = Instant::now();
            let hits: std::collections::HashSet<u32> =
                flat.search(q, k).into_iter().map(|(id, _)| id).collect();
            flat_ns += t0.elapsed().as_nanos();
            hits
        })
        .collect();
    let flat_us = flat_ns as f64 / nq as f64 / 1e3;

    // Build both graphs once at the same ef_construction; query at a sweep of ef.
    let t = Instant::now();
    let mut nsw = NswIndex::new(dim, m, 200);
    for (i, v) in data.iter().enumerate() {
        nsw.add(i as u32, v);
    }
    let nsw_build_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut hnsw = HnswIndex::new(dim, m, 200);
    for (i, v) in data.iter().enumerate() {
        hnsw.add(i as u32, v);
    }
    let hnsw_build_ms = t.elapsed().as_secs_f64() * 1e3;

    println!("sophia-ann benchmark  (n={n}, dim={dim}, queries={nq}, k={k}, m={m})");
    println!("  build               : NSW {nsw_build_ms:.0} ms | HNSW {hnsw_build_ms:.0} ms");
    println!("  flat (exact) latency: {flat_us:.1} us/query  — recall 1.000 by definition");
    println!("  --- recall/latency trade-off (the cost·latency·effect balance) ---");
    println!(
        "  {:>6}  {:>9} {:>9}  {:>9} {:>9}  {:>8}",
        "ef", "NSW r@10", "NSW us", "HNSW r@10", "HNSW us", "speedup"
    );
    for ef in [16usize, 32, 64, 128, 256] {
        let (nsw_r, nsw_us) = measure(&queries, &exact, |q| nsw.search(q, k, ef), k);
        let (hnsw_r, hnsw_us) = measure(&queries, &exact, |q| hnsw.search(q, k, ef), k);
        let speedup = if hnsw_us > 0.0 { flat_us / hnsw_us } else { 0.0 };
        println!(
            "  {ef:>6}  {nsw_r:>9.3} {nsw_us:>9.1}  {hnsw_r:>9.3} {hnsw_us:>9.1}  {speedup:>7.1}x"
        );
    }
    println!(
        "  (random high-dim vectors are the worst case for a graph index; the HNSW hierarchy\n   \
         lifts recall over single-layer NSW at the same ef. Real clustered embeddings do better.)"
    );
}

/// Run all queries through `search`, returning (recall@k, mean latency µs/query).
fn measure(
    queries: &[Vec<f32>],
    exact: &[std::collections::HashSet<u32>],
    search: impl Fn(&[f32]) -> Vec<(u32, f32)>,
    k: usize,
) -> (f64, f64) {
    let mut hit = 0usize;
    let mut ns = 0u128;
    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let approx = search(q);
        ns += t.elapsed().as_nanos();
        hit += approx.iter().filter(|(id, _)| exact[qi].contains(id)).count();
    }
    let recall = hit as f64 / (queries.len() * k) as f64;
    let us = ns as f64 / queries.len() as f64 / 1e3;
    (recall, us)
}
