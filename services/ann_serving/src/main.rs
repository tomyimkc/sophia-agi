// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//
//! Benchmark: NSW approximate vs flat exact search — recall and per-query latency.
//!
//! Quantifies the cost/latency/effect trade-off the JD asks for, on deterministic synthetic
//! data (seeded LCG, no dependencies). Run:  `cargo run --release --bin bench`.

use sophia_ann::{FlatIndex, NswIndex};
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

    // Build once at the largest ef_construction; query at a sweep of ef to trace the curve.
    let build_start = Instant::now();
    let mut nsw = NswIndex::new(dim, m, 256);
    for (i, v) in data.iter().enumerate() {
        nsw.add(i as u32, v);
    }
    let build_ms = build_start.elapsed().as_secs_f64() * 1e3;

    println!("sophia-ann benchmark  (n={n}, dim={dim}, queries={nq}, k={k}, m={m})");
    println!("  NSW build           : {build_ms:.1} ms");
    println!("  flat (exact) latency: {flat_us:.1} us/query  — recall 1.000 by definition");
    println!("  --- NSW recall/latency trade-off (the cost·latency·effect balance) ---");
    println!("  {:>6}  {:>12}  {:>10}  {:>10}", "ef", "latency(us)", "recall@10", "speedup");
    for ef in [16usize, 32, 64, 128, 256] {
        let mut hit = 0usize;
        let mut nsw_ns = 0u128;
        for (qi, q) in queries.iter().enumerate() {
            let t1 = Instant::now();
            let approx = nsw.search(q, k, ef);
            nsw_ns += t1.elapsed().as_nanos();
            hit += approx.iter().filter(|(id, _)| exact[qi].contains(id)).count();
        }
        let recall = hit as f64 / (nq * k) as f64;
        let nsw_us = nsw_ns as f64 / nq as f64 / 1e3;
        let speedup = if nsw_us > 0.0 { flat_us / nsw_us } else { 0.0 };
        println!("  {ef:>6}  {nsw_us:>12.1}  {recall:>10.3}  {speedup:>9.1}x");
    }
    println!(
        "  (random high-dim vectors are the worst case for a single-layer graph; ef lifts\n   \
         recall toward exact at a latency cost. Multi-layer HNSW is the documented next step.)"
    );
}
