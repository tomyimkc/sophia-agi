// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Dependency-free microbenchmark for the LSM engine.
//!
//! Not criterion (we keep the workspace offline-buildable). It reports ops/sec
//! and p50/p99 latency for put and get over a configurable key space. Run with:
//!   cargo bench -p sophia-lsm
//! or as a normal binary for a quick read:
//!   cargo run --release --bench engine_bench

use std::time::Instant;

use sophia_lsm::{Engine, Options};

fn percentile(sorted_nanos: &[u128], p: f64) -> u128 {
    if sorted_nanos.is_empty() {
        return 0;
    }
    let idx = ((sorted_nanos.len() as f64 - 1.0) * p).round() as usize;
    sorted_nanos[idx]
}

fn report(name: &str, mut lat: Vec<u128>, total: std::time::Duration) {
    lat.sort_unstable();
    let n = lat.len();
    let ops = n as f64 / total.as_secs_f64();
    println!(
        "{name:<14} n={n:>7}  {ops:>12.0} ops/s   p50={:>6}ns  p99={:>7}ns",
        percentile(&lat, 0.50),
        percentile(&lat, 0.99),
    );
}

fn main() {
    let n: usize = std::env::var("BENCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let dir = std::env::temp_dir().join(format!("sophia-lsm-bench-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();

    // 16 MiB memtable so most of the run stays in-memory + WAL; flushes exercise
    // the SSTable path. Tune via env to push the on-disk read path.
    let opts = Options::new(&dir).flush_threshold_bytes(16 * 1024 * 1024);
    let mut db = Engine::open(opts).expect("open");

    println!("sophia-lsm bench  (n={n}, dir={})", dir.display());

    // --- PUT ---
    let mut put_lat = Vec::with_capacity(n);
    let t0 = Instant::now();
    for i in 0..n {
        let key = format!("claim:{i:010}");
        let val = format!("verdict=accepted;src=okf://p{};conf=0.7", i % 1000);
        let s = Instant::now();
        db.put(key.as_bytes(), val.as_bytes()).expect("put");
        put_lat.push(s.elapsed().as_nanos());
    }
    let put_total = t0.elapsed();

    // --- GET (hot, existing keys) ---
    let mut get_lat = Vec::with_capacity(n);
    let t1 = Instant::now();
    let mut sink = 0usize;
    for i in 0..n {
        let key = format!("claim:{i:010}");
        let s = Instant::now();
        let got = db.get(key.as_bytes()).expect("get");
        get_lat.push(s.elapsed().as_nanos());
        sink += got.map_or(0, |v| v.len());
    }
    let get_total = t1.elapsed();

    report("put+fsync", put_lat, put_total);
    report("get", get_lat, get_total);
    println!("tables on disk: {}  (checksum sink={sink})", db.table_count());

    std::fs::remove_dir_all(&dir).ok();
}
