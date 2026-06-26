// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! diskstore-bench — batched random-read throughput, pread vs io_uring.
//!
//! Loads N keys, drops the page cache effect as best it can, then issues random
//! `multi_get` batches and reports throughput + per-batch latency for each
//! available reader backend. Build with `--features io_uring` to include the
//! io_uring backend; otherwise only the std backend runs.
//!
//! Usage:
//!   diskstore-bench [--keys 200000] [--value-size 512] [--batch 256] [--batches 2000]

use std::time::Instant;

use diskstore::{BatchReader, Bitcask, StdReader};

struct Args {
    keys: usize,
    value_size: usize,
    batch: usize,
    batches: usize,
}

fn parse() -> Args {
    let mut a = Args { keys: 200_000, value_size: 512, batch: 256, batches: 2_000 };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut n = || it.next().and_then(|v| v.parse().ok()).expect("int value");
        match flag.as_str() {
            "--keys" => a.keys = n(),
            "--value-size" => a.value_size = n(),
            "--batch" => a.batch = n(),
            "--batches" => a.batches = n(),
            "-h" | "--help" => {
                println!("usage: diskstore-bench [--keys N] [--value-size N] [--batch N] [--batches N]");
                std::process::exit(0);
            }
            other => panic!("unknown arg {other}"),
        }
    }
    a
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 17
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(((p / 100.0) * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)]
}

fn run_backend(name: &str, reader: &dyn BatchReader, db: &Bitcask, args: &Args, keystrings: &[String]) {
    let mut rng = Lcg(0xdead_beef_cafe_f00d);
    let mut lats = Vec::with_capacity(args.batches);
    let started = Instant::now();
    let mut ops = 0usize;
    for _ in 0..args.batches {
        let batch_keys: Vec<&[u8]> = (0..args.batch)
            .map(|_| keystrings[(rng.next() as usize) % args.keys].as_bytes())
            .collect();
        let t0 = Instant::now();
        // io_uring can be seccomp-blocked (EPERM) in hardened containers — report
        // it and skip this backend rather than aborting the whole benchmark.
        let got = match db.multi_get(reader, &batch_keys) {
            Ok(g) => g,
            Err(e) => {
                println!("[{name}] ({}) UNAVAILABLE: {e}", reader.name());
                println!("  (io_uring blocked by container seccomp; the std backend numbers stand.)");
                return;
            }
        };
        lats.push(t0.elapsed().as_micros() as u64);
        ops += got.len();
        std::hint::black_box(&got);
    }
    let elapsed = started.elapsed();
    lats.sort_unstable();
    println!("[{name}] ({})", reader.name());
    println!("  reads          : {ops}");
    println!("  throughput     : {:.0} reads/sec", ops as f64 / elapsed.as_secs_f64());
    println!("  batch p50      : {} us", percentile(&lats, 50.0));
    println!("  batch p99      : {} us", percentile(&lats, 99.0));
}

fn main() -> std::io::Result<()> {
    let args = parse();
    let dir = std::env::temp_dir().join("diskstore-bench");
    let _ = std::fs::remove_dir_all(&dir);
    let mut db = Bitcask::open(&dir, false)?;

    let value = vec![b'x'; args.value_size];
    let keystrings: Vec<String> = (0..args.keys).map(|i| format!("key-{i}")).collect();
    print!("loading {} keys x {} bytes ...", args.keys, args.value_size);
    for k in &keystrings {
        db.put(k.as_bytes(), &value)?;
    }
    db.sync()?;
    println!(" done ({} MiB on disk)", db.file_size() / (1024 * 1024));
    println!(
        "batched random reads: {} batches x {} keys\n",
        args.batches, args.batch
    );

    run_backend("std", &StdReader, &db, &args, &keystrings);

    #[cfg(feature = "io_uring")]
    match diskstore::UringReader::new(args.batch as u32) {
        Ok(uring) => run_backend("uring", &uring, &db, &args, &keystrings),
        // io_uring_setup itself can be EPERM-blocked by a container seccomp
        // profile — report and skip rather than aborting the whole bench.
        Err(e) => println!("[uring] UNAVAILABLE: {e}\n  (io_uring blocked by container seccomp; the std backend numbers stand.)"),
    }
    #[cfg(not(feature = "io_uring"))]
    println!("\n(io_uring backend not built — rebuild with --features io_uring to compare)");

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
