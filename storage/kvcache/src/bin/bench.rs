// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! kvcache-bench — closed-loop load generator against an in-process server.
//!
//! Spawns the real TCP server on an ephemeral port, preloads `--keys` entries,
//! then drives `--clients` concurrent connections each issuing `--ops` GETs and
//! reports throughput plus p50/p99/p999 round-trip latency. Honest numbers: this
//! measures the full client→TCP→server→TCP→client path on loopback, not just the
//! in-memory map.
//!
//! Usage:
//!   kvcache-bench [--clients 32] [--ops 50000] [--keys 100000]
//!                 [--value-size 256] [--shards 16] [--write-frac 0.0]

use std::sync::Arc;
use std::time::Instant;

use kvcache::{serve, Client, Request, ShardedCache};
use tokio::net::TcpListener;

struct Args {
    clients: usize,
    ops: usize,
    keys: usize,
    value_size: usize,
    shards: usize,
    write_frac: f64,
    pipeline: usize,
}

fn parse() -> Args {
    let mut a = Args {
        clients: 32,
        ops: 50_000,
        keys: 100_000,
        value_size: 256,
        shards: 16,
        write_frac: 0.0,
        pipeline: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut next_usize = || it.next().and_then(|v| v.parse().ok()).expect("int value");
        match flag.as_str() {
            "--clients" => a.clients = next_usize(),
            "--ops" => a.ops = next_usize(),
            "--keys" => a.keys = next_usize(),
            "--value-size" => a.value_size = next_usize(),
            "--shards" => a.shards = next_usize(),
            "--pipeline" => a.pipeline = next_usize().max(1),
            "--write-frac" => a.write_frac = it.next().and_then(|v| v.parse().ok()).expect("float"),
            "-h" | "--help" => {
                println!("usage: kvcache-bench [--clients N] [--ops N] [--keys N] [--value-size N] [--shards N] [--pipeline DEPTH] [--write-frac F]");
                std::process::exit(0);
            }
            other => panic!("unknown arg {other}"),
        }
    }
    a
}

/// Cheap deterministic PRNG (PCG-style LCG) so the benchmark needs no rand crate
/// and is reproducible run to run.
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
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let args = parse();
    let cache = Arc::new(ShardedCache::new(args.shards, args.keys.max(args.clients * args.ops)));
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(serve(listener, Arc::clone(&cache)));

    let value = vec![b'x'; args.value_size];

    // Preload.
    {
        let mut c = Client::connect(addr).await?;
        for i in 0..args.keys {
            c.set(format!("key-{i}").as_bytes(), &value, 0).await?;
        }
    }

    println!(
        "kvcache-bench: {} clients x {} ops, {} keys, {}-byte values, {} shards, write_frac={}",
        args.clients, args.ops, args.keys, args.value_size, args.shards, args.write_frac
    );

    let started = Instant::now();
    let mut handles = Vec::with_capacity(args.clients);
    for cid in 0..args.clients {
        let value = value.clone();
        let (ops, keys, write_frac, depth) = (args.ops, args.keys, args.write_frac, args.pipeline);
        handles.push(tokio::spawn(async move {
            let mut client = Client::connect(addr).await.expect("connect");
            let mut rng = Lcg(0x9e3779b97f4a7c15 ^ (cid as u64).wrapping_mul(0xbf58476d1ce4e5b9));
            let mut lats = Vec::with_capacity(ops / depth + 1);
            let write_cut = (write_frac * u32::MAX as f64) as u64;
            let mut done = 0;
            while done < ops {
                let batch = depth.min(ops - done);
                // Build a batch of requests; depth==1 degenerates to one op.
                let reqs: Vec<Request> = (0..batch)
                    .map(|_| {
                        let r = rng.next();
                        let key = format!("key-{}", r % keys as u64).into_bytes();
                        if (r & 0xffff_ffff) < write_cut {
                            Request::Set { key, val: value.clone(), ttl_ms: 0 }
                        } else {
                            Request::Get(key)
                        }
                    })
                    .collect();
                let t0 = Instant::now();
                client.pipeline(&reqs).await.expect("pipeline");
                lats.push(t0.elapsed().as_micros() as u64); // per-batch latency
                done += batch;
            }
            lats
        }));
    }

    let mut all = Vec::new(); // per-batch latencies (== per-op when depth==1)
    for h in handles {
        all.extend(h.await.expect("client task"));
    }
    let elapsed = started.elapsed();
    all.sort_unstable();

    let total_ops = (args.clients * args.ops) as f64;
    let qps = total_ops / elapsed.as_secs_f64();
    let stats = cache.stats();
    let lat_unit = if args.pipeline == 1 {
        "per-op".to_string()
    } else {
        format!("per-batch (depth={})", args.pipeline)
    };

    println!("--- results ---");
    println!("total ops      : {}", args.clients * args.ops);
    println!("wall time      : {:.3} s", elapsed.as_secs_f64());
    println!("throughput     : {:.0} ops/sec", qps);
    println!("latency p50    : {} us ({lat_unit})", percentile(&all, 50.0));
    println!("latency p99    : {} us ({lat_unit})", percentile(&all, 99.0));
    println!("latency p99.9  : {} us ({lat_unit})", percentile(&all, 99.9));
    println!("latency max    : {} us ({lat_unit})", all.last().copied().unwrap_or(0));
    println!(
        "server stats   : hits={} misses={} sets={} evictions={} entries={}",
        stats.hits, stats.misses, stats.sets, stats.evictions, stats.entries
    );
    Ok(())
}
