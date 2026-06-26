// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! infcache-bench — demonstrate prefix-cache savings for a shared-prompt workload.
//!
//! Simulates many inference requests that share a long system prompt and each
//! add a unique suffix — the common production pattern (a fixed instruction
//! prefix + per-user query). Reports the prefill token reuse rate, i.e. the
//! fraction of prompt tokens whose KV is served from cache instead of recomputed.
//!
//! Usage:
//!   infcache-bench [--requests 2000] [--system 2048] [--suffix 128]
//!                  [--block 16] [--ram-blocks 20000] [--kv-bytes 4096]

use std::time::Instant;

use infcache::TieredKvCache;

struct Args {
    requests: usize,
    system: usize,
    suffix: usize,
    block: usize,
    ram_blocks: usize,
    kv_bytes: usize,
}

fn parse() -> Args {
    let mut a = Args { requests: 2000, system: 2048, suffix: 128, block: 16, ram_blocks: 20_000, kv_bytes: 4096 };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut n = || it.next().and_then(|v| v.parse().ok()).expect("int value");
        match flag.as_str() {
            "--requests" => a.requests = n(),
            "--system" => a.system = n(),
            "--suffix" => a.suffix = n(),
            "--block" => a.block = n(),
            "--ram-blocks" => a.ram_blocks = n(),
            "--kv-bytes" => a.kv_bytes = n(),
            "-h" | "--help" => {
                println!("usage: infcache-bench [--requests N] [--system N] [--suffix N] [--block N] [--ram-blocks N] [--kv-bytes N]");
                std::process::exit(0);
            }
            other => panic!("unknown arg {other}"),
        }
    }
    a
}

fn main() -> std::io::Result<()> {
    let args = parse();
    let dir = std::env::temp_dir().join("infcache-bench");
    let _ = std::fs::remove_dir_all(&dir);
    let cache = TieredKvCache::open(&dir, args.block, 16, args.ram_blocks, false)?;

    // Shared system prompt (token ids 1..system).
    let system: Vec<u32> = (1..=args.system as u32).collect();
    // Per-block payload size ~ block_tokens * per-token KV bytes.
    let payload_bytes = args.block * (args.kv_bytes / args.block).max(1);
    let make = |_: usize, _: &[u8; 8]| vec![0u8; payload_bytes];

    let mut total_prompt_tokens = 0usize;
    let mut reused_prompt_tokens = 0usize;

    let started = Instant::now();
    for r in 0..args.requests {
        // Unique suffix per request.
        let base = 1_000_000u32 + (r as u32) * (args.suffix as u32 + 1);
        let mut tokens = system.clone();
        tokens.extend((0..args.suffix as u32).map(|i| base + i));

        let plan = cache.plan_prefill(&tokens)?;
        total_prompt_tokens += plan.total_tokens;
        reused_prompt_tokens += plan.reused_tokens;

        // Compute (here: synthesize) and store the full sequence's blocks.
        cache.store_sequence(&tokens, make)?;
    }
    let elapsed = started.elapsed();
    let m = cache.metrics();

    println!("infcache-bench: {} requests, system={} tok, suffix={} tok, block={}, kv/block={} B",
        args.requests, args.system, args.suffix, args.block, payload_bytes);
    println!("--- results ---");
    println!("prompt tokens (total) : {total_prompt_tokens}");
    println!("prompt tokens reused  : {reused_prompt_tokens}");
    println!("token reuse rate      : {:.1}%", 100.0 * reused_prompt_tokens as f64 / total_prompt_tokens as f64);
    println!("block hit rate        : {:.1}%", 100.0 * m.hit_rate());
    println!("tiers                 : l1_hits={} l2_hits={} misses={} promotions={} stores={}",
        m.l1_hits, m.l2_hits, m.misses, m.promotions, m.stores);
    println!("wall time             : {:.3} s ({:.0} req/s)",
        elapsed.as_secs_f64(), args.requests as f64 / elapsed.as_secs_f64());

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
