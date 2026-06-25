// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Dependency-free benchmark for the KV-cache, modeling Sophia's actual reuse
//! pattern: a shared council/best-of-N prompt prefix with divergent suffixes.
//!
//! It reports the prefix hit-ratio and the prefill-block savings vs. a no-cache
//! baseline — the number that translates directly into inference cost. Run:
//!   cargo run --release --bench kvcache_bench

use std::time::Instant;

use sophia_kvcache::{Config, KvCache};

fn main() {
    let prompt_tokens: usize =
        std::env::var("PROMPT").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let fanout: usize = std::env::var("FANOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let block_len = 16;

    // Sized so the shared prefix fits comfortably; suffixes churn the tail.
    let prefix_blocks = prompt_tokens / block_len;
    let cfg = Config::new(block_len, prefix_blocks * 2, prefix_blocks * 4, prefix_blocks * 8);
    let mut cache = KvCache::new(cfg);

    println!(
        "sophia-kvcache bench  (prompt={prompt_tokens} tok, fanout={fanout}, rounds={rounds}, block_len={block_len})"
    );

    let mut prefill_no_cache: u64 = 0;
    let mut prefill_with_cache: u64 = 0;
    let mut hit_ratio_sum = 0.0;
    let mut admissions = 0u64;

    let t0 = Instant::now();
    for r in 0..rounds {
        // A fresh shared prompt per round (simulates a new user turn), reused
        // across `fanout` divergent samples — the council / best-of-N shape.
        let base: u32 = (r as u32) * 1_000_003;
        let prompt: Vec<u32> = (0..prompt_tokens as u32).map(|i| base ^ i).collect();

        // Warm the shared prefix once, pin it for the fan-out.
        cache.admit(&prompt, |_| vec![0u8; 256]);
        let pinned = cache.pin_prefix(&prompt);

        for s in 0..fanout as u32 {
            let mut seq = prompt.clone();
            seq.extend([900_000 + s, 900_001 + s, 900_002 + s]); // divergent suffix
            let res = cache.admit(&seq, |_| vec![0u8; 256]);
            prefill_no_cache += res.chain_len as u64;
            prefill_with_cache += res.computed_blocks as u64;
            hit_ratio_sum += res.hit_ratio();
            admissions += 1;
        }
        cache.unpin_prefix(&pinned);
    }
    let elapsed = t0.elapsed();

    let saved = prefill_no_cache.saturating_sub(prefill_with_cache);
    let pct = if prefill_no_cache > 0 {
        100.0 * saved as f64 / prefill_no_cache as f64
    } else {
        0.0
    };

    println!("admissions:            {admissions}");
    println!("avg prefix hit-ratio:  {:.3}", hit_ratio_sum / admissions as f64);
    println!("prefill blocks (naive):{prefill_no_cache}");
    println!("prefill blocks (cache):{prefill_with_cache}");
    println!("blocks saved:          {saved}  ({pct:.1}% prefill avoided)");
    println!("resident blocks:       {}", cache.resident_blocks());
    println!("stats:                 {:?}", cache.stats);
    println!(
        "throughput:            {:.0} admissions/s",
        admissions as f64 / elapsed.as_secs_f64()
    );
}
