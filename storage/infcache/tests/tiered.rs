// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Tests for the tiered prefix KV-block cache: prefix reuse, RAM↔SSD tiering
//! with promotion, and durability across reopen.

use std::path::PathBuf;

use infcache::TieredKvCache;

struct TmpDir(PathBuf);
impl TmpDir {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        let tid: String = format!("{:?}", std::thread::current().id()).chars().filter(|c| c.is_alphanumeric()).collect();
        p.push(format!("infcache-test-{tag}-{tid}"));
        let _ = std::fs::remove_dir_all(&p);
        TmpDir(p)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn payload(key: &[u8; 8]) -> Vec<u8> {
    // Stand-in for a block's serialized K/V tensor.
    let mut v = vec![0u8; 256];
    v[..8].copy_from_slice(key);
    v
}

#[test]
fn shared_prompt_prefix_is_reused() {
    let dir = TmpDir::new("prefix");
    let cache = TieredKvCache::open(dir.path(), 16, 8, 100_000, false).unwrap();

    // A 320-token system prompt shared by every request (20 blocks of 16).
    let system: Vec<u32> = (1000..1320).collect();

    // First request: shared prompt + unique suffix. Nothing cached yet.
    let mut req1 = system.clone();
    req1.extend(2000..2040);
    let plan1 = cache.plan_prefill(&req1).unwrap();
    assert_eq!(plan1.reused_blocks, 0, "cold cache should reuse nothing");
    cache.store_sequence(&req1, |_, k| payload(k)).unwrap();

    // Second request: same system prompt, different suffix.
    let mut req2 = system.clone();
    req2.extend(3000..3050);
    let plan2 = cache.plan_prefill(&req2).unwrap();
    // The 20 full system-prompt blocks must be reused; the suffix is new.
    assert_eq!(plan2.reused_blocks, 20, "shared prefix not reused");
    assert_eq!(plan2.reused_tokens, 320);
    assert_eq!(plan2.compute_tokens, plan2.total_tokens - 320);

    let m = cache.metrics();
    assert!(m.l1_hits + m.l2_hits >= 20);
}

#[test]
fn ssd_tier_serves_after_ram_eviction_and_promotes() {
    let dir = TmpDir::new("tier");
    // RAM holds only 4 blocks; we store far more so RAM must evict.
    let cache = TieredKvCache::open(dir.path(), 8, 2, 4, false).unwrap();

    let seqs: Vec<Vec<u32>> = (0..50)
        .map(|s| ((s * 100)..(s * 100 + 8)).collect())
        .collect();
    for seq in &seqs {
        cache.store_sequence(seq, |_, k| payload(k)).unwrap();
    }

    // The earliest sequences were almost certainly evicted from the 4-slot RAM
    // tier, but the SSD tier must still serve them (with a promotion).
    let before = cache.metrics();
    let plan = cache.plan_prefill(&seqs[0]).unwrap();
    assert_eq!(plan.reused_blocks, 1, "block lost across tiers");
    let after = cache.metrics();
    assert!(after.l2_hits > before.l2_hits, "expected an SSD-tier hit");
    assert!(after.promotions > before.promotions, "SSD hit should promote to RAM");
}

#[test]
fn blocks_survive_reopen() {
    let dir = TmpDir::new("durable");
    let seq: Vec<u32> = (5000..5128).collect();
    {
        let cache = TieredKvCache::open(dir.path(), 16, 4, 1000, true).unwrap();
        cache.store_sequence(&seq, |_, k| payload(k)).unwrap();
    }
    // Reopen: RAM tier is empty, but the durable SSD tier (diskstore) recovers.
    let cache = TieredKvCache::open(dir.path(), 16, 4, 1000, true).unwrap();
    let plan = cache.plan_prefill(&seq).unwrap();
    assert_eq!(plan.reused_blocks, plan.total_blocks, "blocks did not survive reopen");
    assert!(cache.metrics().l2_hits > 0, "should have served from durable tier");
}
