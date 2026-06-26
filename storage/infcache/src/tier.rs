// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Two-tier KV-block store: a bounded RAM hot tier over a durable SSD tier.
//!
//! `get` checks RAM, then SSD (promoting a hit back into RAM). Writes are
//! write-through: every block lands in both tiers, so the SSD tier is always
//! authoritative and a RAM eviction (silent, LRU) only costs a promotion on the
//! next access — never data. This mirrors a production KVCache where HBM/DRAM
//! holds the hot working set and an SSD/NVMe tier backs the long tail of cached
//! contexts.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use diskstore::Bitcask;
use kvcache::ShardedCache;

use crate::prefix::block_keys;

#[derive(Debug, Clone, Copy, Default)]
pub struct TierMetrics {
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub misses: u64,
    pub promotions: u64,
    pub stores: u64,
}

impl TierMetrics {
    pub fn hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l2_hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.l1_hits + self.l2_hits) as f64 / total as f64
        }
    }
}

/// Result of planning a prefill: how much of the prompt is already cached
/// (reusable as a contiguous prefix) vs. how much must be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPlan {
    pub total_tokens: usize,
    pub total_blocks: usize,
    pub reused_blocks: usize,
    pub reused_tokens: usize,
    pub compute_tokens: usize,
}

pub struct TieredKvCache {
    ram: ShardedCache,
    ssd: Mutex<Bitcask>,
    block_tokens: usize,
    l1_hits: AtomicU64,
    l2_hits: AtomicU64,
    misses: AtomicU64,
    promotions: AtomicU64,
    stores: AtomicU64,
}

impl TieredKvCache {
    /// Open a tiered cache. `ram_blocks` bounds the RAM tier (entries);
    /// `dir` holds the durable SSD tier. `sync` fsyncs each SSD write.
    pub fn open(
        dir: impl AsRef<Path>,
        block_tokens: usize,
        ram_shards: usize,
        ram_blocks: usize,
        sync: bool,
    ) -> std::io::Result<Self> {
        assert!(block_tokens > 0);
        Ok(TieredKvCache {
            ram: ShardedCache::new(ram_shards, ram_blocks),
            ssd: Mutex::new(Bitcask::open(dir, sync)?),
            block_tokens,
            l1_hits: AtomicU64::new(0),
            l2_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
            stores: AtomicU64::new(0),
        })
    }

    pub fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    /// Fetch one block by key, RAM → SSD with promotion on an SSD hit.
    pub fn get_block(&self, key: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
        if let Some(v) = self.ram.get(key) {
            self.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(v));
        }
        let from_ssd = self.ssd.lock().unwrap().get(key)?;
        match from_ssd {
            Some(v) => {
                self.l2_hits.fetch_add(1, Ordering::Relaxed);
                self.promotions.fetch_add(1, Ordering::Relaxed);
                self.ram.set(key, v.clone(), 0); // promote to hot tier
                Ok(Some(v))
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    /// Write a block through to both tiers.
    pub fn put_block(&self, key: &[u8], payload: &[u8]) -> std::io::Result<()> {
        self.ssd.lock().unwrap().put(key, payload)?;
        self.ram.set(key, payload.to_vec(), 0);
        self.stores.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Plan a prefill: probe cached blocks from the start and stop at the first
    /// miss (the reusable region is the contiguous cached *prefix*). Counts the
    /// probes in the tier metrics, so a plan also warms promotions.
    pub fn plan_prefill(&self, tokens: &[u32]) -> std::io::Result<PrefillPlan> {
        let keys = block_keys(tokens, self.block_tokens);
        let mut reused_blocks = 0usize;
        for key in &keys {
            if self.get_block(key)?.is_some() {
                reused_blocks += 1;
            } else {
                break;
            }
        }
        let reused_tokens = (reused_blocks * self.block_tokens).min(tokens.len());
        Ok(PrefillPlan {
            total_tokens: tokens.len(),
            total_blocks: keys.len(),
            reused_blocks,
            reused_tokens,
            compute_tokens: tokens.len() - reused_tokens,
        })
    }

    /// Store every block of a sequence (e.g. after a prefill computes the KV).
    /// `payload_for` produces the block payload given (block_index, key).
    pub fn store_sequence(
        &self,
        tokens: &[u32],
        mut payload_for: impl FnMut(usize, &[u8; 8]) -> Vec<u8>,
    ) -> std::io::Result<()> {
        for (b, key) in block_keys(tokens, self.block_tokens).iter().enumerate() {
            self.put_block(key, &payload_for(b, key))?;
        }
        Ok(())
    }

    pub fn metrics(&self) -> TierMetrics {
        TierMetrics {
            l1_hits: self.l1_hits.load(Ordering::Relaxed),
            l2_hits: self.l2_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            promotions: self.promotions.load(Ordering::Relaxed),
            stores: self.stores.load(Ordering::Relaxed),
        }
    }
}
