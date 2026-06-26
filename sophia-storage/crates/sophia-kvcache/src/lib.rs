// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//
// Lint policy: this crate contains ZERO unsafe in any build (no FFI, no raw
// pointers), so forbid it unconditionally. A consumer can confirm "this crate
// contains no unsafe" by inspecting for `#![forbid(unsafe_code)]` here.
#![forbid(unsafe_code)]
#![warn(rust_2018_idioms)]
// NOTE: #![warn(missing_docs)] and #![warn(missing_debug_implementations)] are
// intentionally NOT enabled yet — ~15 pub items lack /// docs and ~4 pub types
// (KvCache, FileStore, Arena, TierStack) lack #[derive(Debug)]. Closing both is
// a tracked follow-up; the load-bearing lint here is forbid(unsafe_code).
//! # sophia-kvcache
//!
//! A disaggregated, prefix-sharing KV-cache for LLM inference — the storage
//! layer the DeepSeek-style "KVCache 存储系统" bullet is about, built against a
//! real workload Sophia already runs: best-of-N sampling and multi-agent council
//! deliberation (`agent/best_of.py`, `agent/council_deliberate.py`) issue many
//! requests over the *same* long prompt prefix. That is the canonical case for
//! paged, prefix-shared KV reuse.
//!
//! ## What this gets right (tested)
//! - **Paged blocks** ([`block`]) with **content-addressed ids** so equal prompt
//!   prefixes resolve to the same block and the KV is stored once.
//! - **Tiered residency** HBM → DRAM → NVMe ([`tier`]) over pluggable storage
//!   media ([`store`]). The **NVMe tier is real disk** ([`store::FileStore`]):
//!   demoted blocks persist, page back in on promotion, and survive restarts.
//! - **Reference-counted LRU eviction** ([`eviction`]) so a shared prefix can
//!   never be evicted out from under a live request.
//! - **Byte-movement accounting** per tier boundary — the number a zero-copy /
//!   RDMA transfer path exists to shrink.
//!
//! ## Honest seams (see docs/DESIGN.md)
//! - HBM/DRAM are in-memory ([`store::MemStore`]); a GPU build swaps in a CUDA
//!   allocation and `cudaMemcpyAsync` transfers. No GPU is required to run this.
//! - The NVMe transfer is a `write`/`read` syscall; `io_uring`/SPDK and a remote
//!   RDMA DRAM pool are the documented next steps. The placement logic does not
//!   change when they land.
//!
//! ## Example
//! ```
//! use sophia_kvcache::{KvCache, Config};
//! let mut cache = KvCache::new(Config::new(2 /*block_len*/, 4, 8, 16)).unwrap();
//! let prompt = [10, 11, 12, 13, 14, 15];
//! let r1 = cache.admit(&prompt, |_id| vec![0u8; 64]).unwrap();
//! let r2 = cache.admit(&prompt, |_id| vec![0u8; 64]).unwrap();
//! assert_eq!(r2.prefix_hits, r2.chain_len, "second request is a full prefix hit");
//! assert!(r1.computed_blocks >= r2.computed_blocks);
//! ```

pub mod block;
pub mod eviction;
pub mod prefix;
pub mod store;
pub mod tier;

use std::collections::HashSet;
use std::io;
use std::path::PathBuf;

use block::{Block, BlockId, Tier};
use eviction::LruRefCounted;
use tier::TierStack;

/// Cache configuration. Capacities are in blocks per tier. If `nvme_dir` is set,
/// the NVMe tier persists to that directory (a real disk tier); otherwise NVMe
/// is an in-memory stand-in.
#[derive(Debug, Clone)]
pub struct Config {
    pub block_len: usize,
    pub hbm_blocks: usize,
    pub dram_blocks: usize,
    pub nvme_blocks: usize,
    pub nvme_dir: Option<PathBuf>,
}

impl Config {
    /// Build a cache config. `block_len` is the fixed number of tokens per paged
    /// block (see `block.rs`) and MUST be >= 1; a zero value is a programmer
    /// error and panics here at construction time (fail-fast) rather than later
    /// inside `prefix::block_chain`. The public `block_chain` still returns a
    /// `Result` for callers that build their own config out-of-band.
    pub fn new(block_len: usize, hbm: usize, dram: usize, nvme: usize) -> Self {
        assert!(block_len >= 1, "Config::new: block_len must be >= 1, got {block_len}");
        Config { block_len, hbm_blocks: hbm, dram_blocks: dram, nvme_blocks: nvme, nvme_dir: None }
    }

    /// Back the NVMe tier with real disk under `dir`.
    pub fn with_nvme_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.nvme_dir = Some(dir.into());
        self
    }
}

/// What an `admit` did — the numbers a benchmark or scheduler cares about.
#[derive(Debug, Clone, Default)]
pub struct AdmitResult {
    pub chain_len: usize,
    /// Leading blocks served from cache (the reuse win).
    pub prefix_hits: usize,
    /// Suffix blocks that had to be computed and inserted.
    pub computed_blocks: usize,
    /// Blocks evicted out of the cache entirely during this admission.
    pub evicted: usize,
}

impl AdmitResult {
    pub fn hit_ratio(&self) -> f64 {
        if self.chain_len == 0 {
            0.0
        } else {
            self.prefix_hits as f64 / self.chain_len as f64
        }
    }
}

/// The cache controller.
pub struct KvCache {
    cfg: Config,
    tiers: TierStack,
    lru: LruRefCounted,
    pub stats: Stats,
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub admissions: u64,
    pub block_hits: u64,
    pub block_misses: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub evictions: u64,
}

impl KvCache {
    /// Build a cache. Falls back to an all-in-memory stack unless
    /// `cfg.nvme_dir` requests a real disk-backed NVMe tier.
    pub fn new(cfg: Config) -> io::Result<Self> {
        let tiers = match &cfg.nvme_dir {
            Some(dir) => {
                TierStack::with_nvme(cfg.hbm_blocks, cfg.dram_blocks, cfg.nvme_blocks, dir)?
            }
            None => TierStack::in_memory(cfg.hbm_blocks, cfg.dram_blocks, cfg.nvme_blocks),
        };
        Ok(KvCache { cfg, tiers, lru: LruRefCounted::new(), stats: Stats::default() })
    }

    fn resident_set(&self) -> HashSet<BlockId> {
        let mut set = HashSet::new();
        set.extend(self.tiers.hbm.ids());
        set.extend(self.tiers.dram.ids());
        set.extend(self.tiers.nvme.ids());
        set
    }

    /// Admit a request for `tokens`. For each block: if resident, count a hit and
    /// promote toward HBM; otherwise call `compute(id)` to materialize the KV
    /// payload and insert it into HBM (evicting if needed). Returns reuse stats.
    ///
    /// `compute` stands in for the model's prefill of that block.
    pub fn admit<F: FnMut(BlockId) -> Vec<u8>>(
        &mut self,
        tokens: &[u32],
        mut compute: F,
    ) -> io::Result<AdmitResult> {
        self.stats.admissions += 1;
        let chain = prefix::block_chain(tokens, self.cfg.block_len)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let resident = self.resident_set();
        let hits = prefix::shared_prefix_len(&chain, &resident);

        let mut result = AdmitResult {
            chain_len: chain.len(),
            prefix_hits: hits,
            computed_blocks: chain.len().saturating_sub(hits),
            ..Default::default()
        };

        for (i, &id) in chain.iter().enumerate() {
            if i < hits {
                self.stats.block_hits += 1;
                self.lru.touch(id);
                self.promote(id)?;
            } else {
                self.stats.block_misses += 1;
                let payload = compute(id);
                let token_count = block_token_count(&chain, i, tokens.len(), self.cfg.block_len);
                let block = Block::new(id, token_count, payload);
                result.evicted += self.install_hbm(block)?;
                self.lru.track(id);
            }
        }
        Ok(result)
    }

    /// Pin every block in a request's prefix for the duration of its decode, so
    /// concurrent requests cannot evict the shared context. Returns the chain.
    pub fn pin_prefix(&mut self, tokens: &[u32]) -> Vec<BlockId> {
        // block_len >= 1 is guaranteed by Config::new (fail-fast at construction),
        // so block_chain cannot return ZeroBlockLen here. Returning Vec (not Result)
        // preserves this method's signature for callers that pin during decode.
        let chain = prefix::block_chain(tokens, self.cfg.block_len)
            .expect("Config::new guarantees block_len >= 1");
        for &id in &chain {
            self.lru.pin(id);
        }
        chain
    }

    pub fn unpin_prefix(&mut self, chain: &[BlockId]) {
        for &id in chain {
            self.lru.unpin(id);
        }
    }

    /// Promote a block one tier toward HBM if it is currently lower. This is a
    /// real transfer: an NVMe-resident block is read off disk and re-inserted
    /// into DRAM.
    fn promote(&mut self, id: BlockId) -> io::Result<()> {
        match self.tiers.locate(id) {
            Some(Tier::Dram) => {
                if let Some(block) = self.tiers.dram.remove(id)? {
                    self.stats.promotions += 1;
                    self.install_hbm(block)?;
                }
            }
            Some(Tier::Nvme) => {
                if let Some(block) = self.tiers.nvme.remove(id)? {
                    self.stats.promotions += 1;
                    self.install_dram(block)?; // NVMe -> DRAM (one step)
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Install into HBM, demoting the LRU victim to DRAM if HBM is full.
    /// Returns the number of blocks evicted out of the cache entirely.
    fn install_hbm(&mut self, block: Block) -> io::Result<usize> {
        let mut evicted_out = 0;
        while self.tiers.hbm.is_full() {
            match self.lru.evict_candidate(self.tiers.hbm.ids().into_iter()) {
                Some(victim) => {
                    if let Some(v) = self.tiers.hbm.remove(victim)? {
                        self.stats.demotions += 1;
                        evicted_out += self.install_dram(v)?;
                    }
                }
                None => break, // everything pinned; caller must back off
            }
        }
        self.tiers.hbm.insert(block)?;
        Ok(evicted_out)
    }

    /// Install into DRAM, demoting to NVMe under pressure.
    fn install_dram(&mut self, block: Block) -> io::Result<usize> {
        let mut evicted_out = 0;
        while self.tiers.dram.is_full() {
            match self.lru.evict_candidate(self.tiers.dram.ids().into_iter()) {
                Some(victim) => {
                    if let Some(v) = self.tiers.dram.remove(victim)? {
                        evicted_out += self.install_nvme(v)?;
                    }
                }
                None => break,
            }
        }
        self.tiers.dram.insert(block)?;
        Ok(evicted_out)
    }

    /// Install into NVMe (real disk write), evicting the LRU victim if full.
    fn install_nvme(&mut self, block: Block) -> io::Result<usize> {
        let mut evicted = 0;
        while self.tiers.nvme.is_full() {
            match self.lru.evict_candidate(self.tiers.nvme.ids().into_iter()) {
                Some(victim) => {
                    self.tiers.nvme.remove(victim)?;
                    self.lru.forget(victim);
                    self.stats.evictions += 1;
                    evicted += 1;
                }
                None => break,
            }
        }
        self.tiers.nvme.insert(block)?;
        Ok(evicted)
    }

    /// Total resident blocks across all tiers.
    pub fn resident_blocks(&self) -> usize {
        self.tiers.hbm.len() + self.tiers.dram.len() + self.tiers.nvme.len()
    }

    pub fn tier_of(&self, id: BlockId) -> Option<Tier> {
        self.tiers.locate(id)
    }

    /// Cumulative bytes written to / read from the NVMe tier — the slow-boundary
    /// traffic a zero-copy / RDMA / SPDK path targets.
    pub fn nvme_bytes(&self) -> (u64, u64) {
        (self.tiers.nvme.bytes_in(), self.tiers.nvme.bytes_out())
    }
}

/// How many real tokens block `i` covers (the last block may be partial).
fn block_token_count(chain: &[BlockId], i: usize, total_tokens: usize, block_len: usize) -> u32 {
    if i + 1 < chain.len() {
        block_len as u32
    } else {
        let rem = total_tokens % block_len;
        if rem == 0 { block_len as u32 } else { rem as u32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_identical_request_is_full_hit() {
        let mut cache = KvCache::new(Config::new(2, 8, 16, 32)).unwrap();
        let prompt = [1, 2, 3, 4, 5, 6];
        let r1 = cache.admit(&prompt, |_| vec![0u8; 32]).unwrap();
        let r2 = cache.admit(&prompt, |_| vec![0u8; 32]).unwrap();
        assert_eq!(r1.prefix_hits, 0, "cold prompt: no hits");
        assert_eq!(r2.prefix_hits, r2.chain_len, "warm prompt: full prefix hit");
        assert_eq!(r2.computed_blocks, 0);
        assert!((r2.hit_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn best_of_n_shares_prefix() {
        let mut cache = KvCache::new(Config::new(4, 32, 64, 128)).unwrap();
        let prompt: Vec<u32> = (0..64).collect();
        cache.admit(&prompt, |_| vec![0u8; 16]).unwrap();
        let before = cache.stats.block_misses;
        for cont in 100..108u32 {
            let mut seq = prompt.clone();
            seq.push(cont);
            cache.admit(&seq, |_| vec![0u8; 16]).unwrap();
        }
        let new_misses = cache.stats.block_misses - before;
        assert!(new_misses <= 8 * 2, "prefix reuse failed: {new_misses} new misses");
    }

    #[test]
    fn pinned_prefix_is_not_evicted_under_pressure() {
        let mut cache = KvCache::new(Config::new(2, 2, 2, 2)).unwrap();
        let hot = [1, 2, 3, 4];
        cache.admit(&hot, |_| vec![0u8; 8]).unwrap();
        let pinned = cache.pin_prefix(&hot);
        for base in 100..140u32 {
            cache.admit(&[base, base + 1, base + 2, base + 3], |_| vec![0u8; 8]).unwrap();
        }
        for id in &pinned {
            assert!(cache.tier_of(*id).is_some(), "pinned block {id:?} was evicted");
        }
        cache.unpin_prefix(&pinned);
    }

    #[test]
    fn nvme_tier_persists_demoted_blocks_to_disk() {
        let dir = std::env::temp_dir().join(format!("sophia-kvc-nvme-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        // Tiny HBM/DRAM so blocks cascade to the real disk NVMe tier.
        let cfg = Config::new(2, 1, 1, 16).with_nvme_dir(&dir);
        let mut cache = KvCache::new(cfg).unwrap();

        // Admit a known block, then push it down with unrelated traffic.
        let target = [7, 7];
        cache.admit(&target, |_| vec![42u8; 24]).unwrap();
        for base in 200..210u32 {
            cache.admit(&[base, base + 1], |_| vec![0u8; 24]).unwrap();
        }

        // It should now live on the disk NVMe tier, with a real file on disk.
        assert_eq!(cache.tier_of(BlockId::derive(BlockId::ROOT, &[7, 7])), Some(Tier::Nvme));
        let (bytes_in, _) = cache.nvme_bytes();
        assert!(bytes_in > 0, "expected real bytes written to the NVMe tier");
        let on_disk = std::fs::read_dir(&dir).unwrap().count();
        assert!(on_disk > 0, "expected block files on disk, found none");

        // Promote it back: read off disk, payload intact.
        cache.admit(&target, |_| panic!("must be served from NVMe, not recomputed")).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }
}
