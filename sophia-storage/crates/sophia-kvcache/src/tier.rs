// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Tiered residency: HBM → DRAM → NVMe.
//!
//! Each tier is a capacity-bounded [`Arena`] backed by a [`BlockStore`]
//! ([`crate::store`]). The placement/promotion/demotion logic lives in the
//! controller ([`crate::KvCache`]) and is identical no matter which medium backs
//! a tier — that is the whole point of the split:
//!
//! - HBM and DRAM are backed by [`MemStore`] (device-memory stand-in).
//! - NVMe is backed by [`FileStore`] — **real disk**, so demoted blocks persist.
//!
//! The transfer methods that move bytes between tiers are `store.take` +
//! `store.put`; that pair is where `cudaMemcpyAsync` (HBM↔DRAM) and RDMA / SPDK
//! (DRAM↔NVMe / remote) replace the syscall path without touching anything here.

use std::io;
use std::path::Path;

use crate::block::{Block, BlockId, Tier};
use crate::store::{BlockStore, FileStore, MemStore};

/// One capacity-bounded arena over some storage medium. Capacity is counted in
/// blocks; a byte-budget variant is a trivial change.
pub struct Arena {
    pub tier: Tier,
    capacity_blocks: usize,
    store: Box<dyn BlockStore>,
}

impl Arena {
    /// In-memory arena (HBM/DRAM).
    pub fn memory(tier: Tier, capacity_blocks: usize) -> Self {
        Arena { tier, capacity_blocks, store: Box::new(MemStore::new()) }
    }

    /// Disk-backed arena (NVMe), persisting blocks under `dir`.
    pub fn file(tier: Tier, capacity_blocks: usize, dir: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Arena { tier, capacity_blocks, store: Box::new(FileStore::open(dir)?) })
    }

    pub fn contains(&self, id: BlockId) -> bool {
        self.store.contains(id)
    }

    pub fn get(&self, id: BlockId) -> io::Result<Option<Block>> {
        self.store.get(id)
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.store.len() >= self.capacity_blocks
    }

    pub fn capacity(&self) -> usize {
        self.capacity_blocks
    }

    /// Resident block ids (eviction-candidate enumeration).
    pub fn ids(&self) -> Vec<BlockId> {
        self.store.ids()
    }

    /// Insert without an eviction decision (caller guarantees space).
    pub fn insert(&mut self, block: Block) -> io::Result<()> {
        self.store.put(block)
    }

    /// Remove and return a block (a demotion/promotion source).
    pub fn remove(&mut self, id: BlockId) -> io::Result<Option<Block>> {
        self.store.take(id)
    }

    /// Cumulative bytes written into / read out of this tier's medium.
    pub fn bytes_in(&self) -> u64 {
        self.store.bytes_in()
    }
    pub fn bytes_out(&self) -> u64 {
        self.store.bytes_out()
    }
}

/// The full residency hierarchy.
pub struct TierStack {
    pub hbm: Arena,
    pub dram: Arena,
    pub nvme: Arena,
}

impl TierStack {
    /// All-in-memory stack (NVMe is an in-memory stand-in). Used when no NVMe
    /// directory is configured — keeps the cache usable with zero disk setup.
    pub fn in_memory(hbm_blocks: usize, dram_blocks: usize, nvme_blocks: usize) -> Self {
        TierStack {
            hbm: Arena::memory(Tier::Hbm, hbm_blocks),
            dram: Arena::memory(Tier::Dram, dram_blocks),
            nvme: Arena::memory(Tier::Nvme, nvme_blocks),
        }
    }

    /// Stack with a **real disk-backed NVMe tier** under `nvme_dir`.
    pub fn with_nvme(
        hbm_blocks: usize,
        dram_blocks: usize,
        nvme_blocks: usize,
        nvme_dir: impl AsRef<Path>,
    ) -> io::Result<Self> {
        Ok(TierStack {
            hbm: Arena::memory(Tier::Hbm, hbm_blocks),
            dram: Arena::memory(Tier::Dram, dram_blocks),
            nvme: Arena::file(Tier::Nvme, nvme_blocks, nvme_dir)?,
        })
    }

    pub fn arena(&self, tier: Tier) -> &Arena {
        match tier {
            Tier::Hbm => &self.hbm,
            Tier::Dram => &self.dram,
            Tier::Nvme => &self.nvme,
        }
    }

    pub fn arena_mut(&mut self, tier: Tier) -> &mut Arena {
        match tier {
            Tier::Hbm => &mut self.hbm,
            Tier::Dram => &mut self.dram,
            Tier::Nvme => &mut self.nvme,
        }
    }

    /// Find which tier currently holds `id`, if any. In-memory, infallible.
    pub fn locate(&self, id: BlockId) -> Option<Tier> {
        if self.hbm.contains(id) {
            Some(Tier::Hbm)
        } else if self.dram.contains(id) {
            Some(Tier::Dram)
        } else if self.nvme.contains(id) {
            Some(Tier::Nvme)
        } else {
            None
        }
    }
}
