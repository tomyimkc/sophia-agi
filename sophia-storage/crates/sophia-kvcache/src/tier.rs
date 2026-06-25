// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Tiered residency: HBM → DRAM → NVMe.
//!
//! Each tier is a capacity-bounded arena. The skeleton models every tier as an
//! in-memory map (NVMe included) because the *placement* and *promotion/demotion*
//! logic is what we want to exercise and test; swapping the NVMe arena for an
//! mmap'd file or the HBM arena for a CUDA allocation does not change that logic.
//!
//! The transfer methods are where the zero-copy / RDMA work lands: today they
//! `clone` payloads; the production path would `cudaMemcpyAsync` (HBM↔DRAM) and
//! issue RDMA reads (DRAM↔remote) / `io_uring` reads (DRAM↔NVMe).

use std::collections::HashMap;

use crate::block::{Block, BlockId, Tier};

/// One capacity-bounded arena. Capacity is counted in *blocks* for clarity;
/// a byte-budget variant is a trivial change.
pub struct Arena {
    pub tier: Tier,
    capacity_blocks: usize,
    blocks: HashMap<BlockId, Block>,
}

impl Arena {
    pub fn new(tier: Tier, capacity_blocks: usize) -> Self {
        Arena { tier, capacity_blocks, blocks: HashMap::new() }
    }

    pub fn contains(&self, id: BlockId) -> bool {
        self.blocks.contains_key(&id)
    }

    pub fn get(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(&id)
    }

    /// Snapshot of resident block ids (eviction-candidate enumeration).
    pub fn ids(&self) -> Vec<BlockId> {
        self.blocks.keys().copied().collect()
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.blocks.len() >= self.capacity_blocks
    }

    pub fn capacity(&self) -> usize {
        self.capacity_blocks
    }

    /// Insert without an eviction decision (caller guarantees space).
    pub fn insert(&mut self, block: Block) {
        self.blocks.insert(block.id, block);
    }

    /// Remove and return a block (a demotion source).
    pub fn remove(&mut self, id: BlockId) -> Option<Block> {
        self.blocks.remove(&id)
    }
}

/// The full residency hierarchy.
pub struct TierStack {
    pub hbm: Arena,
    pub dram: Arena,
    pub nvme: Arena,
}

impl TierStack {
    pub fn new(hbm_blocks: usize, dram_blocks: usize, nvme_blocks: usize) -> Self {
        TierStack {
            hbm: Arena::new(Tier::Hbm, hbm_blocks),
            dram: Arena::new(Tier::Dram, dram_blocks),
            nvme: Arena::new(Tier::Nvme, nvme_blocks),
        }
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

    /// Find which tier currently holds `id`, if any.
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
