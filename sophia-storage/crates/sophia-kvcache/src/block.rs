// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Paged KV blocks — the unit of allocation, sharing, and eviction.
//!
//! A block holds the key/value attention tensors for a fixed number of tokens
//! (`block_len`), exactly like PagedAttention's pages. Blocks are immutable once
//! sealed and identified by a content hash of *(prefix-hash, token ids)* so two
//! requests that share a prompt prefix resolve to the *same* block id and we
//! store the KV once.
//!
//! In this skeleton a block's payload is an opaque `Vec<u8>` standing in for the
//! device tensor; the real engine would hold a handle into an HBM/DRAM/NVMe
//! arena. The id math, ref-counting, and sharing logic are the parts that have
//! to be right, and they are independent of where the bytes physically live.

use std::fmt;

/// Content-addressed block identifier (64-bit FNV-1a over prefix + tokens).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u64);

impl fmt::Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "blk:{:016x}", self.0)
    }
}

impl BlockId {
    /// Derive the id of the block that extends `prefix` with `tokens`.
    /// Chaining the parent's hash in makes ids position-dependent, so the same
    /// token run after a different prefix is correctly a different block.
    pub fn derive(prefix: BlockId, tokens: &[u32]) -> BlockId {
        let mut h = prefix.0 ^ 0xcbf2_9ce4_8422_2325; // mix in parent
        for &t in tokens {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        BlockId(h)
    }

    /// The conventional root (empty prefix).
    pub const ROOT: BlockId = BlockId(0xcbf2_9ce4_8422_2325);
}

/// Which physical tier a resident block currently lives on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// GPU high-bandwidth memory — hottest, smallest, lowest latency.
    Hbm = 0,
    /// Host DRAM — warm spill, reachable over PCIe/NVLink.
    Dram = 1,
    /// NVMe SSD — cold, large, persists across requests.
    Nvme = 2,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Hbm => "HBM",
            Tier::Dram => "DRAM",
            Tier::Nvme => "NVMe",
        }
    }
}

/// A sealed KV block.
#[derive(Clone)]
pub struct Block {
    pub id: BlockId,
    pub token_count: u32,
    /// Opaque KV payload (device tensor stand-in).
    pub payload: Vec<u8>,
}

impl Block {
    pub fn new(id: BlockId, token_count: u32, payload: Vec<u8>) -> Self {
        Block { id, token_count, payload }
    }

    pub fn bytes(&self) -> usize {
        self.payload.len()
    }
}

impl fmt::Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Block")
            .field("id", &self.id)
            .field("tokens", &self.token_count)
            .field("bytes", &self.payload.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_prefix_same_id() {
        let a = BlockId::derive(BlockId::ROOT, &[1, 2, 3]);
        let b = BlockId::derive(BlockId::ROOT, &[1, 2, 3]);
        assert_eq!(a, b, "identical prefix+tokens must hash equal (sharing)");
    }

    #[test]
    fn different_prefix_different_id() {
        let p1 = BlockId::derive(BlockId::ROOT, &[1]);
        let p2 = BlockId::derive(BlockId::ROOT, &[2]);
        assert_ne!(
            BlockId::derive(p1, &[9, 9]),
            BlockId::derive(p2, &[9, 9]),
            "same tokens under different prefixes must differ (position-dependent)"
        );
    }
}
