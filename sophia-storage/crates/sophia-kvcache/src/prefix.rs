// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Prefix index: map a token sequence to the longest run of already-cached
//! blocks, so a new request reuses the shared prompt prefix instead of
//! recomputing it.
//!
//! This is the data structure that makes best-of-N and council deliberation
//! cheap: N samples over the same prompt share one chain of prefix blocks, and
//! we only ever materialize the divergent suffix.
//!
//! Tokenization into blocks is fixed-length (`block_len` tokens per block),
//! matching the paged layout in `block.rs`. The chain of block ids is computed
//! by folding [`BlockId::derive`] along the sequence.

use std::collections::HashSet;

use crate::block::BlockId;

/// Split a token stream into the chain of block ids it would occupy. Each id
/// depends on all tokens before it, so equal prefixes yield equal id prefixes.
pub fn block_chain(tokens: &[u32], block_len: usize) -> Vec<BlockId> {
    assert!(block_len > 0, "block_len must be positive");
    let mut chain = Vec::with_capacity(tokens.len() / block_len + 1);
    let mut parent = BlockId::ROOT;
    for window in tokens.chunks(block_len) {
        let id = BlockId::derive(parent, window);
        chain.push(id);
        parent = id;
    }
    chain
}

/// Given the block chain for a request and the set of currently-resident block
/// ids, return how many leading blocks are cache hits (the reusable prefix).
pub fn shared_prefix_len(chain: &[BlockId], resident: &HashSet<BlockId>) -> usize {
    let mut n = 0;
    for id in chain {
        if resident.contains(id) {
            n += 1;
        } else {
            break; // prefix sharing is contiguous from the front
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_prompts_share_full_chain() {
        let a = block_chain(&[1, 2, 3, 4, 5], 2);
        let b = block_chain(&[1, 2, 3, 4, 5], 2);
        assert_eq!(a, b);
    }

    #[test]
    fn divergent_suffix_splits() {
        let a = block_chain(&[1, 2, 3, 4], 2); // blocks: [1,2],[3,4]
        let b = block_chain(&[1, 2, 9, 9], 2); // blocks: [1,2],[9,9]
        assert_eq!(a[0], b[0], "shared first block");
        assert_ne!(a[1], b[1], "divergent second block");
    }

    #[test]
    fn prefix_len_is_contiguous() {
        let chain = block_chain(&[1, 2, 3, 4, 5, 6], 2);
        let resident: HashSet<BlockId> = [chain[0], chain[2]].into_iter().collect();
        // block[1] is missing, so sharing stops at 1 even though block[2] exists.
        assert_eq!(shared_prefix_len(&chain, &resident), 1);
    }
}
