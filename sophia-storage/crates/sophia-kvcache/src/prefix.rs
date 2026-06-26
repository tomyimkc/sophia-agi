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
use std::fmt;

use crate::block::BlockId;

/// Errors returned by prefix-index operations. Returned (not panicked) so a bad
/// config is a diagnosable caller error, not a process abort — important for an
/// embeddable library and especially because the workspace no longer sets
/// `panic = "abort"` (which would have made any panic unrecoverable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixError {
    /// `block_len` was zero, which would divide the token stream into empty
    /// windows. This is a misconfiguration (`Config::block_len` must be >= 1).
    ZeroBlockLen,
}

impl fmt::Display for PrefixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrefixError::ZeroBlockLen => write!(f, "block_len must be >= 1 (got 0)"),
        }
    }
}

impl std::error::Error for PrefixError {}

/// Split a token stream into the chain of block ids it would occupy. Each id
/// depends on all tokens before it, so equal prefixes yield equal id prefixes.
///
/// Returns `Err(PrefixError::ZeroBlockLen)` if `block_len == 0` rather than
/// panicking — a zero block length is a `Config` error the caller can report.
pub fn block_chain(tokens: &[u32], block_len: usize) -> Result<Vec<BlockId>, PrefixError> {
    if block_len == 0 {
        return Err(PrefixError::ZeroBlockLen);
    }
    let mut chain = Vec::with_capacity(tokens.len() / block_len + 1);
    let mut parent = BlockId::ROOT;
    for window in tokens.chunks(block_len) {
        let id = BlockId::derive(parent, window);
        chain.push(id);
        parent = id;
    }
    Ok(chain)
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
        let a = block_chain(&[1, 2, 3, 4, 5], 2).unwrap();
        let b = block_chain(&[1, 2, 3, 4, 5], 2).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn divergent_suffix_splits() {
        let a = block_chain(&[1, 2, 3, 4], 2).unwrap(); // blocks: [1,2],[3,4]
        let b = block_chain(&[1, 2, 9, 9], 2).unwrap(); // blocks: [1,2],[9,9]
        assert_eq!(a[0], b[0], "shared first block");
        assert_ne!(a[1], b[1], "divergent second block");
    }

    #[test]
    fn prefix_len_is_contiguous() {
        let chain = block_chain(&[1, 2, 3, 4, 5, 6], 2).unwrap();
        let resident: HashSet<BlockId> = [chain[0], chain[2]].into_iter().collect();
        // block[1] is missing, so sharing stops at 1 even though block[2] exists.
        assert_eq!(shared_prefix_len(&chain, &resident), 1);
    }

    #[test]
    fn zero_block_len_is_an_error_not_a_panic() {
        // A zero block_len is a config error surfaced as Result, not a process abort.
        assert_eq!(block_chain(&[1, 2, 3], 0), Err(PrefixError::ZeroBlockLen));
    }
}
