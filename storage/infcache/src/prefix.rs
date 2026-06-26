// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Prefix-block keying — the idea behind LLM "context caching".
//!
//! During prefill, the attention KV for token *i* depends only on tokens
//! `0..=i`. So two requests that share a prompt prefix produce **identical** KV
//! for that prefix and can reuse it instead of recomputing. We chunk the token
//! stream into fixed-size blocks and give block *b* a key that hashes *every
//! token up to the end of block b*. Identical prefixes therefore yield identical
//! block keys up to the point they diverge — and the first differing token
//! changes that block's key and every key after it. (Same principle as vLLM
//! prefix caching / SGLang RadixAttention / DeepSeek context caching.)

/// FNV-1a over a token id, folded into a running prefix hash. Deterministic
/// across processes so cached blocks are shareable.
fn fold(mut h: u64, token: u32) -> u64 {
    for b in token.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Per-block keys for `tokens`, chunked by `block_tokens`. Block `b`'s key
/// covers all tokens `0..end_of_block_b`, so the keys form a prefix chain. The
/// last block may be partial. Returns 8-byte big-endian keys.
pub fn block_keys(tokens: &[u32], block_tokens: usize) -> Vec<[u8; 8]> {
    assert!(block_tokens > 0, "block_tokens must be > 0");
    let mut keys = Vec::with_capacity(tokens.len().div_ceil(block_tokens));
    let mut h = 0xcbf2_9ce4_8422_2325u64; // FNV offset basis
    for (i, &tok) in tokens.iter().enumerate() {
        h = fold(h, tok);
        if (i + 1).is_multiple_of(block_tokens) {
            keys.push(h.to_be_bytes());
        }
    }
    // Trailing partial block.
    if !tokens.len().is_multiple_of(block_tokens) {
        keys.push(h.to_be_bytes());
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::block_keys;

    #[test]
    fn shared_prefix_shares_keys_until_divergence() {
        let a: Vec<u32> = (0..100).collect();
        let mut b = a.clone();
        b[55] = 9999; // diverge inside block 3 (block size 16)

        let ka = block_keys(&a, 16);
        let kb = block_keys(&b, 16);
        assert_eq!(ka.len(), kb.len());
        // Blocks 0,1,2 cover tokens 0..48 — identical, so keys match.
        assert_eq!(ka[0], kb[0]);
        assert_eq!(ka[1], kb[1]);
        assert_eq!(ka[2], kb[2]);
        // Block 3 covers tokens 48..64 which includes index 55 — diverges.
        assert_ne!(ka[3], kb[3]);
        // And everything after stays diverged (prefix chain).
        assert_ne!(ka[4], kb[4]);
    }

    #[test]
    fn block_count_and_partial_tail() {
        assert_eq!(block_keys(&[1, 2, 3], 16).len(), 1); // one partial block
        assert_eq!(block_keys(&(0..32).collect::<Vec<_>>(), 16).len(), 2); // exactly two
        assert_eq!(block_keys(&(0..33).collect::<Vec<_>>(), 16).len(), 3); // two + partial
    }

    #[test]
    fn identical_sequences_are_fully_equal() {
        let s: Vec<u32> = (0..64).collect();
        assert_eq!(block_keys(&s, 16), block_keys(&s, 16));
    }
}
