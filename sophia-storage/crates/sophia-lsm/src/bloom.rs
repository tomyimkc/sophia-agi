// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Bloom filter per SSTable.
//!
//! A point lookup that misses still has to be *ruled out* of every table. The
//! sparse index narrows the scan, but the read still touches the file. A bloom
//! filter answers "this key is **definitely not** here" without any I/O, so a
//! `get` skips tables that can't hold the key entirely — the standard fix for
//! LSM read amplification on absent/old keys.
//!
//! Implementation: a bit array with `k` probes derived by double hashing from a
//! single 64-bit FNV-1a (Kirsch–Mitzenmacher), sized to a target ~1% false
//! positive rate. No false negatives, so skipping on a negative is always safe.

/// Bits per key for ~1% false positives (≈ -ln(0.01)/ln(2)^2 ≈ 9.6).
const BITS_PER_KEY: usize = 10;

#[derive(Debug, Clone)]
pub struct Bloom {
    bits: Vec<u8>,
    k: u32,
}

impl Bloom {
    /// Build a filter sized for `n` keys (caller adds them via [`Bloom::add`]).
    pub fn with_capacity(n: usize) -> Self {
        let nbits = (n * BITS_PER_KEY).max(64);
        let nbytes = nbits.div_ceil(8);
        // Optimal k = bits_per_key * ln2 ≈ 0.69 * 10 ≈ 7.
        let k = ((BITS_PER_KEY as f64) * std::f64::consts::LN_2).round().max(1.0) as u32;
        Bloom { bits: vec![0u8; nbytes], k }
    }

    /// Build directly from a set of keys.
    pub fn build<'a, I: IntoIterator<Item = &'a [u8]>>(keys: I, n: usize) -> Self {
        let mut b = Bloom::with_capacity(n);
        for key in keys {
            b.add(key);
        }
        b
    }

    fn nbits(&self) -> u64 {
        (self.bits.len() as u64) * 8
    }

    pub fn add(&mut self, key: &[u8]) {
        let (h1, h2) = hashes(key);
        let nbits = self.nbits();
        for i in 0..self.k {
            let bit = h1.wrapping_add((i as u64).wrapping_mul(h2)) % nbits;
            self.bits[(bit / 8) as usize] |= 1 << (bit % 8);
        }
    }

    /// `true` = key *might* be present (scan the table); `false` = definitely
    /// absent (skip it). Never returns a false negative.
    pub fn maybe_contains(&self, key: &[u8]) -> bool {
        let (h1, h2) = hashes(key);
        let nbits = self.nbits();
        for i in 0..self.k {
            let bit = h1.wrapping_add((i as u64).wrapping_mul(h2)) % nbits;
            if self.bits[(bit / 8) as usize] & (1 << (bit % 8)) == 0 {
                return false;
            }
        }
        true
    }

    /// Serialize: `[ k: u32-le ][ bits ]`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.bits.len());
        out.extend_from_slice(&self.k.to_le_bytes());
        out.extend_from_slice(&self.bits);
        out
    }

    pub fn decode(buf: &[u8]) -> Self {
        let k = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        Bloom { bits: buf[4..].to_vec(), k }
    }
}

/// Two independent 64-bit hashes for double hashing. FNV-1a with two seeds.
fn hashes(key: &[u8]) -> (u64, u64) {
    let mut h1: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h2: u64 = 0x84222325cbf29ce4u64 ^ 0xff51_afd7_ed55_8ccd;
    for &b in key {
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(0x0000_0100_0000_01B3);
        h2 = h2.wrapping_add(b as u64);
        h2 = h2.wrapping_mul(0x0000_0100_0000_01B3);
    }
    // Ensure h2 is odd so the probe sequence covers distinct bits.
    (h1, h2 | 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_false_negatives() {
        let keys: Vec<Vec<u8>> = (0..1000u32).map(|i| format!("key:{i}").into_bytes()).collect();
        let bloom = Bloom::build(keys.iter().map(|k| k.as_slice()), keys.len());
        for k in &keys {
            assert!(bloom.maybe_contains(k), "false negative for {k:?}");
        }
    }

    #[test]
    fn false_positive_rate_is_low() {
        let keys: Vec<Vec<u8>> = (0..1000u32).map(|i| format!("key:{i}").into_bytes()).collect();
        let bloom = Bloom::build(keys.iter().map(|k| k.as_slice()), keys.len());
        let mut fp = 0;
        for i in 1000..11000u32 {
            if bloom.maybe_contains(format!("key:{i}").as_bytes()) {
                fp += 1;
            }
        }
        // ~1% expected; allow generous headroom for hash variance.
        assert!(fp < 300, "false positive rate too high: {fp}/10000");
    }

    #[test]
    fn round_trips() {
        let keys: Vec<Vec<u8>> = (0..100u32).map(|i| format!("k{i}").into_bytes()).collect();
        let bloom = Bloom::build(keys.iter().map(|k| k.as_slice()), keys.len());
        let back = Bloom::decode(&bloom.encode());
        for k in &keys {
            assert!(back.maybe_contains(k));
        }
    }
}
