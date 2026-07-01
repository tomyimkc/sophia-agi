// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Byte-exact mirror of the content-shingle near-duplicate scan in
//! `tools/assert_decontam.py`:
//!
//! ```python
//! def _shingles(text, k):
//!     toks = normalize(text).split()
//!     if len(toks) < k:
//!         return {" ".join(toks)} if toks else set()
//!     return {" ".join(toks[i:i+k]) for i in range(len(toks)-k+1)}
//!
//! def _jaccard(a, b):
//!     if not a or not b: return 0.0
//!     return len(a & b) / len(a | b)
//! ```
//!
//! The Python tool caps the eval side at `--max-eval-shingle 4000` "for perf".
//! That cap silently bounds coverage — exactly the failure mode the repo's own
//! measurement thesis condemns. This DFA-tokenized implementation is fast enough
//! that the accelerated path removes the cap and scans the full eval surface.

use std::collections::HashSet;

use crate::normalize::normalize;

/// Word `k`-shingles over the normalized text. Mirrors `_shingles`: when there
/// are fewer than `k` tokens, the whole (joined) token string is the single
/// shingle; an empty text yields the empty set.
pub fn shingles(text: &str, k: usize) -> HashSet<String> {
    let norm = normalize(text);
    let toks: Vec<&str> = norm.split(' ').filter(|t| !t.is_empty()).collect();
    let mut out = HashSet::new();
    if toks.is_empty() {
        return out;
    }
    if toks.len() < k {
        out.insert(toks.join(" "));
        return out;
    }
    for i in 0..=(toks.len() - k) {
        out.insert(toks[i..i + k].join(" "));
    }
    out
}

/// Jaccard similarity of two shingle sets. Mirrors `_jaccard` (empty -> 0.0).
pub fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    let uni = a.len() + b.len() - inter;
    inter as f64 / uni as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_single_shingle() {
        let s = shingles("a b", 5);
        assert_eq!(s.len(), 1);
        assert!(s.contains("a b"));
    }

    #[test]
    fn empty_text_empty_set() {
        assert!(shingles("   ", 5).is_empty());
    }

    #[test]
    fn windowed_shingles() {
        let s = shingles("the quick brown fox jumps", 3);
        assert_eq!(s.len(), 3);
        assert!(s.contains("the quick brown"));
        assert!(s.contains("brown fox jumps"));
    }

    #[test]
    fn jaccard_identity_and_disjoint() {
        let a = shingles("the quick brown fox", 2);
        let b = shingles("the quick brown fox", 2);
        assert!((jaccard(&a, &b) - 1.0).abs() < 1e-9);
        let c = shingles("zero one two three", 2);
        assert_eq!(jaccard(&a, &c), 0.0);
    }
}
