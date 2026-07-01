// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Byte-exact mirror of `provenance_bench.dataset_guard.normalize`:
//!
//! ```python
//! def normalize(text: str) -> str:
//!     return re.sub(r"\s+", " ", str(text or "").strip().lower())
//! ```
//!
//! Parity is load-bearing: the decontamination assertion compares normalized
//! prompts, so any divergence here is a contamination-detection divergence.

/// Lowercase, collapse every run of ASCII/Unicode whitespace to a single space,
/// and trim. Matches Python's `str.strip()` / `str.lower()` / `re.sub(r"\s+",...)`.
///
/// Python's `\s` (with `re.UNICODE`, the str default) and `str.strip()` both act
/// on the Unicode whitespace set; Rust's `char::is_whitespace` is the same set,
/// so `split_whitespace().join(" ")` reproduces `strip()` + `\s+ -> " "` exactly.
pub fn normalize(text: &str) -> String {
    let lowered = text.to_lowercase();
    // split_whitespace() drops leading/trailing whitespace (== strip) and treats
    // any run of whitespace as one separator (== collapse), then we rejoin with a
    // single ASCII space.
    let mut out = String::with_capacity(lowered.len());
    for (i, tok) in lowered.split_whitespace().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(tok);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::normalize;

    #[test]
    fn collapses_and_trims() {
        assert_eq!(normalize("  Hello   World \t\n"), "hello world");
    }

    #[test]
    fn empty_and_whitespace_only() {
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("   \t\n "), "");
    }

    #[test]
    fn lowercases_unicode() {
        assert_eq!(normalize("GÖDEL  Machine"), "gödel machine");
    }
}
