// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! sophia-lex — deterministic single-DFA scanners for the Sophia measurement gate.
//!
//! Three logos-backed components, each an OPTIONAL accelerator whose verdicts
//! agree with a pure-Python reference oracle (or, for SCL, are net-new):
//!   * [`overclaim`] — the no-overclaim gate (mirrors `tools/lint_claims.py`).
//!   * [`shingle`] / [`normalize`] — decontamination near-dup scan
//!     (mirrors `tools/assert_decontam.py`, without its perf coverage cap).
//!   * [`scl`] — the Sophia Claim Language: deterministic claim -> triple.

pub mod normalize;
pub mod overclaim;
pub mod scl;
pub mod shingle;
