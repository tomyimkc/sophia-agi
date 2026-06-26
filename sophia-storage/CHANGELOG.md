# Changelog

All notable changes to the `sophia-storage` workspace are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it hits 1.0.0; until then the 0.x line may break between minor bumps.

## [Unreleased]

### Changed (publish-hygiene, no behavior change to the data path)
- **Workspace:** declared `rust-version = "1.85"` (MSRV) — edition 2024 + `let`-chains
  require rustc >= 1.85 (Feb 2025). Consumers now get a clear MSRV error instead of
  a syntax error.
- **Workspace:** removed `panic = "abort"` from the `[profile.release]`. This
  workspace publishes *library* crates; `panic = "abort"` is inherited by any
  downstream dependent, which is inappropriate for an embeddable library and breaks
  `catch_unwind` users. The perf-relevant code is exercised via the benches, which
  keep thin-LTO + opt-level 3.
- **Workspace:** added shared `categories` and `keywords` for crates.io metadata.
- **sophia-lsm:** default (std) build is now `#![forbid(unsafe_code)]`; the
  `io_uring` feature downgrades to `#![allow(unsafe_code)]` for the audited FFI
  path. Added `#![warn(rust_2018_idioms)]`.
- **sophia-kvcache:** `#![forbid(unsafe_code)]` in every build (was already unsafe-free;
  now enforced). Added `#![warn(rust_2018_idioms)]`.

### Fixed
- **sophia-kvcache:** `prefix::block_chain` now returns `Result<Vec<BlockId>,
  PrefixError>` instead of panicking on `block_len == 0`. A zero block length is a
  config error surfaced as a diagnosable error, not a process abort. The `block_len
  >= 1` invariant is enforced fail-fast at `Config::new`.
- **README:** corrected stale test count (27 → 35 std tests + 1 io_uring).

### Added
- Per-crate `README.md` for `sophia-lsm` and `sophia-kvcache` (crates.io publishes
  each crate with its own readme).
- New test `zero_block_len_is_an_error_not_a_panic` in `sophia-kvcache`.

### Known gaps (tracked, not blocking 0.1.0)
- ~48 pub items across both crates lack `///` docs; ~13 pub types (Engine, SsTable,
  Wal, IoUringIo, KvCache, FileStore, Arena, TierStack, ...) lack `#[derive(Debug)]`.
  `#![warn(missing_docs)]` and `#![warn(missing_debug_implementations)]` are
  intentionally not enabled until both close (they would fail the clippy gate in CI).
- No crates.io release yet (`cargo publish` not run); this CHANGELOG documents the
  state at first publish.

## [0.1.0] — workspace baseline (pre-publish)

Initial workspace: `sophia-lsm` (LSM engine) and `sophia-kvcache` (tiered
prefix-sharing KV cache). Tested data paths, documented seams in `docs/DESIGN.md`.
