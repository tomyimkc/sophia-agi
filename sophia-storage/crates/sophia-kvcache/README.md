# sophia-kvcache

[![crates.io](https://img.shields.io/crates/v/sophia-kvcache.svg)](https://crates.io/crates/sophia-kvcache)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)

A **disaggregated, prefix-sharing KV-cache for LLM inference** — paged blocks
tiered across HBM / DRAM / NVMe with reference-counted prefix reuse. It targets
the canonical case for KV reuse: best-of-N sampling and multi-agent council
deliberation issuing many requests over the *same* long prompt prefix. N samples
share one chain of prefix blocks; only the divergent suffix is materialized.

## Status

**0.1.0 — alpha.** Tested tiering + prefix sharing + LRU pinning + a real
disk-backed NVMe tier. Honest about its seams (see
[`docs/DESIGN.md`](../../docs/DESIGN.md)).

## Safety

**`#![forbid(unsafe_code)]` in every build — zero unsafe, no FFI, no raw
pointers.** Verifiable by inspecting `src/lib.rs`.

## Example

```rust
use sophia_kvcache::{KvCache, Config};

let cfg = Config::new(16, 1024, 4096, 0); // block_len=16 tokens; HBM/DRAM/NVMe block counts
let mut cache = KvCache::new(cfg).unwrap();
let prompt: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8];
// First admission computes and stores; second admission hits the shared prefix.
let r1 = cache.admit(prompt, |_id| vec![0u8; 64]).unwrap();
let r2 = cache.admit(prompt, |_id| vec![0u8; 64]).unwrap();
assert!(r2.prefix_hits >= r1.prefix_hits);
```

## Build

```bash
cargo test -p sophia-kvcache      # std backend, no deps
cargo bench -p sophia-kvcache     # prefix hit-ratio + prefill avoided + NVMe bytes
```

**MSRV:** Rust 1.85+ (edition 2024, `let`-chains). Zero external dependencies.

## License

Apache-2.0. Part of the [sophia-agi](https://github.com/tomyimkc/sophia-agi) project.
