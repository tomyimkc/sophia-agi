# sophia-lsm

[![crates.io](https://img.shields.io/crates/v/sophia-lsm.svg)](https://crates.io/crates/sophia-lsm)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)

A small, honest **log-structured storage engine**: WAL → memtable → SSTable →
compaction, with pluggable I/O and CRC-framed crash recovery. It replaces
append-only JSONL stores with a real engine that keeps the same idempotent,
durable, hand-auditable semantics while paying down read-amplification and the
single-file-rewrite costs of JSONL.

It is deliberately a **skeleton**: the data path is correct and tested, and the
performance levers a storage engineer reaches for (io_uring backend, bloom
filters, leveled compaction, block cache) are present as documented seams rather
than half-built features. See [`docs/DESIGN.md`](../../docs/DESIGN.md).

## Status

**0.1.0 — alpha.** Tested data path, honest about its seams. Not a drop-in
replacement for RocksDB; it is the durable engine Sophia's trust layer is built to
sit on top of, published standalone because nothing in the Python codebase depends
on it.

## Safety

- **Default (std) build: `#![forbid(unsafe_code)]` — zero unsafe, verifiable with
  `cargo build`.**
- The `io_uring` feature downgrades that to `#![allow(unsafe_code)]` because the
  `io-uring` FFI legitimately needs it; that path is isolated in `src/io.rs`
  behind `#[cfg(feature = "io_uring")]`, audited, and SAFETY-commented.

## Example

```rust
use sophia_lsm::{Engine, Options};

let dir = std::env::temp_dir().join(format!("sophia-lsm-demo-{}", std::process::id()));
let mut db = Engine::open(Options::new(&dir)).unwrap();
db.put(b"claim:42", b"accepted").unwrap();
assert_eq!(db.get(b"claim:42").unwrap().as_deref(), Some(&b"accepted"[..]));
db.delete(b"claim:42").unwrap();
assert_eq!(db.get(b"claim:42").unwrap(), None);
std::fs::remove_dir_all(&dir).ok();
```

## Build

```bash
cargo test -p sophia-lsm                         # std backend (default)
cargo test -p sophia-lsm --features io_uring     # + the real io_uring backend (Linux 5.1+)
cargo bench -p sophia-lsm                        # put/get latency + group-commit scaling
```

**MSRV:** Rust 1.85+ (edition 2024, `let`-chains).

## License

Apache-2.0. Part of the [sophia-agi](https://github.com/tomyimkc/sophia-agi) project.
