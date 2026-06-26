// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! `diskstore` — a bitcask-style durable KV engine with a batched read path.
//!
//! Phase 2 of the Sophia distributed-storage roadmap
//! (`../../docs/storage/STORAGE_ROADMAP.md`): turns the in-memory cache tier
//! into a crash-consistent on-disk engine, and demonstrates first-hand
//! `io_uring` use on the batched-read path (`reader::UringReader`, feature
//! `io_uring`). The default build uses a portable `pread` backend.
//!
//! ```no_run
//! use diskstore::{Bitcask, StdReader};
//! # fn run() -> std::io::Result<()> {
//! let mut db = Bitcask::open("/tmp/sophia-diskstore", /* sync_on_put */ true)?;
//! db.put(b"k", b"v")?;
//! assert_eq!(db.get(b"k")?, Some(b"v".to_vec()));
//! let got = db.multi_get(&StdReader, &[b"k", b"absent"])?;
//! assert_eq!(got, vec![Some(b"v".to_vec()), None]);
//! # Ok(()) }
//! ```

mod crc;
mod record;

pub mod engine;
pub mod reader;

pub use engine::Bitcask;
pub use reader::{BatchReader, ReadOp, StdReader};

#[cfg(feature = "io_uring")]
pub use reader::UringReader;
