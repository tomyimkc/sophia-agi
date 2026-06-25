// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! `kvcache` — a sharded async in-memory KV cache.
//!
//! Phase 1 of the Sophia distributed-storage roadmap
//! (`../../docs/storage/STORAGE_ROADMAP.md`). Single node today; the shard +
//! consistent-hash routing and the deterministic FNV-1a key hash are the seams
//! the multi-node phases (on-disk engine, Raft replication) build on.
//!
//! ```no_run
//! use std::sync::Arc;
//! use kvcache::{serve, ShardedCache, Client};
//! use tokio::net::TcpListener;
//!
//! # async fn run() -> std::io::Result<()> {
//! let cache = Arc::new(ShardedCache::new(16, 1_000_000));
//! let listener = TcpListener::bind("127.0.0.1:7070").await?;
//! tokio::spawn(serve(listener, cache));
//!
//! let mut c = Client::connect("127.0.0.1:7070").await?;
//! c.set(b"k", b"v", 0).await?;
//! assert_eq!(c.get(b"k").await?, Some(b"v".to_vec()));
//! # Ok(()) }
//! ```

pub mod cache;
pub mod client;
pub mod lru;
pub mod protocol;
pub mod server;

pub use cache::ShardedCache;
pub use client::Client;
pub use protocol::{Request, Response, StatsSnapshot};
pub use server::serve;
