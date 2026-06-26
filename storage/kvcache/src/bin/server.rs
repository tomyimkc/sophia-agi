// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! kvcache-server — run the cache over TCP.
//!
//! Usage:
//!   kvcache-server [--addr 127.0.0.1:7070] [--shards 16] [--capacity 1000000]
//!
//! Env overrides (lower precedence than flags): KVCACHE_ADDR, KVCACHE_SHARDS,
//! KVCACHE_CAPACITY.

use std::sync::Arc;

use kvcache::{serve, ShardedCache};
use tokio::net::TcpListener;

struct Config {
    addr: String,
    shards: usize,
    capacity: usize,
}

fn parse_config() -> Config {
    let mut cfg = Config {
        addr: std::env::var("KVCACHE_ADDR").unwrap_or_else(|_| "127.0.0.1:7070".into()),
        shards: env_usize("KVCACHE_SHARDS", 16),
        capacity: env_usize("KVCACHE_CAPACITY", 1_000_000),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--addr" => cfg.addr = args.next().expect("--addr needs a value"),
            "--shards" => cfg.shards = args.next().and_then(|v| v.parse().ok()).expect("--shards int"),
            "--capacity" => cfg.capacity = args.next().and_then(|v| v.parse().ok()).expect("--capacity int"),
            "-h" | "--help" => {
                println!("usage: kvcache-server [--addr A] [--shards N] [--capacity N]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    cfg
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let cfg = parse_config();
    let cache = Arc::new(ShardedCache::new(cfg.shards, cfg.capacity));
    let listener = TcpListener::bind(&cfg.addr).await?;
    eprintln!(
        "kvcache-server listening on {} ({} shards, {} entries capacity)",
        listener.local_addr()?,
        cfg.shards,
        cfg.capacity
    );
    serve(listener, cache).await
}
