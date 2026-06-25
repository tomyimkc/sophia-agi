// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! End-to-end tests over a real loopback TCP server.

use std::sync::Arc;
use std::time::Duration;

use kvcache::{serve, Client, ShardedCache};
use tokio::net::TcpListener;

async fn start(shards: usize, cap: usize) -> std::net::SocketAddr {
    let cache = Arc::new(ShardedCache::new(shards, cap));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, cache));
    addr
}

#[tokio::test]
async fn set_get_del_roundtrip() {
    let addr = start(4, 1024).await;
    let mut c = Client::connect(addr).await.unwrap();
    c.ping().await.unwrap();
    assert_eq!(c.get(b"absent").await.unwrap(), None);
    c.set(b"k", b"v", 0).await.unwrap();
    assert_eq!(c.get(b"k").await.unwrap(), Some(b"v".to_vec()));
    assert!(c.del(b"k").await.unwrap());
    assert!(!c.del(b"k").await.unwrap());
    assert_eq!(c.get(b"k").await.unwrap(), None);
}

#[tokio::test]
async fn ttl_expires_over_the_wire() {
    let addr = start(2, 64).await;
    let mut c = Client::connect(addr).await.unwrap();
    c.set(b"ephemeral", b"1", 30).await.unwrap();
    assert_eq!(c.get(b"ephemeral").await.unwrap(), Some(b"1".to_vec()));
    tokio::time::sleep(Duration::from_millis(45)).await;
    assert_eq!(c.get(b"ephemeral").await.unwrap(), None);
}

#[tokio::test]
async fn binary_safe_keys_and_values() {
    let addr = start(4, 1024).await;
    let mut c = Client::connect(addr).await.unwrap();
    let key = vec![0u8, 1, 2, 255, b'\n'];
    let val = vec![255u8, 0, 128, b'\t', 7];
    c.set(&key, &val, 0).await.unwrap();
    assert_eq!(c.get(&key).await.unwrap(), Some(val));
}

#[tokio::test]
async fn concurrent_clients_share_state() {
    let addr = start(8, 100_000).await;
    let mut handles = Vec::new();
    for t in 0..16 {
        handles.push(tokio::spawn(async move {
            let mut c = Client::connect(addr).await.unwrap();
            for i in 0..200 {
                let k = format!("t{t}-k{i}");
                c.set(k.as_bytes(), b"v", 0).await.unwrap();
                assert_eq!(c.get(k.as_bytes()).await.unwrap(), Some(b"v".to_vec()));
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    let mut c = Client::connect(addr).await.unwrap();
    let stats = c.stats().await.unwrap();
    assert_eq!(stats.entries, 16 * 200);
}

#[tokio::test]
async fn eviction_under_capacity_pressure() {
    // Capacity 8 over 1 shard: inserting 100 distinct keys must evict.
    let addr = start(1, 8).await;
    let mut c = Client::connect(addr).await.unwrap();
    for i in 0..100 {
        c.set(format!("k{i}").as_bytes(), b"v", 0).await.unwrap();
    }
    let stats = c.stats().await.unwrap();
    assert_eq!(stats.entries, 8);
    assert!(stats.evictions >= 92, "expected evictions, got {}", stats.evictions);
}
