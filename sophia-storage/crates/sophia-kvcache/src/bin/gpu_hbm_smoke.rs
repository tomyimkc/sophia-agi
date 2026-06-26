// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! GPU HBM tier smoke test + bandwidth probe. Runs on a real CUDA GPU (the
//! RunPod box); built only with `--features cuda`.
//!
//! It drives the production-shaped path: paged KV blocks are written into GPU
//! **HBM** (host→device cudaMemcpy), read back (device→host), and verified
//! byte-for-byte — then it reports the achieved host↔device bandwidth, the
//! number the disaggregated KVCache transfer path lives and dies by.
//!
//!   cargo run --release -p sophia-kvcache --features cuda --bin gpu_hbm_smoke

use std::time::Instant;

use sophia_kvcache::block::{Block, BlockId};
use sophia_kvcache::store::{BlockStore, CudaHbmStore};

fn main() {
    let n_blocks: usize = std::env::var("BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let block_bytes: usize =
        std::env::var("BLOCK_BYTES").ok().and_then(|s| s.parse().ok()).unwrap_or(256 * 1024);

    println!("== sophia-kvcache GPU HBM smoke ==");
    println!("blocks={n_blocks}  block_bytes={block_bytes}  total={} MiB", (n_blocks * block_bytes) >> 20);

    let mut hbm = match CudaHbmStore::open(0) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("FAILED to open CUDA device 0: {e}");
            std::process::exit(1);
        }
    };

    // Build deterministic payloads so we can verify the round trip.
    let payloads: Vec<Vec<u8>> = (0..n_blocks)
        .map(|i| {
            let seed = (i as u8).wrapping_mul(31).wrapping_add(7);
            (0..block_bytes).map(|j| seed.wrapping_add(j as u8)).collect()
        })
        .collect();

    // --- host -> HBM (cudaMemcpy H2D) ---
    let t0 = Instant::now();
    for (i, p) in payloads.iter().enumerate() {
        let id = BlockId(i as u64);
        hbm.put(Block::new(id, 16, p.clone())).expect("H2D put");
    }
    let h2d = t0.elapsed();

    // --- HBM -> host (cudaMemcpy D2H) + verify ---
    let t1 = Instant::now();
    let mut verified = 0usize;
    for (i, p) in payloads.iter().enumerate() {
        let id = BlockId(i as u64);
        let back = hbm.get(id).expect("D2H get").expect("block resident in HBM");
        assert_eq!(&back.payload, p, "HBM round-trip corrupted block {i}");
        verified += 1;
    }
    let d2h = t1.elapsed();

    let total_bytes = (n_blocks * block_bytes) as f64;
    let gib = |bytes: f64, secs: f64| (bytes / (1024.0 * 1024.0 * 1024.0)) / secs;
    let (bytes_in, bytes_out) = (hbm.bytes_in(), hbm.bytes_out());

    println!("resident blocks in HBM: {}", hbm.len());
    println!("verified round-trips:   {verified}/{n_blocks}");
    println!("H2D: {:.2} GiB/s   ({:.1} ms)", gib(total_bytes, h2d.as_secs_f64()), h2d.as_secs_f64() * 1e3);
    println!("D2H: {:.2} GiB/s   ({:.1} ms)", gib(total_bytes, d2h.as_secs_f64()), d2h.as_secs_f64() * 1e3);
    println!("accounting: bytes_in={bytes_in} bytes_out={bytes_out}");

    // Evict everything (frees device memory) and confirm the tier drains.
    for i in 0..n_blocks {
        hbm.take(BlockId(i as u64)).expect("take").expect("present");
    }
    assert_eq!(hbm.len(), 0, "HBM tier did not drain");
    println!("RESULT: PASS — real GPU HBM tier round-trips and drains");
}
