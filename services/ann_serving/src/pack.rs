// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//
//! Build a persisted sharded index once, so `serve` can load it instantly instead of rebuilding.
//!
//!   usage:  pack <text_vectors_in> <out.idx> [shards] [m] [ef_construction]
//!
//! Reads `id f0 f1 …` lines (the format `tools/export_rag_index.py` writes), builds a
//! [`ShardedHnsw`](sophia_ann::ShardedHnsw), and writes the portable `.idx` blob. Graph
//! construction is the expensive step; packing pays it once.

use std::time::Instant;

use sophia_ann::{parse_index_line, ShardedHnsw};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: pack <text_vectors_in> <out.idx> [shards] [m] [ef_construction]");
        std::process::exit(2);
    }
    let input = &args[1];
    let out = &args[2];
    let shards: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let m: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let efc: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(200);

    let text = std::fs::read_to_string(input).unwrap_or_else(|e| {
        eprintln!("read {input}: {e}");
        std::process::exit(1);
    });

    let t = Instant::now();
    let mut dim = 0usize;
    let mut index: Option<ShardedHnsw> = None;
    for line in text.lines() {
        if let Some((id, v)) = parse_index_line(line) {
            if index.is_none() {
                dim = v.len();
                index = Some(ShardedHnsw::new(shards, dim, m, efc));
            }
            if v.len() == dim {
                index.as_mut().unwrap().add(id, &v);
            }
        }
    }
    let index = index.unwrap_or_else(|| {
        eprintln!("empty or invalid input");
        std::process::exit(1);
    });
    let build_ms = t.elapsed().as_secs_f64() * 1e3;

    let blob = index.to_bytes();
    std::fs::write(out, &blob).unwrap_or_else(|e| {
        eprintln!("write {out}: {e}");
        std::process::exit(1);
    });
    println!(
        "packed {} vectors, dim {}, {} shard(s) -> {} ({:.1} KB) in {:.0} ms",
        index.len(), dim, index.num_shards(), out, blob.len() as f64 / 1024.0, build_ms
    );
}
