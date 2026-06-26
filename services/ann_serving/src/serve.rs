// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//
//! Streaming nearest-neighbour query server — the Rust half of the Python↔Rust bridge.
//!
//! Builds an [`HnswIndex`] once from an exported vectors file (`id f0 f1 …` per line, written
//! by `tools/export_rag_index.py`), prints `READY <n> <dim>`, then answers one query per stdin
//! line. This is the architecture-track demonstration that the dense recall view can be served
//! by the Rust core while the Python side keeps query understanding, fusion, and rerank.
//!
//!   usage:  serve <vectors_file> [m] [ef_construction]
//!   query line:  `<k> <ef> f0 f1 …`      →  reply: `id:score id:score …` (best first)
//!   `QUIT` (or EOF) shuts down.

use sophia_ann::{parse_index_line, HnswIndex};
use std::io::{self, BufRead, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = match args.get(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: serve <vectors_file> [m] [ef_construction]");
            std::process::exit(2);
        }
    };
    let m: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let efc: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);

    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });

    let mut dim = 0usize;
    let mut index: Option<HnswIndex> = None;
    for line in text.lines() {
        if let Some((id, v)) = parse_index_line(line) {
            if index.is_none() {
                dim = v.len();
                index = Some(HnswIndex::new(dim, m, efc));
            }
            if v.len() == dim {
                index.as_mut().unwrap().add(id, &v);
            }
        }
    }
    let index = match index {
        Some(i) if !i.is_empty() => i,
        _ => {
            eprintln!("empty or invalid index file");
            std::process::exit(1);
        }
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    // Handshake so the client knows the build finished and can start sending queries.
    writeln!(out, "READY {} {}", index.len(), dim).ok();
    out.flush().ok();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t == "QUIT" {
            break;
        }
        let mut it = t.split_whitespace();
        let k: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(10);
        let ef: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(64);
        let q: Vec<f32> = it.filter_map(|s| s.parse().ok()).collect();
        if q.len() != dim {
            writeln!(out, "ERR expected dim {dim}, got {}", q.len()).ok();
            out.flush().ok();
            continue;
        }
        let hits = index.search(&q, k, ef);
        let parts: Vec<String> = hits.iter().map(|(id, sc)| format!("{id}:{sc:.6}")).collect();
        writeln!(out, "{}", parts.join(" ")).ok();
        out.flush().ok();
    }
}
