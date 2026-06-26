// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//
//! Streaming nearest-neighbour query server — the Rust half of the Python↔Rust bridge.
//!
//! Builds (or loads) a [`ShardedHnsw`] index, prints `READY <n> <dim>`, then answers one query
//! per stdin line. The dense recall view is served by the sharded Rust core while Python keeps
//! query understanding, fusion, rerank, and provenance.
//!
//!   usage:  serve <index> [m] [ef_construction] [--shards N] [--save out.idx]
//!     <index> ending in `.idx`  → load a persisted ShardedHnsw (fast: no rebuild)
//!     <index> otherwise         → build from a text vectors file (`id f0 f1 …` per line)
//!   query line:  `<k> <ef> f0 f1 …`  →  reply: `id:score id:score …` (best first)
//!   `QUIT` (or EOF) shuts down.

use std::io::{self, BufRead, Write};

use sophia_ann::{parse_index_line, ShardedHnsw};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = match args.get(1) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: serve <index> [m] [ef_construction] [--shards N] [--save out.idx]");
            std::process::exit(2);
        }
    };

    // Positional [m] [ef_construction] (back-compat with the existing Python client) + flags.
    let positionals: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
    let m: usize = positionals.first().and_then(|s| s.parse().ok()).unwrap_or(16);
    let efc: usize = positionals.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let shards: usize = flag_value(&args, "--shards").and_then(|s| s.parse().ok()).unwrap_or(1);
    let save: Option<String> = flag_value(&args, "--save");

    let index = if path.ends_with(".idx") {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| fail(&format!("read {path}: {e}")));
        ShardedHnsw::from_bytes(&bytes).unwrap_or_else(|| fail("corrupt or incompatible .idx"))
    } else {
        build_from_text(&path, shards, m, efc)
    };
    if index.is_empty() {
        fail("empty index");
    }

    if let Some(out) = &save {
        if let Err(e) = std::fs::write(out, index.to_bytes()) {
            eprintln!("warning: could not write {out}: {e}");
        }
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    // Handshake: the client waits for READY before sending queries.
    writeln!(out, "READY {} {}", index.len(), index.dim()).ok();
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
        if q.len() != index.dim() {
            writeln!(out, "ERR expected dim {}, got {}", index.dim(), q.len()).ok();
            out.flush().ok();
            continue;
        }
        let hits = index.search(&q, k, ef);
        let parts: Vec<String> = hits.iter().map(|(id, sc)| format!("{id}:{sc:.6}")).collect();
        writeln!(out, "{}", parts.join(" ")).ok();
        out.flush().ok();
    }
}

fn build_from_text(path: &str, shards: usize, m: usize, efc: usize) -> ShardedHnsw {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| fail(&format!("read {path}: {e}")));
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
    index.unwrap_or_else(|| fail("empty or invalid index file"))
}

fn flag_value(args: &[String], name: &str) -> Option<String> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            return it.next().cloned();
        }
        if let Some(v) = a.strip_prefix(&format!("{name}=")) {
            return Some(v.to_string());
        }
    }
    None
}

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}
