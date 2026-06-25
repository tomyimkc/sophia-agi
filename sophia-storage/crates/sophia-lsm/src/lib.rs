// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! # sophia-lsm
//!
//! A small, honest log-structured storage engine: **WAL → memtable → SSTable →
//! compaction**, with pluggable I/O and crash-checked framing. It exists to
//! replace the append-only JSONL stores in Sophia's Python trust layer
//! (`sophia_contract/stores.py`, `queue.py`) with a real engine that keeps the
//! *same* semantics — idempotent, durable, hand-auditable — while paying down
//! the read-amplification and single-file-rewrite costs of JSONL.
//!
//! It is deliberately a **skeleton**: the data path is correct and tested, and
//! the performance levers a storage engineer reaches for (io_uring backend,
//! bloom filters, leveled compaction, block cache) are present as documented
//! seams rather than half-built features. See `docs/DESIGN.md`.
//!
//! ## Example
//! ```
//! use sophia_lsm::{Engine, Options};
//! let dir = std::env::temp_dir().join(format!("sophia-lsm-doctest-{}", std::process::id()));
//! let mut db = Engine::open(Options::new(&dir)).unwrap();
//! db.put(b"claim:42", b"accepted").unwrap();
//! assert_eq!(db.get(b"claim:42").unwrap().as_deref(), Some(&b"accepted"[..]));
//! db.delete(b"claim:42").unwrap();
//! assert_eq!(db.get(b"claim:42").unwrap(), None);
//! std::fs::remove_dir_all(&dir).ok();
//! ```

pub mod compaction;
pub mod io;
pub mod memtable;
pub mod record;
pub mod sstable;
pub mod wal;

use std::io as stdio;
use std::path::{Path, PathBuf};

use crate::io::{IoBackend, StdIo};
use crate::memtable::MemTable;
use crate::record::Record;
use crate::sstable::SsTable;
use crate::wal::Wal;

/// Engine tuning. `flush_threshold_bytes` bounds the memtable before it spills
/// to an SSTable; `compaction_trigger` bounds how many SSTables accumulate
/// before a full merge.
#[derive(Debug, Clone)]
pub struct Options {
    pub dir: PathBuf,
    pub flush_threshold_bytes: usize,
    pub compaction_trigger: usize,
}

impl Options {
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Options {
            dir: dir.as_ref().to_path_buf(),
            flush_threshold_bytes: 4 * 1024 * 1024,
            compaction_trigger: 4,
        }
    }
    pub fn flush_threshold_bytes(mut self, n: usize) -> Self {
        self.flush_threshold_bytes = n;
        self
    }
    pub fn compaction_trigger(mut self, n: usize) -> Self {
        self.compaction_trigger = n;
        self
    }
}

/// The engine, generic over the I/O backend (std today, io_uring tomorrow).
pub struct Engine<B: IoBackend = StdIo> {
    opts: Options,
    backend: B,
    wal: Wal<B::Handle>,
    mem: MemTable,
    /// SSTable paths, newest first.
    tables: Vec<PathBuf>,
    next_table: u64,
}

impl Engine<StdIo> {
    /// Open (creating if needed) an engine rooted at `opts.dir` on the std backend.
    pub fn open(opts: Options) -> stdio::Result<Self> {
        Self::open_with(opts, StdIo)
    }
}

impl<B: IoBackend> Engine<B> {
    pub fn open_with(opts: Options, backend: B) -> stdio::Result<Self> {
        std::fs::create_dir_all(&opts.dir)?;

        // Discover existing SSTables (named NNNN.sst), newest = highest number.
        let mut tables: Vec<(u64, PathBuf)> = Vec::new();
        let mut next_table = 0u64;
        for entry in std::fs::read_dir(&opts.dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("sst")
                && let Some(n) =
                    path.file_stem().and_then(|s| s.to_str()).and_then(|s| s.parse::<u64>().ok())
            {
                next_table = next_table.max(n + 1);
                tables.push((n, path));
            }
        }
        tables.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
        let tables = tables.into_iter().map(|(_, p)| p).collect();

        // Replay the WAL into a fresh memtable.
        let mut wal = Wal::open(&backend, &opts.dir.join("wal.log"))?;
        let mut mem = MemTable::new();
        wal.replay(|rec| match rec.kind {
            record::KIND_DELETE => mem.delete(rec.key),
            _ => mem.put(rec.key, rec.value),
        })?;

        Ok(Engine { opts, backend, wal, mem, tables, next_table })
    }

    /// Durable put: WAL append+fsync, then memtable, then maybe flush.
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> stdio::Result<()> {
        let rec = Record::put(key.to_vec(), value.to_vec());
        self.wal.append(&rec)?;
        self.mem.put(key.to_vec(), value.to_vec());
        self.maybe_flush()
    }

    /// Durable delete (tombstone).
    pub fn delete(&mut self, key: &[u8]) -> stdio::Result<()> {
        let rec = Record::delete(key.to_vec());
        self.wal.append(&rec)?;
        self.mem.delete(key.to_vec());
        self.maybe_flush()
    }

    /// Point read: memtable first, then SSTables newest→oldest. Stops at the
    /// first table that knows the key (value or tombstone).
    pub fn get(&mut self, key: &[u8]) -> stdio::Result<Option<Vec<u8>>> {
        if let Some(slot) = self.mem.get(key) {
            return Ok(slot.clone());
        }
        for path in self.tables.clone() {
            let mut sst = SsTable::open(&self.backend, &path)?;
            if let Some(slot) = sst.get(key)? {
                return Ok(slot);
            }
        }
        Ok(None)
    }

    fn maybe_flush(&mut self) -> stdio::Result<()> {
        if self.mem.approx_bytes() >= self.opts.flush_threshold_bytes && !self.mem.is_empty() {
            self.flush()?;
        }
        Ok(())
    }

    /// Force the memtable to an SSTable and truncate the WAL.
    pub fn flush(&mut self) -> stdio::Result<()> {
        if self.mem.is_empty() {
            return Ok(());
        }
        let sorted = self.mem.drain_sorted();
        let path = self.opts.dir.join(format!("{:08}.sst", self.next_table));
        self.next_table += 1;
        SsTable::create(&self.backend, &path, &sorted)?;
        self.tables.insert(0, path); // newest first
        self.wal.reset(&self.backend, &self.opts.dir.join("wal.log"))?;
        self.maybe_compact()?;
        Ok(())
    }

    fn maybe_compact(&mut self) -> stdio::Result<()> {
        if self.tables.len() < self.opts.compaction_trigger {
            return Ok(());
        }
        // Full merge: read every table newest→oldest, reap tombstones (these
        // are all the data there is), write one table, delete the rest.
        let mut runs = Vec::new();
        for path in &self.tables {
            let mut sst = SsTable::open(&self.backend, path)?;
            runs.push(sst.records()?);
        }
        let merged = compaction::merge_and_reap(runs);
        let path = self.opts.dir.join(format!("{:08}.sst", self.next_table));
        self.next_table += 1;
        SsTable::create(&self.backend, &path, &merged)?;
        let old: Vec<PathBuf> = std::mem::take(&mut self.tables);
        for p in old {
            std::fs::remove_file(p).ok();
        }
        self.tables.push(path);
        Ok(())
    }

    /// Number of on-disk SSTables (observability for tests/benchmarks).
    pub fn table_count(&self) -> usize {
        self.tables.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("sophia-lsm-{}-{}", tag, std::process::id()))
    }

    #[test]
    fn put_get_delete() {
        let dir = tmp("pgd");
        std::fs::remove_dir_all(&dir).ok();
        let mut db = Engine::open(Options::new(&dir)).unwrap();
        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        assert_eq!(db.get(b"a").unwrap().as_deref(), Some(&b"1"[..]));
        db.delete(b"a").unwrap();
        assert_eq!(db.get(b"a").unwrap(), None);
        assert_eq!(db.get(b"b").unwrap().as_deref(), Some(&b"2"[..]));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn survives_reopen() {
        let dir = tmp("reopen");
        std::fs::remove_dir_all(&dir).ok();
        {
            let mut db = Engine::open(Options::new(&dir)).unwrap();
            db.put(b"durable", b"yes").unwrap();
        } // drop without explicit flush — WAL must carry it
        let mut db = Engine::open(Options::new(&dir)).unwrap();
        assert_eq!(db.get(b"durable").unwrap().as_deref(), Some(&b"yes"[..]));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flush_and_compact() {
        let dir = tmp("compact");
        std::fs::remove_dir_all(&dir).ok();
        let opts = Options::new(&dir).flush_threshold_bytes(1).compaction_trigger(3);
        let mut db = Engine::open(opts).unwrap();
        for i in 0..10u32 {
            db.put(format!("k{i}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
            db.flush().unwrap();
        }
        // Compaction should have collapsed the runs.
        assert!(db.table_count() <= 3, "tables={}", db.table_count());
        assert_eq!(db.get(b"k7").unwrap().as_deref(), Some(&b"v7"[..]));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn newest_value_wins_across_tables() {
        let dir = tmp("overwrite");
        std::fs::remove_dir_all(&dir).ok();
        let opts = Options::new(&dir).flush_threshold_bytes(1).compaction_trigger(100);
        let mut db = Engine::open(opts).unwrap();
        db.put(b"k", b"old").unwrap();
        db.flush().unwrap();
        db.put(b"k", b"new").unwrap();
        db.flush().unwrap();
        assert_eq!(db.get(b"k").unwrap().as_deref(), Some(&b"new"[..]));
        std::fs::remove_dir_all(&dir).ok();
    }
}
