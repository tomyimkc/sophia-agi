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

pub mod bloom;
pub mod compaction;
pub mod io;
pub mod levels;
pub mod memtable;
pub mod record;
pub mod sstable;
pub mod wal;

use std::io as stdio;
use std::path::{Path, PathBuf};

use crate::io::{IoBackend, StdIo};
use crate::levels::{Levels, TableMeta};
use crate::memtable::MemTable;
use crate::record::Record;
use crate::sstable::SsTable;
use crate::wal::Wal;

const MANIFEST: &str = "MANIFEST";

/// Engine tuning. `flush_threshold_bytes` bounds the memtable before it spills
/// to an SSTable; `compaction_trigger` is how many L0 tables accumulate before
/// L0 is merged into L1; `l1_base_records` is the L1 size budget (each deeper
/// level is `levels::FANOUT`× larger).
#[derive(Debug, Clone)]
pub struct Options {
    pub dir: PathBuf,
    pub flush_threshold_bytes: usize,
    pub compaction_trigger: usize,
    pub l1_base_records: usize,
}

impl Options {
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Options {
            dir: dir.as_ref().to_path_buf(),
            flush_threshold_bytes: 4 * 1024 * 1024,
            compaction_trigger: 4,
            l1_base_records: 4096,
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
    pub fn l1_base_records(mut self, n: usize) -> Self {
        self.l1_base_records = n.max(1);
        self
    }
}

/// A set of mutations committed atomically with a *single* fsync (group
/// commit). Build it up, then hand it to [`Engine::write_batch`]. This is the
/// RocksDB `WriteBatch` shape and the primitive that amortizes the durability
/// cost the per-op `put`/`delete` path pays on every call.
#[derive(Debug, Default, Clone)]
pub struct WriteBatch {
    records: Vec<Record>,
}

impl WriteBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn put(&mut self, key: &[u8], value: &[u8]) -> &mut Self {
        self.records.push(Record::put(key.to_vec(), value.to_vec()));
        self
    }

    pub fn delete(&mut self, key: &[u8]) -> &mut Self {
        self.records.push(Record::delete(key.to_vec()));
        self
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// The engine, generic over the I/O backend (std today, io_uring tomorrow).
pub struct Engine<B: IoBackend = StdIo> {
    opts: Options,
    backend: B,
    wal: Wal<B::Handle>,
    mem: MemTable,
    /// Leveled table layout (L0 + deeper levels).
    levels: Levels,
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

        // Load the level layout from the manifest (authoritative); SSTables not
        // listed in it are orphans from an interrupted compaction and ignored.
        let manifest_path = opts.dir.join(MANIFEST);
        let levels = match std::fs::read_to_string(&manifest_path) {
            Ok(text) => Levels::decode(&text),
            Err(_) => Levels::new(),
        };
        // next_table must clear *every* existing .sst id, including orphans left
        // by an interrupted compaction — otherwise a reused id would append onto
        // a stale file (handles open with truncate=false) and corrupt it.
        let mut max_id = levels.read_order().iter().copied().max();
        if let Ok(rd) = std::fs::read_dir(&opts.dir) {
            for entry in rd.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("sst")
                    && let Some(n) =
                        path.file_stem().and_then(|s| s.to_str()).and_then(|s| s.parse::<u64>().ok())
                {
                    max_id = Some(max_id.map_or(n, |m| m.max(n)));
                }
            }
        }
        let next_table = max_id.map_or(0, |m| m + 1);

        // Replay the WAL into a fresh memtable.
        let mut wal = Wal::open(&backend, &opts.dir.join("wal.log"))?;
        let mut mem = MemTable::new();
        wal.replay(|rec| match rec.kind {
            record::KIND_DELETE => mem.delete(rec.key),
            _ => mem.put(rec.key, rec.value),
        })?;

        Ok(Engine { opts, backend, wal, mem, levels, next_table })
    }

    /// Durable put: WAL append+fsync, then memtable, then maybe flush.
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> stdio::Result<()> {
        let rec = Record::put(key.to_vec(), value.to_vec());
        self.wal.append(&rec)?;
        self.mem.put(key.to_vec(), value.to_vec());
        self.maybe_flush()
    }

    /// Commit a batch with a single fsync (group commit), then apply all
    /// mutations to the memtable and flush at most once. Same durability as N
    /// separate `put`/`delete` calls; one fsync instead of N.
    pub fn write_batch(&mut self, batch: &WriteBatch) -> stdio::Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        self.wal.append_batch(&batch.records)?;
        for rec in &batch.records {
            match rec.kind {
                record::KIND_DELETE => self.mem.delete(rec.key.clone()),
                _ => self.mem.put(rec.key.clone(), rec.value.clone()),
            }
        }
        self.maybe_flush()
    }

    /// Durable delete (tombstone).
    pub fn delete(&mut self, key: &[u8]) -> stdio::Result<()> {
        let rec = Record::delete(key.to_vec());
        self.wal.append(&rec)?;
        self.mem.delete(key.to_vec());
        self.maybe_flush()
    }

    /// Point read: memtable first, then tables in level read-order (L0
    /// newest→oldest, then L1, L2, …). Stops at the first table that knows the
    /// key. Each table's bloom filter skips it with no I/O on a definite miss.
    pub fn get(&mut self, key: &[u8]) -> stdio::Result<Option<Vec<u8>>> {
        if let Some(slot) = self.mem.get(key) {
            return Ok(slot.clone());
        }
        for id in self.levels.read_order() {
            let mut sst = SsTable::open(&self.backend, &self.path_for(id))?;
            if !sst.might_contain(key) {
                continue; // bloom miss: definitely absent, no scan
            }
            if let Some(slot) = sst.get(key)? {
                return Ok(slot);
            }
        }
        Ok(None)
    }

    fn path_for(&self, id: u64) -> PathBuf {
        self.opts.dir.join(format!("{id:08}.sst"))
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_table;
        self.next_table += 1;
        id
    }

    fn read_records(&self, id: u64) -> stdio::Result<Vec<Record>> {
        SsTable::open(&self.backend, &self.path_for(id))?.records()
    }

    fn write_manifest(&self) -> stdio::Result<()> {
        let tmp = self.opts.dir.join("MANIFEST.tmp");
        std::fs::write(&tmp, self.levels.encode())?;
        std::fs::rename(&tmp, self.opts.dir.join(MANIFEST))
    }

    fn maybe_flush(&mut self) -> stdio::Result<()> {
        if self.mem.approx_bytes() >= self.opts.flush_threshold_bytes && !self.mem.is_empty() {
            self.flush()?;
        }
        Ok(())
    }

    /// Force the memtable to an L0 SSTable, commit the manifest, truncate the
    /// WAL, then run any triggered compaction.
    ///
    /// Ordering is crash-safe: the SSTable is durable before the manifest names
    /// it, and the WAL is only reset after the manifest commits — so a crash at
    /// any point leaves either the WAL or a committed table holding the data.
    pub fn flush(&mut self) -> stdio::Result<()> {
        if self.mem.is_empty() {
            return Ok(());
        }
        let sorted = self.mem.drain_sorted();
        let id = self.alloc_id();
        SsTable::create(&self.backend, &self.path_for(id), &sorted)?;
        self.levels.push_l0(TableMeta { id, records: sorted.len() });
        self.write_manifest()?;
        self.wal.reset(&self.backend, &self.opts.dir.join("wal.log"))?;
        self.maybe_compact()?;
        Ok(())
    }

    /// Leveled compaction: merge L0→L1 when L0 fills, then cascade any level
    /// that exceeds its record budget down into the next. A key is rewritten
    /// O(levels) times, not O(dataset) as full-merge would.
    fn maybe_compact(&mut self) -> stdio::Result<()> {
        if self.levels.l0.len() >= self.opts.compaction_trigger {
            self.compact_l0_into_l1()?;
        }
        let mut level = 1;
        while level <= self.levels.deeper.len() {
            if let Some(meta) = self.levels.get_deeper(level)
                && meta.records > Levels::budget(level, self.opts.l1_base_records)
            {
                self.compact_down(level)?;
            }
            level += 1;
        }
        Ok(())
    }

    fn compact_l0_into_l1(&mut self) -> stdio::Result<()> {
        // L0 (newest→oldest) over L1 (oldest). Reap tombstones only if L1 is the
        // deepest level — otherwise a tombstone must keep shadowing deeper data.
        let reap = self.levels.max_level() <= 1;
        let mut runs = Vec::new();
        for t in self.levels.l0.clone() {
            runs.push(self.read_records(t.id)?);
        }
        if let Some(l1) = self.levels.get_deeper(1) {
            runs.push(self.read_records(l1.id)?);
        }
        let old_ids: Vec<u64> =
            self.levels.l0.iter().map(|t| t.id).chain(self.levels.get_deeper(1).map(|t| t.id)).collect();
        self.replace_level(1, runs, reap, old_ids, /*clear_l0=*/ true)
    }

    fn compact_down(&mut self, level: usize) -> stdio::Result<()> {
        let target = level + 1;
        let reap = self.levels.max_level() <= target;
        let src = self.levels.get_deeper(level).expect("compact_down on empty level");
        let mut runs = vec![self.read_records(src.id)?]; // Li is newer than L(i+1)
        if let Some(t) = self.levels.get_deeper(target) {
            runs.push(self.read_records(t.id)?);
        }
        let old_ids: Vec<u64> =
            std::iter::once(src.id).chain(self.levels.get_deeper(target).map(|t| t.id)).collect();
        self.levels.set_deeper(level, None);
        self.replace_level(target, runs, reap, old_ids, /*clear_l0=*/ false)
    }

    /// Merge `runs` (newest-first) into a single table at `target` level, commit
    /// the manifest, then delete the now-superseded source files.
    fn replace_level(
        &mut self,
        target: usize,
        runs: Vec<Vec<Record>>,
        reap: bool,
        old_ids: Vec<u64>,
        clear_l0: bool,
    ) -> stdio::Result<()> {
        let merged =
            if reap { compaction::merge_and_reap(runs) } else { compaction::merge(runs) };

        if clear_l0 {
            self.levels.l0.clear();
        }
        if merged.is_empty() {
            self.levels.set_deeper(target, None);
        } else {
            let id = self.alloc_id();
            SsTable::create(&self.backend, &self.path_for(id), &merged)?;
            self.levels.set_deeper(target, Some(TableMeta { id, records: merged.len() }));
        }
        self.write_manifest()?; // commit before deleting old inputs
        for id in old_ids {
            std::fs::remove_file(self.path_for(id)).ok();
        }
        Ok(())
    }

    /// Number of live SSTables across all levels (observability).
    pub fn table_count(&self) -> usize {
        self.levels.table_count()
    }

    /// Deepest populated level (observability for tests/benchmarks).
    pub fn max_level(&self) -> usize {
        self.levels.max_level()
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
    fn write_batch_is_durable_and_atomic() {
        let dir = tmp("batch");
        std::fs::remove_dir_all(&dir).ok();
        {
            let mut db = Engine::open(Options::new(&dir)).unwrap();
            let mut wb = WriteBatch::new();
            wb.put(b"a", b"1").put(b"b", b"2").delete(b"a");
            assert_eq!(wb.len(), 3);
            db.write_batch(&wb).unwrap();
            // within the same engine: delete shadows the put
            assert_eq!(db.get(b"a").unwrap(), None);
            assert_eq!(db.get(b"b").unwrap().as_deref(), Some(&b"2"[..]));
        }
        // after reopen (WAL replay of the batch): same view
        let mut db = Engine::open(Options::new(&dir)).unwrap();
        assert_eq!(db.get(b"a").unwrap(), None);
        assert_eq!(db.get(b"b").unwrap().as_deref(), Some(&b"2"[..]));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "io_uring")]
    #[test]
    fn io_uring_backend_round_trips() {
        use crate::io::IoUringIo;
        let dir = tmp("uring");
        std::fs::remove_dir_all(&dir).ok();
        let opts = Options::new(&dir).flush_threshold_bytes(1).compaction_trigger(100);
        {
            let mut db = Engine::open_with(opts.clone(), IoUringIo).unwrap();
            // single writes
            db.put(b"a", b"1").unwrap();
            db.put(b"b", b"2").unwrap();
            db.flush().unwrap(); // forces an SSTable write+read through the ring
            // group commit through the ring (one submission, one fsync)
            let mut wb = WriteBatch::new();
            wb.put(b"c", b"3").put(b"d", b"4").delete(b"a");
            db.write_batch(&wb).unwrap();
            assert_eq!(db.get(b"a").unwrap(), None);
            assert_eq!(db.get(b"b").unwrap().as_deref(), Some(&b"2"[..]));
            assert_eq!(db.get(b"c").unwrap().as_deref(), Some(&b"3"[..]));
        }
        // reopen on the ring backend: WAL replay + SSTable reads must agree
        let mut db = Engine::open_with(opts, IoUringIo).unwrap();
        assert_eq!(db.get(b"d").unwrap().as_deref(), Some(&b"4"[..]));
        assert_eq!(db.get(b"b").unwrap().as_deref(), Some(&b"2"[..]));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_batch_is_noop() {
        let dir = tmp("emptybatch");
        std::fs::remove_dir_all(&dir).ok();
        let mut db = Engine::open(Options::new(&dir)).unwrap();
        db.write_batch(&WriteBatch::new()).unwrap();
        assert_eq!(db.get(b"x").unwrap(), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn leveled_compaction_cascades_and_stays_correct() {
        let dir = tmp("leveled");
        std::fs::remove_dir_all(&dir).ok();
        // Tiny budgets so a few hundred keys force L0->L1->L2 cascades.
        let opts = Options::new(&dir)
            .flush_threshold_bytes(1) // flush on every write
            .compaction_trigger(2) // merge L0 into L1 every 2 tables
            .l1_base_records(4); // L1 budget 4 records, L2 budget 40
        let mut db = Engine::open(opts).unwrap();
        for i in 0..300u32 {
            db.put(format!("key:{i:04}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
        }
        assert!(db.max_level() >= 2, "expected a cascade past L1, got L{}", db.max_level());
        // Every key still reads back correctly through the leveled layout.
        for i in [0u32, 1, 42, 150, 299] {
            assert_eq!(
                db.get(format!("key:{i:04}").as_bytes()).unwrap().as_deref(),
                Some(format!("v{i}").as_bytes()),
                "lost key {i} after compaction"
            );
        }
        // A deleted key stays deleted through reaping.
        db.delete(b"key:0150").unwrap();
        for i in 0..20u32 {
            db.put(format!("flush{i}").as_bytes(), b"x").unwrap(); // churn to force compaction
        }
        assert_eq!(db.get(b"key:0150").unwrap(), None, "tombstone resurrected by compaction");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bloom_filters_keep_absent_lookups_correct() {
        let dir = tmp("bloom");
        std::fs::remove_dir_all(&dir).ok();
        let opts = Options::new(&dir).flush_threshold_bytes(1).compaction_trigger(3);
        let mut db = Engine::open(opts).unwrap();
        for i in 0..200u32 {
            db.put(format!("present:{i}").as_bytes(), b"1").unwrap();
        }
        // Present keys found; absent keys correctly None (bloom short-circuits).
        for i in 0..200u32 {
            assert!(db.get(format!("present:{i}").as_bytes()).unwrap().is_some());
        }
        for i in 0..500u32 {
            assert_eq!(db.get(format!("absent:{i}").as_bytes()).unwrap(), None);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn survives_reopen_after_compaction() {
        let dir = tmp("reopen-compact");
        std::fs::remove_dir_all(&dir).ok();
        {
            let opts =
                Options::new(&dir).flush_threshold_bytes(1).compaction_trigger(2).l1_base_records(4);
            let mut db = Engine::open(opts).unwrap();
            for i in 0..100u32 {
                db.put(format!("k{i:03}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
            }
        }
        // Reopen: manifest restores the level layout; data intact.
        let mut db = Engine::open(Options::new(&dir)).unwrap();
        for i in [0u32, 50, 99] {
            assert_eq!(
                db.get(format!("k{i:03}").as_bytes()).unwrap().as_deref(),
                Some(format!("v{i}").as_bytes())
            );
        }
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
