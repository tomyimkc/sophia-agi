// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Bitcask-style durable KV engine over a single append-only log.
//!
//! Writes append a record to `data.log` and update an in-memory keydir
//! (`key -> (value_offset, value_len)`). Reads are one positional read of the
//! value bytes. Deletes append a tombstone. On open, the log is scanned to
//! rebuild the keydir; a torn tail (failed CRC) is truncated so recovery never
//! trusts a partial write. `compact()` rewrites only live values, reclaiming the
//! space held by overwritten/deleted keys.
//!
//! Simplifications vs production bitcask (documented, not hidden): one active
//! file rather than rolling segments, and no separate hint file — recovery
//! replays the whole log. Both are natural next steps; neither affects
//! correctness.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::reader::{BatchReader, ReadOp};
use crate::record::{self, HEADER_LEN};

#[derive(Clone, Copy)]
struct KeyEntry {
    value_offset: u64,
    value_len: u32,
    tstamp: u64,
}

pub struct Bitcask {
    dir: PathBuf,
    file: File,
    write_offset: u64,
    keydir: HashMap<Vec<u8>, KeyEntry>,
    next_tstamp: u64,
    sync_on_put: bool,
    /// Bytes written that are now dead (overwritten or tombstoned), for compaction heuristics.
    dead_bytes: u64,
}

impl Bitcask {
    const DATA_FILE: &'static str = "data.log";

    /// Open (creating if needed) the store at `dir`. `sync_on_put` fsyncs after
    /// every write for durability at the cost of throughput; `false` batches
    /// durability to OS flushing (use `sync()` at safe points).
    pub fn open(dir: impl AsRef<Path>, sync_on_put: bool) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(Self::DATA_FILE);
        // Never truncate: an existing log must be preserved and recovered.
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?;
        let mut store = Bitcask {
            dir,
            file,
            write_offset: 0,
            keydir: HashMap::new(),
            next_tstamp: 1,
            sync_on_put,
            dead_bytes: 0,
        };
        store.recover()?;
        Ok(store)
    }

    /// Replay the log, rebuilding the keydir and truncating a torn tail.
    fn recover(&mut self) -> io::Result<()> {
        let file_len = self.file.metadata()?.len();
        let mut offset = 0u64;
        let mut max_tstamp = 0u64;

        while offset + HEADER_LEN as u64 <= file_len {
            let mut header = [0u8; HEADER_LEN];
            if self.file.read_exact_at(&mut header, offset).is_err() {
                break;
            }
            let blen = record::body_len(&header);
            if offset + HEADER_LEN as u64 + blen as u64 > file_len {
                break; // truncated tail: body doesn't fit
            }
            let mut body = vec![0u8; blen];
            if self.file.read_exact_at(&mut body, offset + HEADER_LEN as u64).is_err() {
                break;
            }
            let rec = match record::parse(offset, &header, &body) {
                Some(r) => r,
                None => break, // CRC/length failure — stop at last good record
            };
            max_tstamp = max_tstamp.max(rec.tstamp);
            match rec.value {
                Some((value_offset, value_len)) => {
                    if let Some(old) = self.keydir.insert(
                        rec.key,
                        KeyEntry { value_offset, value_len, tstamp: rec.tstamp },
                    ) {
                        self.dead_bytes += old.value_len as u64;
                    }
                }
                None => {
                    if let Some(old) = self.keydir.remove(&rec.key) {
                        self.dead_bytes += old.value_len as u64;
                    }
                }
            }
            offset += rec.total_len;
        }

        // Trim any partial trailing bytes so the next append starts clean.
        if offset != file_len {
            self.file.set_len(offset)?;
            self.file.sync_all()?;
        }
        self.write_offset = offset;
        self.next_tstamp = max_tstamp + 1;
        Ok(())
    }

    pub fn put(&mut self, key: &[u8], val: &[u8]) -> io::Result<()> {
        let tstamp = self.next_tstamp;
        self.next_tstamp += 1;
        let rec = record::encode(key, Some(val), tstamp);
        self.file.write_all_at(&rec, self.write_offset)?;
        if self.sync_on_put {
            self.file.sync_data()?;
        }
        let value_offset = self.write_offset + HEADER_LEN as u64 + key.len() as u64;
        if let Some(old) = self.keydir.insert(
            key.to_vec(),
            KeyEntry { value_offset, value_len: val.len() as u32, tstamp },
        ) {
            self.dead_bytes += old.value_len as u64;
        }
        self.write_offset += rec.len() as u64;
        Ok(())
    }

    pub fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        let entry = match self.keydir.get(key) {
            Some(e) => *e,
            None => return Ok(None),
        };
        let mut buf = vec![0u8; entry.value_len as usize];
        self.file.read_exact_at(&mut buf, entry.value_offset)?;
        Ok(Some(buf))
    }

    pub fn delete(&mut self, key: &[u8]) -> io::Result<bool> {
        if !self.keydir.contains_key(key) {
            return Ok(false);
        }
        let tstamp = self.next_tstamp;
        self.next_tstamp += 1;
        let rec = record::encode(key, None, tstamp);
        self.file.write_all_at(&rec, self.write_offset)?;
        if self.sync_on_put {
            self.file.sync_data()?;
        }
        self.write_offset += rec.len() as u64;
        if let Some(old) = self.keydir.remove(key) {
            self.dead_bytes += old.value_len as u64;
        }
        Ok(true)
    }

    /// Batched point lookups. Missing keys map to `None`; present keys are read
    /// in one call through `reader` (the io_uring win lives here). Output is
    /// positional: `out[i]` corresponds to `keys[i]`.
    pub fn multi_get(
        &self,
        reader: &dyn BatchReader,
        keys: &[&[u8]],
    ) -> io::Result<Vec<Option<Vec<u8>>>> {
        let mut ops: Vec<ReadOp> = Vec::new();
        let mut slot_for_op: Vec<usize> = Vec::new();
        let mut out: Vec<Option<Vec<u8>>> = vec![None; keys.len()];

        for (i, key) in keys.iter().enumerate() {
            if let Some(e) = self.keydir.get(*key) {
                ops.push(ReadOp { offset: e.value_offset, buf: vec![0u8; e.value_len as usize] });
                slot_for_op.push(i);
            }
        }
        reader.read_batch(&self.file, &mut ops)?;
        for (op, slot) in ops.into_iter().zip(slot_for_op) {
            out[slot] = Some(op.buf);
        }
        Ok(out)
    }

    /// Rewrite the log keeping only live values, then atomically swap it in.
    /// Reclaims `dead_bytes`. Safe across crash: writes to a temp file and
    /// renames (rename is atomic on the same filesystem).
    pub fn compact(&mut self) -> io::Result<()> {
        let tmp_path = self.dir.join("data.compact");
        let tmp = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&tmp_path)?;

        let mut new_keydir: HashMap<Vec<u8>, KeyEntry> = HashMap::with_capacity(self.keydir.len());
        let mut offset = 0u64;
        // Collect keys first to avoid borrowing self while reading.
        let keys: Vec<Vec<u8>> = self.keydir.keys().cloned().collect();
        for key in keys {
            let entry = self.keydir[&key];
            let mut val = vec![0u8; entry.value_len as usize];
            self.file.read_exact_at(&mut val, entry.value_offset)?;
            let rec = record::encode(&key, Some(&val), entry.tstamp);
            tmp.write_all_at(&rec, offset)?;
            let value_offset = offset + HEADER_LEN as u64 + key.len() as u64;
            new_keydir.insert(key, KeyEntry { value_offset, value_len: entry.value_len, tstamp: entry.tstamp });
            offset += rec.len() as u64;
        }
        tmp.sync_all()?;

        let data_path = self.dir.join(Self::DATA_FILE);
        std::fs::rename(&tmp_path, &data_path)?;
        // Reopen the swapped-in file as the active handle.
        self.file = OpenOptions::new().read(true).write(true).open(&data_path)?;
        self.file.sync_all()?;
        self.keydir = new_keydir;
        self.write_offset = offset;
        self.dead_bytes = 0;
        Ok(())
    }

    /// fsync the log (use with `sync_on_put = false` at consistency points).
    pub fn sync(&self) -> io::Result<()> {
        self.file.sync_data()
    }

    pub fn len(&self) -> usize {
        self.keydir.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keydir.is_empty()
    }

    pub fn live_bytes(&self) -> u64 {
        self.write_offset.saturating_sub(self.dead_bytes)
    }

    pub fn dead_bytes(&self) -> u64 {
        self.dead_bytes
    }

    pub fn file_size(&self) -> u64 {
        self.write_offset
    }
}
