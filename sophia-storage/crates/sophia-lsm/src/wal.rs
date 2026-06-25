// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Write-ahead log.
//!
//! Every mutation is framed (see [`crate::record`]) and appended here *before*
//! it touches the memtable, with an `fsync` at the durability boundary. On open
//! we replay the log to reconstruct the memtable; a torn or corrupt tail stops
//! replay cleanly (the last partial write is simply lost — at-least-once).

use std::io;
use std::path::Path;

use crate::io::{FileHandle, IoBackend};
use crate::record::Record;

pub struct Wal<H: FileHandle> {
    handle: H,
}

impl<H: FileHandle> Wal<H> {
    pub fn open<B: IoBackend<Handle = H>>(backend: &B, path: &Path) -> io::Result<Self> {
        Ok(Wal { handle: backend.open(path)? })
    }

    /// Append one record and fsync. Returns once the write is durable.
    pub fn append(&mut self, record: &Record) -> io::Result<()> {
        let bytes = record.encode();
        self.handle.append(&bytes)?;
        self.handle.sync()
    }

    /// Replay the whole log, invoking `apply` for each surviving record in
    /// order. Stops at the first torn/corrupt frame.
    pub fn replay<F: FnMut(Record)>(&mut self, mut apply: F) -> io::Result<()> {
        let len = self.handle.len()?;
        if len == 0 {
            return Ok(());
        }
        let mut buf = vec![0u8; len as usize];
        self.handle.read_at(0, &mut buf)?;
        let mut off = 0usize;
        while off < buf.len() {
            match Record::decode(&buf[off..])? {
                Some((rec, consumed)) => {
                    off += consumed;
                    apply(rec);
                }
                None => break, // clean end / torn tail
            }
        }
        Ok(())
    }

    /// Truncate the log to empty (called after a successful memtable flush; the
    /// data now lives durably in an SSTable, so the WAL prefix is redundant).
    pub fn reset<B: IoBackend<Handle = H>>(&mut self, backend: &B, path: &Path) -> io::Result<()> {
        // Truncate by reopening with O_TRUNC semantics via std; the std backend
        // re-creates an empty file. (A uring backend would ftruncate(0).)
        std::fs::File::create(path)?;
        self.handle = backend.open(path)?;
        Ok(())
    }
}
