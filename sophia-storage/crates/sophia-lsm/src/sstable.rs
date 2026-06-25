// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Sorted String Table — an immutable, sorted, on-disk run of records.
//!
//! Layout: a sequence of framed records ([`crate::record`]) in ascending key
//! order, ending with a fixed footer holding a CRC-checked sparse index and the
//! key count. The sparse index lets a `get` binary-search to the right block
//! and scan a bounded window instead of the whole file.
//!
//! ```text
//! [ record* ][ index_entry* ][ index_len: u32-le ][ magic: u32-le ]
//! index_entry := [ klen: u32-le ][ key ][ offset: u64-le ]
//! ```
//!
//! This is a skeleton: one sparse entry every [`INDEX_STRIDE`] records, no
//! bloom filter yet (documented as the next step in DESIGN.md). It is correct
//! and crash-checked, which is the part that has to be right first.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use crate::io::{FileHandle, IoBackend};
use crate::record::{crc32, Record};

const SST_MAGIC: u32 = 0x5350_4831; // "SPH1"
const INDEX_STRIDE: usize = 16;

pub struct SsTable<H: FileHandle> {
    handle: H,
    /// Sparse index: first key of every Nth record -> byte offset.
    index: Vec<(Vec<u8>, u64)>,
    data_end: u64,
}

impl<H: FileHandle> SsTable<H> {
    /// Write a sorted map out as a new SSTable, then reopen it for reads.
    pub fn create<B: IoBackend<Handle = H>>(
        backend: &B,
        path: &Path,
        sorted: &BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    ) -> io::Result<Self> {
        let mut handle = backend.open(path)?;
        let mut offset = 0u64;
        let mut index: Vec<(Vec<u8>, u64)> = Vec::new();
        for (i, (key, val)) in sorted.iter().enumerate() {
            let rec = match val {
                Some(v) => Record::put(key.clone(), v.clone()),
                None => Record::delete(key.clone()),
            };
            let bytes = rec.encode();
            if i % INDEX_STRIDE == 0 {
                index.push((key.clone(), offset));
            }
            handle.append(&bytes)?;
            offset += bytes.len() as u64;
        }
        let data_end = offset;

        // Footer: index, then index length, then magic. CRC guards the index.
        let mut footer = Vec::new();
        for (key, off) in &index {
            footer.extend_from_slice(&(key.len() as u32).to_le_bytes());
            footer.extend_from_slice(key);
            footer.extend_from_slice(&off.to_le_bytes());
        }
        let index_crc = crc32(&footer);
        footer.extend_from_slice(&index_crc.to_le_bytes());
        // `index_len` counts [entries][crc] only; the trailing [index_len][magic]
        // (8 bytes) is fixed-size and read from the file tail.
        let index_len = footer.len() as u32;
        footer.extend_from_slice(&index_len.to_le_bytes());
        footer.extend_from_slice(&SST_MAGIC.to_le_bytes());
        handle.append(&footer)?;
        handle.sync()?;

        Ok(SsTable { handle, index, data_end })
    }

    /// Open an existing SSTable and load its sparse index from the footer.
    pub fn open<B: IoBackend<Handle = H>>(backend: &B, path: &Path) -> io::Result<Self> {
        let mut handle = backend.open(path)?;
        let len = handle.len()?;
        if len < 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "sstable too small"));
        }
        let mut tail = [0u8; 8];
        handle.read_at(len - 8, &mut tail)?;
        let index_len = u32::from_le_bytes(tail[0..4].try_into().unwrap()) as u64;
        let magic = u32::from_le_bytes(tail[4..8].try_into().unwrap());
        if magic != SST_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad sstable magic"));
        }
        // [entries][crc] live just before the fixed 8-byte [index_len][magic] tail.
        let footer_start = len - 8 - index_len;
        let mut footer = vec![0u8; index_len as usize]; // [entries][crc]
        handle.read_at(footer_start, &mut footer)?;
        let crc_pos = footer.len() - 4;
        let stored_crc = u32::from_le_bytes(footer[crc_pos..].try_into().unwrap());
        if crc32(&footer[..crc_pos]) != stored_crc {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "sstable index crc mismatch"));
        }
        let mut index = Vec::new();
        let mut p = 0usize;
        while p < crc_pos {
            let klen = u32::from_le_bytes(footer[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let key = footer[p..p + klen].to_vec();
            p += klen;
            let off = u64::from_le_bytes(footer[p..p + 8].try_into().unwrap());
            p += 8;
            index.push((key, off));
        }
        Ok(SsTable { handle, index, data_end: footer_start })
    }

    /// Point lookup. `Some(Some(v))` = value, `Some(None)` = tombstone here,
    /// `None` = key not in this table.
    pub fn get(&mut self, key: &[u8]) -> io::Result<Option<Option<Vec<u8>>>> {
        // Binary-search the sparse index for the block that could hold `key`.
        let start = match self.index.binary_search_by(|(k, _)| k.as_slice().cmp(key)) {
            Ok(i) => self.index[i].1,
            Err(0) => return Ok(None), // before the first key
            Err(i) => self.index[i - 1].1,
        };
        // Scan forward from the block start until we pass `key`.
        let mut off = start;
        while off < self.data_end {
            let rec = self.read_record_at(off)?;
            match rec.key.as_slice().cmp(key) {
                std::cmp::Ordering::Equal => {
                    return Ok(Some(if rec.is_tombstone() { None } else { Some(rec.value) }));
                }
                std::cmp::Ordering::Greater => return Ok(None),
                std::cmp::Ordering::Less => {
                    off += rec_len(&rec);
                }
            }
        }
        Ok(None)
    }

    /// Stream every record in key order (used by compaction).
    pub fn records(&mut self) -> io::Result<Vec<Record>> {
        let mut out = Vec::new();
        let mut off = 0u64;
        while off < self.data_end {
            let rec = self.read_record_at(off)?;
            off += rec_len(&rec);
            out.push(rec);
        }
        Ok(out)
    }

    fn read_record_at(&mut self, off: u64) -> io::Result<Record> {
        // Header is fixed 9 bytes; read it, then the payload + crc.
        let mut header = [0u8; 9];
        self.handle.read_at(off, &mut header)?;
        let klen = u32::from_le_bytes(header[1..5].try_into().unwrap()) as usize;
        let vlen = u32::from_le_bytes(header[5..9].try_into().unwrap()) as usize;
        let total = 9 + klen + vlen + 4;
        let mut buf = vec![0u8; total];
        self.handle.read_at(off, &mut buf)?;
        Record::decode(&buf)?
            .map(|(r, _)| r)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "sstable record corrupt"))
    }
}

fn rec_len(rec: &Record) -> u64 {
    (9 + rec.key.len() + rec.value.len() + 4) as u64
}
