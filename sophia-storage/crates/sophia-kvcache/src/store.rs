// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Block storage media behind a tier.
//!
//! A [`BlockStore`] is *where the bytes of a tier physically live*. Separating
//! it from the placement/eviction logic ([`crate::tier`], [`crate`]) is what
//! lets each tier use a different medium without the controller changing:
//!
//! - [`MemStore`] — an in-memory map. Stands in for device memory: real **HBM**
//!   would be a CUDA allocation, real host **DRAM** a pinned-buffer pool. The
//!   transfer into/out of it is a `memcpy` today, `cudaMemcpyAsync` for HBM.
//! - [`FileStore`] — **real on-disk persistence**, one file per block in a
//!   directory. This is a genuine **NVMe** tier: demoted blocks survive, are
//!   read back on promotion, and persist across cache restarts. The transfer is
//!   a `write`/`read` syscall today; `io_uring`/SPDK is the documented next step.
//!
//! Each store tracks cumulative `bytes_in`/`bytes_out` so the cache can report
//! the data volume crossing each tier boundary — exactly the number a zero-copy
//! / RDMA transfer path is built to shrink.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::block::{Block, BlockId};

/// The medium a tier's blocks live on.
pub trait BlockStore: Send {
    /// Store `block` (overwriting any prior copy of the same id).
    fn put(&mut self, block: Block) -> io::Result<()>;
    /// Read a block back without removing it.
    fn get(&self, id: BlockId) -> io::Result<Option<Block>>;
    /// Remove and return a block (the source side of a demotion/promotion).
    fn take(&mut self, id: BlockId) -> io::Result<Option<Block>>;
    /// Cheap in-memory presence check (never touches the medium).
    fn contains(&self, id: BlockId) -> bool;
    /// Resident ids (eviction-candidate enumeration). In-memory.
    fn ids(&self) -> Vec<BlockId>;
    /// Resident block count. In-memory.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Cumulative payload bytes written into this store.
    fn bytes_in(&self) -> u64;
    /// Cumulative payload bytes read out of this store.
    fn bytes_out(&self) -> u64;
}

/// In-memory store (HBM/DRAM stand-in). Transfers are `clone`/`memcpy`.
#[derive(Default)]
pub struct MemStore {
    blocks: HashMap<BlockId, Block>,
    bytes_in: u64,
    bytes_out: u64,
}

impl MemStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl BlockStore for MemStore {
    fn put(&mut self, block: Block) -> io::Result<()> {
        self.bytes_in += block.bytes() as u64;
        self.blocks.insert(block.id, block);
        Ok(())
    }
    fn get(&self, id: BlockId) -> io::Result<Option<Block>> {
        // A read is a copy out of the medium (a memcpy on a real device).
        Ok(self.blocks.get(&id).cloned())
    }
    fn take(&mut self, id: BlockId) -> io::Result<Option<Block>> {
        let b = self.blocks.remove(&id);
        if let Some(ref blk) = b {
            self.bytes_out += blk.bytes() as u64;
        }
        Ok(b)
    }
    fn contains(&self, id: BlockId) -> bool {
        self.blocks.contains_key(&id)
    }
    fn ids(&self) -> Vec<BlockId> {
        self.blocks.keys().copied().collect()
    }
    fn len(&self) -> usize {
        self.blocks.len()
    }
    fn bytes_in(&self) -> u64 {
        self.bytes_in
    }
    fn bytes_out(&self) -> u64 {
        self.bytes_out
    }
}

/// On-disk store (real NVMe tier). One file per block, named by id; an in-memory
/// index keeps `contains`/`ids`/`len` off the disk. Survives process restarts.
///
/// File format: `[ token_count: u32-le ][ payload ]`. The id is the filename, so
/// it is not duplicated in the body.
pub struct FileStore {
    dir: PathBuf,
    index: HashMap<BlockId, ()>,
    bytes_in: u64,
    bytes_out: u64,
}

impl FileStore {
    /// Open (creating if needed) a block directory, rebuilding the index from
    /// any block files already present.
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let mut index = HashMap::new();
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("blk")
                && let Some(id) = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| u64::from_str_radix(s, 16).ok())
            {
                index.insert(BlockId(id), ());
            }
        }
        Ok(FileStore { dir, index, bytes_in: 0, bytes_out: 0 })
    }

    fn path_of(&self, id: BlockId) -> PathBuf {
        self.dir.join(format!("{:016x}.blk", id.0))
    }
}

impl BlockStore for FileStore {
    fn put(&mut self, block: Block) -> io::Result<()> {
        let mut buf = Vec::with_capacity(4 + block.payload.len());
        buf.extend_from_slice(&block.token_count.to_le_bytes());
        buf.extend_from_slice(&block.payload);
        // Write to a temp then rename, so a reader never sees a half-written block.
        let final_path = self.path_of(block.id);
        let tmp_path = final_path.with_extension("blk.tmp");
        fs::write(&tmp_path, &buf)?;
        fs::rename(&tmp_path, &final_path)?;
        self.bytes_in += block.payload.len() as u64;
        self.index.insert(block.id, ());
        Ok(())
    }

    fn get(&self, id: BlockId) -> io::Result<Option<Block>> {
        if !self.index.contains_key(&id) {
            return Ok(None);
        }
        let raw = fs::read(self.path_of(id))?;
        if raw.len() < 4 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated block file"));
        }
        let token_count = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        let payload = raw[4..].to_vec();
        Ok(Some(Block::new(id, token_count, payload)))
    }

    fn take(&mut self, id: BlockId) -> io::Result<Option<Block>> {
        let block = self.get(id)?;
        if let Some(ref b) = block {
            self.bytes_out += b.payload.len() as u64;
            fs::remove_file(self.path_of(id)).ok();
            self.index.remove(&id);
        }
        Ok(block)
    }

    fn contains(&self, id: BlockId) -> bool {
        self.index.contains_key(&id)
    }
    fn ids(&self) -> Vec<BlockId> {
        self.index.keys().copied().collect()
    }
    fn len(&self) -> usize {
        self.index.len()
    }
    fn bytes_in(&self) -> u64 {
        self.bytes_in
    }
    fn bytes_out(&self) -> u64 {
        self.bytes_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_store_round_trips() {
        let mut s = MemStore::new();
        s.put(Block::new(BlockId(1), 4, vec![7u8; 16])).unwrap();
        assert!(s.contains(BlockId(1)));
        assert_eq!(s.get(BlockId(1)).unwrap().unwrap().payload, vec![7u8; 16]);
        assert_eq!(s.take(BlockId(1)).unwrap().unwrap().token_count, 4);
        assert!(!s.contains(BlockId(1)));
    }

    #[test]
    fn file_store_persists_to_disk_and_survives_reopen() {
        let dir = std::env::temp_dir().join(format!("sophia-kvc-store-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        {
            let mut s = FileStore::open(&dir).unwrap();
            s.put(Block::new(BlockId(0xABCD), 8, vec![3u8; 64])).unwrap();
            assert!(dir.join("000000000000abcd.blk").exists(), "block must hit the disk");
            assert!(s.bytes_in() >= 64);
        }
        // Reopen: the index rebuilds from disk, payload reads back intact.
        let s2 = FileStore::open(&dir).unwrap();
        assert!(s2.contains(BlockId(0xABCD)));
        let back = s2.get(BlockId(0xABCD)).unwrap().unwrap();
        assert_eq!(back.payload, vec![3u8; 64]);
        assert_eq!(back.token_count, 8);
        std::fs::remove_dir_all(&dir).ok();
    }
}
