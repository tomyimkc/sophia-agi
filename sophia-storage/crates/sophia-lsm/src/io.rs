// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Pluggable I/O backend.
//!
//! The engine never calls `std::fs` directly for the hot path; it goes through
//! [`IoBackend`]. Today there is one implementation, [`StdIo`] (blocking
//! `pread`/`pwrite` + `fsync`). The point of the trait is the *seam*: an
//! `io_uring` backend (batched, async, `O_DIRECT`, registered buffers) drops in
//! here without the WAL, SSTable, or compaction code changing a line.
//!
//! Why a seam and not a real uring backend yet: the `io-uring` crate pulls a
//! Linux-only dependency and we keep this workspace dependency-free so it builds
//! and tests on any host. The contract below is what the uring backend must
//! satisfy; `IoUringIo` documents the wiring points.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// A positioned, durable file abstraction. Offsets are explicit so a future
/// backend can issue concurrent positioned reads without a shared cursor.
pub trait IoBackend: Send + Sync {
    type Handle: FileHandle;

    /// Open (creating if absent) a file for read+append/positioned writes.
    fn open(&self, path: &Path) -> io::Result<Self::Handle>;
}

pub trait FileHandle: Send {
    /// Append `buf`, returning the offset at which it landed.
    fn append(&mut self, buf: &[u8]) -> io::Result<u64>;
    /// Positioned read of exactly `buf.len()` bytes at `offset`.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
    /// Current logical length in bytes.
    fn len(&self) -> io::Result<u64>;
    /// Flush OS buffers to the device. The durability boundary.
    fn sync(&mut self) -> io::Result<()>;
    fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// Blocking std backend. Correct and portable; not the fast path.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdIo;

impl IoBackend for StdIo {
    type Handle = StdFile;
    fn open(&self, path: &Path) -> io::Result<StdFile> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false) // keep existing contents; we append / positioned-write
            .open(path)?;
        Ok(StdFile { file })
    }
}

#[derive(Debug)]
pub struct StdFile {
    file: File,
}

impl FileHandle for StdFile {
    fn append(&mut self, buf: &[u8]) -> io::Result<u64> {
        let offset = self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(buf)?;
        Ok(offset)
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(buf)
    }

    fn len(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    fn sync(&mut self) -> io::Result<()> {
        self.file.sync_all()
    }
}

/// Documented seam for the io_uring backend.
///
/// Wiring plan (when `feature = "io_uring"` is built out):
///  - submit batched `Readv`/`Writev`/`Fsync` SQEs against a per-engine ring;
///  - open files `O_DIRECT` and use registered, page-aligned buffers to make
///    `read_at`/`append` zero-copy from the page cache's perspective;
///  - keep `sync` mapped to an `Fsync` SQE so the WAL group-commit batches.
///
/// Until then this is intentionally unimplemented so nothing silently falls
/// back to slow I/O while pretending to be fast.
#[cfg(feature = "io_uring")]
pub struct IoUringIo;

#[cfg(feature = "io_uring")]
impl IoBackend for IoUringIo {
    type Handle = StdFile;
    fn open(&self, _path: &Path) -> io::Result<StdFile> {
        unimplemented!("io_uring backend: see io.rs wiring plan")
    }
}
