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
    /// Append `buf`, returning the offset at which it landed. Does NOT sync.
    fn append(&mut self, buf: &[u8]) -> io::Result<u64>;
    /// Append many buffers back-to-back in one logical batch, returning the
    /// offset of the first. Does NOT sync — the caller issues one [`sync`] for
    /// the whole batch (this is the group-commit primitive). The default loops
    /// over [`append`]; the io_uring backend overrides it to submit all writes
    /// in a single ring submission.
    ///
    /// [`sync`]: FileHandle::sync
    /// [`append`]: FileHandle::append
    fn append_many(&mut self, bufs: &[&[u8]]) -> io::Result<u64> {
        let mut first = None;
        for buf in bufs {
            let off = self.append(buf)?;
            first.get_or_insert(off);
        }
        Ok(first.unwrap_or(self.len()?))
    }
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

// ---------------------------------------------------------------------------
// io_uring backend (real, feature-gated). Linux 5.1+.
// ---------------------------------------------------------------------------
//
// This submits Write/Read/Fsync SQEs against a per-file ring. The group-commit
// win lives in `append_many`: every record in a batch becomes one Write SQE and
// they are submitted in a single `io_uring_enter`, then a single Fsync SQE makes
// the whole batch durable — instead of a syscall-per-write + fsync-per-write.
//
// Next steps (documented, not yet wired): `O_DIRECT` + page-aligned registered
// buffers for true zero-copy, and SQPOLL to drop the submit syscall entirely.
#[cfg(feature = "io_uring")]
mod uring {
    use super::*;
    use io_uring::{opcode, types, IoUring};
    use std::os::fd::AsRawFd;

    const RING_ENTRIES: u32 = 256;

    pub struct IoUringIo;

    impl IoBackend for IoUringIo {
        type Handle = UringFile;
        fn open(&self, path: &Path) -> io::Result<UringFile> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)?;
            let len = file.metadata()?.len();
            let ring = IoUring::new(RING_ENTRIES)?;
            Ok(UringFile { file, ring, len })
        }
    }

    pub struct UringFile {
        file: File,
        ring: IoUring,
        len: u64,
    }

    impl UringFile {
        /// Push one SQE, submitting+draining if the submission queue is full.
        fn push(&mut self, entry: &io_uring::squeue::Entry, inflight: &mut u32) -> io::Result<()> {
            // SAFETY: the buffers referenced by `entry` outlive the submission
            // (callers hold them across the whole append_many/read_at/sync call).
            while unsafe { self.ring.submission().push(entry).is_err() } {
                self.ring.submit()?;
                self.drain(inflight)?;
            }
            *inflight += 1;
            Ok(())
        }

        /// Wait for and validate `inflight` completions.
        fn drain(&mut self, inflight: &mut u32) -> io::Result<()> {
            if *inflight == 0 {
                return Ok(());
            }
            self.ring.submit_and_wait(*inflight as usize)?;
            let mut cq = self.ring.completion();
            let mut seen = 0u32;
            for cqe in &mut cq {
                let res = cqe.result();
                if res < 0 {
                    return Err(io::Error::from_raw_os_error(-res));
                }
                seen += 1;
            }
            *inflight -= seen;
            Ok(())
        }
    }

    impl FileHandle for UringFile {
        fn append(&mut self, buf: &[u8]) -> io::Result<u64> {
            self.append_many(&[buf])
        }

        fn append_many(&mut self, bufs: &[&[u8]]) -> io::Result<u64> {
            let fd = types::Fd(self.file.as_raw_fd());
            let first = self.len;
            let mut off = self.len;
            let mut total = 0u64;
            let mut inflight = 0u32;
            for buf in bufs {
                let w = opcode::Write::new(fd, buf.as_ptr(), buf.len() as u32)
                    .offset(off)
                    .build()
                    .user_data(off);
                self.push(&w, &mut inflight)?;
                off += buf.len() as u64;
                total += buf.len() as u64;
            }
            self.drain(&mut inflight)?;
            self.len += total;
            Ok(first)
        }

        fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            let fd = types::Fd(self.file.as_raw_fd());
            let r = opcode::Read::new(fd, buf.as_mut_ptr(), buf.len() as u32)
                .offset(offset)
                .build()
                .user_data(offset);
            let mut inflight = 0u32;
            self.push(&r, &mut inflight)?;
            self.drain(&mut inflight)
        }

        fn len(&self) -> io::Result<u64> {
            Ok(self.len)
        }

        fn sync(&mut self) -> io::Result<()> {
            let fd = types::Fd(self.file.as_raw_fd());
            let f = opcode::Fsync::new(fd).build().user_data(u64::MAX);
            let mut inflight = 0u32;
            self.push(&f, &mut inflight)?;
            self.drain(&mut inflight)
        }
    }
}

#[cfg(feature = "io_uring")]
pub use uring::{IoUringIo, UringFile};
