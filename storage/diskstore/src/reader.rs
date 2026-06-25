// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Batched positional reads — the seam where io_uring earns its keep.
//!
//! A point `get` is a single `pread`; there is nothing to gain there. The win
//! is `multi_get`: many random-offset reads issued together. `StdReader` loops
//! `pread` (one syscall per read); `UringReader` pushes them all into one
//! io_uring submission and reaps the completions, collapsing N syscalls into
//! ~one and letting the kernel pipeline the I/O.

use std::fs::File;
use std::io;

/// One positional read: fill `buf` (pre-sized to the wanted length) from `offset`.
pub struct ReadOp {
    pub offset: u64,
    pub buf: Vec<u8>,
}

pub trait BatchReader: Send + Sync {
    /// Fill every op's buffer. On success all buffers are fully populated.
    fn read_batch(&self, file: &File, ops: &mut [ReadOp]) -> io::Result<()>;
    fn name(&self) -> &'static str;
}

/// Portable backend: `pread` per op via `FileExt::read_exact_at`. Always built.
pub struct StdReader;

impl BatchReader for StdReader {
    fn read_batch(&self, file: &File, ops: &mut [ReadOp]) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        for op in ops.iter_mut() {
            file.read_exact_at(&mut op.buf, op.offset)?;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "std(pread)"
    }
}

#[cfg(feature = "io_uring")]
pub use uring::UringReader;

#[cfg(feature = "io_uring")]
mod uring {
    use super::{BatchReader, ReadOp};
    use io_uring::{opcode, types, IoUring};
    use std::fs::File;
    use std::io;
    use std::os::unix::io::AsRawFd;
    use std::sync::Mutex;

    /// io_uring backend. Holds one ring (guarded by a Mutex so the reader is
    /// `Sync`); batches larger than the ring depth are submitted in chunks.
    pub struct UringReader {
        ring: Mutex<IoUring>,
        depth: usize,
    }

    impl UringReader {
        pub fn new(depth: u32) -> io::Result<Self> {
            let depth = depth.next_power_of_two().clamp(8, 4096);
            Ok(UringReader {
                ring: Mutex::new(IoUring::new(depth)?),
                depth: depth as usize,
            })
        }
    }

    impl BatchReader for UringReader {
        fn read_batch(&self, file: &File, ops: &mut [ReadOp]) -> io::Result<()> {
            let fd = types::Fd(file.as_raw_fd());
            let mut ring = self.ring.lock().unwrap();
            for chunk in ops.chunks_mut(self.depth) {
                // Stage every read in the submission queue, tagged by index.
                for (i, op) in chunk.iter_mut().enumerate() {
                    let len = op.buf.len() as u32;
                    let entry = opcode::Read::new(fd, op.buf.as_mut_ptr(), len)
                        .offset(op.offset)
                        .build()
                        .user_data(i as u64);
                    // Safe: buffers in `chunk` outlive submit_and_wait below.
                    unsafe {
                        ring.submission()
                            .push(&entry)
                            .map_err(|_| io::Error::other("io_uring submission queue full"))?;
                    }
                }
                ring.submit_and_wait(chunk.len())?;

                // Reap completions; verify each fully read its expected length.
                let mut completed = 0;
                let cq = ring.completion();
                for cqe in cq {
                    let idx = cqe.user_data() as usize;
                    let res = cqe.result();
                    if res < 0 {
                        return Err(io::Error::from_raw_os_error(-res));
                    }
                    if res as usize != chunk[idx].buf.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "io_uring short read",
                        ));
                    }
                    completed += 1;
                }
                if completed != chunk.len() {
                    return Err(io::Error::other("io_uring missing completions"));
                }
            }
            Ok(())
        }

        fn name(&self) -> &'static str {
            "io_uring"
        }
    }
}
