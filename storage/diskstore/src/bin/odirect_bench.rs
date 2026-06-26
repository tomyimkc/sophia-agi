// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! diskstore-odirect-bench — the *real* io_uring win, under O_DIRECT cold I/O.
//!
//! The page-cached benchmark (`diskstore-bench`) showed io_uring tying pread,
//! because a warm `pread` is a cheap syscall with nothing to block on. This
//! benchmark removes the page cache from the picture: it reads with **O_DIRECT**
//! so every read goes to the device. Now serial `pread` is latency-bound (one
//! outstanding I/O at a time) while io_uring keeps `--depth` reads in flight, so
//! the device queue stays full — the regime where io_uring earns its keep.
//!
//! Requires an O_DIRECT-capable filesystem (not tmpfs) and the `io_uring`
//! feature for the io_uring side. Usage:
//!   diskstore-odirect-bench [--blocks 200000] [--reads 100000] [--depth 128]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::os::unix::io::{AsRawFd, RawFd};
use std::time::Instant;

const BLOCK: usize = 4096; // O_DIRECT alignment unit (also page size)

struct Args {
    blocks: usize,
    reads: usize,
    depth: usize,
}

fn parse() -> Args {
    let mut a = Args { blocks: 200_000, reads: 100_000, depth: 128 };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut n = || it.next().and_then(|v| v.parse().ok()).expect("int value");
        match flag.as_str() {
            "--blocks" => a.blocks = n(),
            "--reads" => a.reads = n(),
            "--depth" => a.depth = n(),
            "-h" | "--help" => {
                println!("usage: diskstore-odirect-bench [--blocks N] [--reads N] [--depth N]");
                std::process::exit(0);
            }
            other => panic!("unknown arg {other}"),
        }
    }
    a
}

/// A heap buffer aligned to `BLOCK`, required for O_DIRECT.
struct Aligned {
    ptr: *mut u8,
    len: usize,
}
impl Aligned {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len, BLOCK).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "alloc failed");
        Aligned { ptr, len }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, BLOCK).unwrap();
        unsafe { dealloc(self.ptr, layout) };
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 17
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(((p / 100.0) * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)]
}

/// Serial pread loop on an O_DIRECT fd — one outstanding I/O at a time.
fn run_pread(fd: RawFd, args: &Args) {
    let buf = Aligned::new(BLOCK);
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let mut lats = Vec::with_capacity(args.reads);
    let started = Instant::now();
    for _ in 0..args.reads {
        let block = (rng.next() as usize) % args.blocks;
        let offset = (block * BLOCK) as libc::off_t;
        let t0 = Instant::now();
        let n = unsafe { libc::pread(fd, buf.ptr as *mut libc::c_void, BLOCK, offset) };
        assert_eq!(n, BLOCK as isize, "pread short/failed: {}", std::io::Error::last_os_error());
        lats.push(t0.elapsed().as_micros() as u64);
    }
    report("pread (serial, O_DIRECT)", started.elapsed(), args.reads, &mut lats);
}

#[cfg(feature = "io_uring")]
fn run_uring(fd: RawFd, args: &Args) {
    use io_uring::{opcode, types, IoUring};
    let depth = args.depth.max(1);
    let mut ring = IoUring::new(depth.next_power_of_two() as u32).unwrap();
    // One aligned buffer per in-flight slot.
    let mut bufs: Vec<Aligned> = (0..depth).map(|_| Aligned::new(BLOCK)).collect();
    let mut rng = Lcg(0x1234_5678_9abc_def0); // same seed as pread for a fair offset stream
    let mut lats = Vec::with_capacity(args.reads / depth + 1);
    let started = Instant::now();

    let mut issued = 0;
    while issued < args.reads {
        let batch = depth.min(args.reads - issued);
        let t0 = Instant::now();
        for (i, b) in bufs.iter_mut().enumerate().take(batch) {
            let block = (rng.next() as usize) % args.blocks;
            let offset = (block * BLOCK) as u64;
            let e = opcode::Read::new(types::Fd(fd), b.ptr, BLOCK as u32)
                .offset(offset)
                .build()
                .user_data(i as u64);
            unsafe { ring.submission().push(&e).expect("sq full") };
        }
        ring.submit_and_wait(batch).unwrap();
        let mut done = 0;
        for cqe in ring.completion() {
            assert_eq!(cqe.result(), BLOCK as i32, "uring read failed");
            done += 1;
        }
        assert_eq!(done, batch);
        lats.push(t0.elapsed().as_micros() as u64); // per-batch latency
        issued += batch;
    }
    report(&format!("io_uring (depth {depth}, O_DIRECT)"), started.elapsed(), args.reads, &mut lats);
}

fn report(name: &str, elapsed: std::time::Duration, reads: usize, lats: &mut [u64]) {
    lats.sort_unstable();
    println!("[{name}]");
    println!("  reads        : {reads}");
    println!("  throughput   : {:.0} reads/sec ({:.0} MiB/s)",
        reads as f64 / elapsed.as_secs_f64(),
        (reads * BLOCK) as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0));
    println!("  unit p50     : {} us", percentile(lats, 50.0));
    println!("  unit p99     : {} us", percentile(lats, 99.0));
}

fn main() {
    let args = parse();
    let path = std::env::temp_dir().join("diskstore-odirect-bench.dat");

    // 1) Lay down the data file with buffered writes, then flush + drop cache.
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        let chunk = vec![b'x'; BLOCK * 256];
        let mut written = 0;
        while written < args.blocks {
            let n = (args.blocks - written).min(256);
            f.write_all(&chunk[..n * BLOCK]).unwrap();
            written += n;
        }
        f.sync_all().unwrap();
    }
    let size_mib = args.blocks * BLOCK / (1024 * 1024);
    println!("data file: {} blocks x {BLOCK} B = {} MiB on ext4 (O_DIRECT)\n", args.blocks, size_mib);

    // 2) Open O_DIRECT for reads (bypasses the page cache entirely).
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(&path)
        .expect("O_DIRECT open (needs a non-tmpfs filesystem)");
    let fd = file.as_raw_fd();
    // Belt-and-suspenders: advise the kernel to drop any cached pages.
    unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };

    println!("random {}-block reads, {} total:\n", BLOCK, args.reads);
    run_pread(fd, &args);
    #[cfg(feature = "io_uring")]
    run_uring(fd, &args);
    #[cfg(not(feature = "io_uring"))]
    println!("\n(rebuild with --features io_uring to compare the io_uring backend)");

    let _ = std::fs::remove_file(&path);
}
