# Sophia storage benchmarks — RunPod (real NVMe)

generated: 2026-06-26T06:10:48Z

## Host
```
kernel: 6.8.0-59-generic
CPU(s):                               128
Model name:                           AMD EPYC 75F3 32-Core Processor
Thread(s) per core:                   2
Core(s) per socket:                   32
memory: MemTotal:       528168748 kB
fs: overlayfs
NAME    ROTA  SIZE MODEL
nvme2n1    0    7T SAMSUNG MZQL27T6HBLA-00B7C              
nvme1n1    0    7T SAMSUNG MZQL27T6HBLA-00B7C              
nvme0n1    0  1.7T SAMSUNG MZ1L21T9HCLS-00A07              
(ROTA=0 => SSD/NVMe)
```

## diskstore — O_DIRECT cold I/O (pread vs io_uring) — real NVMe
```
+ ./target/release/diskstore-odirect-bench --blocks 500000 --reads 200000 --depth 128
data file: 500000 blocks x 4096 B = 1953 MiB on ext4 (O_DIRECT)

random 4096-block reads, 200000 total:

[pread (serial, O_DIRECT)]
  reads        : 200000
  throughput   : 12861 reads/sec (50 MiB/s)
  unit p50     : 74 us
  unit p99     : 98 us

[io_uring] UNAVAILABLE on this host: Operation not permitted (os error 1)
  io_uring_enter failing with EPERM means the container seccomp profile
  blocks io_uring (common in hardened runtimes). The pread numbers above
  still stand; run on a host that permits io_uring to compare.
```

## diskstore — page-cached batched reads
```
+ ./target/release/diskstore-bench --keys 300000 --value-size 512 --batch 256 --batches 3000
loading 300000 keys x 512 bytes ... done (154 MiB on disk)
batched random reads: 3000 batches x 256 keys

[std] (std(pread))
  reads          : 768000
  throughput     : 635690 reads/sec
  batch p50      : 392 us
  batch p99      : 501 us
Error: Os { code: 1, kind: PermissionDenied, message: "Operation not permitted" }
```

## kvcache — no pipelining
```
+ ./target/release/kvcache-bench --clients 64 --ops 50000 --pipeline 1
kvcache-bench: 64 clients x 50000 ops, 100000 keys, 256-byte values, 16 shards, write_frac=0
--- results ---
total ops      : 3200000
wall time      : 15.163 s
throughput     : 211037 ops/sec
latency p50    : 279 us (per-op)
latency p99    : 680 us (per-op)
latency p99.9  : 846 us (per-op)
latency max    : 25259 us (per-op)
server stats   : hits=3200000 misses=0 sets=100000 evictions=0 entries=100000
```

## kvcache — pipeline depth 16
```
+ ./target/release/kvcache-bench --clients 64 --ops 50000 --pipeline 16
kvcache-bench: 64 clients x 50000 ops, 100000 keys, 256-byte values, 16 shards, write_frac=0
--- results ---
total ops      : 3200000
wall time      : 1.346 s
throughput     : 2377745 ops/sec
latency p50    : 407 us (per-batch (depth=16))
latency p99    : 793 us (per-batch (depth=16))
latency p99.9  : 1057 us (per-batch (depth=16))
latency max    : 10549 us (per-batch (depth=16))
server stats   : hits=3200000 misses=0 sets=100000 evictions=0 entries=100000
```

## infcache — prefix-cache token reuse
```
+ ./target/release/infcache-bench --requests 2000 --system 4096 --suffix 128
infcache-bench: 2000 requests, system=4096 tok, suffix=128 tok, block=16, kv/block=4096 B
--- results ---
prompt tokens (total) : 8448000
prompt tokens reused  : 8187904
token reuse rate      : 96.9%
block hit rate        : 99.6%
tiers                 : l1_hits=511744 l2_hits=0 misses=2000 promotions=0 stores=528000
wall time             : 6.868 s (291 req/s)
```

