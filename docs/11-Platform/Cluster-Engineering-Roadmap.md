# Cluster Engineering Roadmap — from single-pod RunPod to a measured cluster model

> **Scope, stated plainly.** This is a *simulator and roadmap* for reasoning about
> AI-supercomputer scheduling and resilience trade-offs, in this repo's
> measured-not-claimed style. It is **not** a production scheduler and the numbers
> below are **simulated** with illustrative constants — not a measurement of any
> real fleet. The deliverable is honest machinery + reproducible trade-off curves +
> a clear map of what is built vs. still open.

## Why this exists

This repo already operates GPUs at *single-pod* scale: `tools/runpod_train.py` /
`runpod_rlvr.py` run the full pod lifecycle (create → poll SSH → rsync → run →
**always delete**), `skills/registry/runpod-gpu-orchestration.json` picks a run
mode (separate-pods / on-pod-parallel / on-pod-sequential) from seed count, VRAM,
quota and stockout, and `tools/estimate_runpod_eta.py` models wall-clock.

The gap from there to a 万卡→数十万卡 (10k→100k-GPU) cluster is the gap between
*renting one pod* and *scheduling a shared fleet*: topology-aware placement,
utilization-vs-latency policy, observability, and fault tolerance. The `cluster/`
package makes that gap **analyzable and measured** rather than hand-waved.

## Map: job responsibilities → repo assets

| JD responsibility (DeepSeek 超算集群研发工程师) | What this repo now has | Status |
|---|---|---|
| 异构算力调度：CPU/GPU/NPU 抽象、池化与**拓扑感知调度** | `cluster/topology.py` (heterogeneous device classes, NVLink islands, racks) + `cluster/scheduler.py` (`TopologyAware`, `BackfillTopo`, fragmentation metric) | **Built (sim)** |
| 调度算法在吞吐 / 排队延迟 / 利用率间取得均衡 | `cluster/simulator.py` + `tools/run_cluster_sim.py` → measured utilization vs. p99-wait vs. fragmentation curve | **Built (sim)** |
| 集群管理：任务/节点生命周期、**故障发现与自动容灾** | `cluster/faults.py` (node-failure injection + checkpoint/restart recovery, goodput accounting); single-pod lifecycle in `tools/runpod_train.py` | **Built (sim)** + real single-pod |
| 端到端性能调优：**性能抖动、长尾延迟、性能不均** | `cluster/observability.py` (`summarize` jitter/tail, `straggler_report` all-reduce step-slowdown) | **Built (analysis)** |
| 关键性能数据收集与可视化 | `*.public-report.json` artifacts + markdown tables from both CLI tools | **Built (reporting)** |
| RDMA / RoCEv2 / InfiniBand 拓扑、路由、多路径、拥塞控制 | Network-tax derived from ring-all-reduce cost per interconnect tier (`cluster/netcalib.py`); real NCCL bus-bandwidth benchmark harness (`tools/bench_nccl_allreduce.py` + `tools/runpod_nccl_bench.py`) wired and dry-run-verified | **Modeled + measurement harness ready** |
| 新一代集群架构、国产 AI 加速器、DPU / P4 | `heterogeneous_cluster()` device-class abstraction (e.g. `klass="domestic-x1"`); hardware study **open** | **Interface only** |

## What's built (this PR)

A pure-stdlib `cluster/` package (runs in CI, no deps), deterministic and seeded:

- **`topology.py`** — `Device / Node / Cluster`, NVLink-island & rack structure,
  `homogeneous_cluster()` and `heterogeneous_cluster()` (mixed accelerator classes).
- **`scheduler.py`** — three placement policies spanning the trade-off:
  `FifoFirstFit` (simple, scatters collectives), `TopologyAware` (best-fit packing
  into the fewest islands), `BackfillTopo` (+ EASY backfilling). `fragmentation()`
  scores placement locality (0 = one island, 1 = fully scattered).
- **`simulator.py`** — discrete-event replay with a **network-tax model**: a
  collective-heavy job placed across N islands runs slower (all-reduce that would
  stay on NVLink now crosses the NIC), so bad placement is physically punished in
  both runtime and utilization.
- **`observability.py`** — `summarize()` (mean/p50/p90/p99/cv jitter) and
  `straggler_report()` (the all-reduce *step-slowdown* a single long-pole rank imposes).
- **`faults.py`** — Poisson node-failure injection + checkpoint/restart recovery;
  separates raw busy time from **goodput** (work that survived) and quantifies the
  wasted-compute tax of an MTBF + checkpoint cadence.
- **`netcalib.py`** — derives the simulator's `island_tax` / `node_tax` from ring
  all-reduce cost per interconnect tier; loads/saves `cluster/netcalib.json`; tags each
  tier's provenance (measured vs modeled).
- **Tools** — `tools/run_cluster_sim.py`, `tools/run_cluster_faultsim.py`,
  `tools/calibrate_network_tax.py`, plus the measurement harness
  `tools/bench_nccl_allreduce.py` (on-pod torchrun) + `tools/runpod_nccl_bench.py`
  (RunPod lifecycle) → `agi-proof/benchmark-results/cluster/*.public-report.json`.

### Network-tax calibration (modeled → measured path)

The simulator's comm penalty is no longer a guessed constant. `cluster/netcalib.py`
derives `island_tax` / `node_tax` from **ring all-reduce cost per interconnect tier**
(`T = 2(N-1)/N · size/bw + 2(N-1)·lat`), using a worst-tier model: a collective runs at
the speed of its slowest hop (cross-node NIC ≫ cross-island NVSwitch ≫ intra-island
NVLink). `tools/calibrate_network_tax.py` writes `cluster/netcalib.json`, which the
simulator loads automatically.

Committed default is **MODELED** from documented line rates (NVLink 400 / NVSwitch 120 /
RoCE 50 GB/s, exposed-comm 0.15) → `island_tax≈0.345`, `node_tax≈1.036` (a cross-node
collective job runs ~2x its nominal time). To make it **MEASURED**:
`tools/runpod_nccl_bench.py` rents one multi-GPU SXM pod, runs
`tools/bench_nccl_allreduce.py` under torchrun, copies the bus-bandwidth report back, and
re-fits the NVLink tier. The harness is dry-run-verified and unit-tested; the live run is
currently **blocked** in this environment (no SSH egress / API key) — see
`agi-proof/benchmark-results/cluster/nccl-measure-blocker.public-report.json`. Cross-node
tiers stay MODELED (one pod can't measure them); provenance records the mix honestly.

### Measured trade-off (16×8 = 128 GPUs, 400 jobs, seed 7) — *simulated, calibrated tax*

| policy | utilization | jobs/hr | wait p50 (s) | wait p99 (s) | fragmentation | net tax |
|---|---|---|---|---|---|---|
| fifo-firstfit | 0.892 | 51.4 | 10203 | 21293 | 0.481 | 1.88 |
| topology-aware | 0.886 | 53.0 | 9781 | 20355 | 0.198 | 1.68 |
| backfill-topo | 0.873 | 54.5 | 7467 | 19472 | 0.117 | 1.52 |

Reading: topology-aware packing **cuts fragmentation 0.48→0.20 and the network tax
1.88→1.68x**; backfilling then **cuts p50 queue wait 10203→7467s** and lifts throughput,
at a small utilization cost. This is exactly the 吞吐 / 排队延迟 / 利用率 balance the
role optimizes — and the tax that drives it is now derived from interconnect bandwidth,
not assumed.

### Resilience sweep (128 GPUs, MTBF 500s) — *simulated*

Goodput is maximized by the **most frequent checkpoint interval** under this MTBF
(less work lost per failure); the sweep makes the wasted-compute vs. checkpoint-I/O
trade-off explicit. See `agi-proof/benchmark-results/cluster/faults.public-report.json`.

## What's still open (honest ledger)

- **RDMA tax is bandwidth-derived but the NVLink tier is not yet measured on hardware.**
  The harness to measure it (`tools/runpod_nccl_bench.py` → `bench_nccl_allreduce.py` →
  `calibrate_network_tax.py --from-nccl`) is built, dry-run-verified, and unit-tested; the
  live run is blocked here (no SSH egress / API key). *Next:* run it from a host with SSH
  egress to replace the modeled NVLink bandwidth with measured busbw. Cross-node RoCE/IB
  bandwidth still needs a real 2-pod fabric to measure.
- **No real telemetry pipeline.** `observability.py` analyzes series; it does not
  yet *collect* live GPU/step metrics off running pods. *Next:* wire a collector
  into `tools/runpod_train.py` and feed `straggler_report()` real per-rank step times.
- **Heterogeneous/NPU/DPU is interface-only.** Device classes exist; no
  domestic-accelerator or P4/DPU performance characterization. *Next:* a measured
  cross-architecture study.
- **Simulator constants are illustrative.** Network-tax and MTBF/checkpoint numbers
  are not calibrated to a real fleet; only the *shape* of the trade-off is claimed.

## How to reproduce

```bash
# (re)derive the network tax from interconnect bandwidth → cluster/netcalib.json
python tools/calibrate_network_tax.py --markdown

# scheduler + resilience simulations (consume the calibration automatically)
python tools/run_cluster_sim.py --nodes 16 --gpus-per-node 8 --islands 2 \
    --jobs 400 --seed 7 --markdown
python tools/run_cluster_faultsim.py --nodes 16 --gpus-per-node 8 \
    --jobs 200 --seed 7 --mtbf 500 --checkpoints 60,180,300,600,1200 --markdown

# measure real NCCL bandwidth on a GPU pod, then recalibrate (needs SSH + RUNPOD_API_KEY)
python tools/runpod_nccl_bench.py --dry-run --gpu-count 2          # inspect first
RUNPOD_API_KEY=... python tools/runpod_nccl_bench.py --yes --gpu-count 2 --calibrate

python tests/test_cluster_scheduler.py
python tests/test_cluster_observability.py
python tests/test_cluster_faults.py
python tests/test_cluster_netcalib.py
```
