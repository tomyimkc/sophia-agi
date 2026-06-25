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
| RDMA / RoCEv2 / InfiniBand 拓扑、路由、多路径、拥塞控制 | Network *tax* model in the simulator (cross-island/cross-node penalty); real RDMA study **open** | **Modeled, not measured** |
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
- **Tools** — `tools/run_cluster_sim.py`, `tools/run_cluster_faultsim.py` →
  `agi-proof/benchmark-results/cluster/*.public-report.json`.

### Measured trade-off (16×8 = 128 GPUs, 400 jobs, seed 7) — *simulated*

| policy | utilization | jobs/hr | wait p50 (s) | wait p99 (s) | fragmentation | net tax |
|---|---|---|---|---|---|---|
| fifo-firstfit | 0.856 | 51.9 | 9474 | 20225 | 0.433 | 1.53 |
| topology-aware | 0.863 | 61.4 | 7868 | 16634 | 0.162 | 1.31 |
| backfill-topo | 0.851 | 64.0 | 7064 | 16071 | 0.111 | 1.23 |

Reading: topology-aware packing **cuts fragmentation 0.43→0.16 and the network tax
1.53→1.31x**, lifting throughput; backfilling then **cuts p50 queue wait 9474→7064s**
and lifts throughput further, at a small utilization cost. This is exactly the
吞吐 / 排队延迟 / 利用率 balance the role optimizes.

### Resilience sweep (128 GPUs, MTBF 500s) — *simulated*

Goodput is maximized by the **most frequent checkpoint interval** under this MTBF
(less work lost per failure); the sweep makes the wasted-compute vs. checkpoint-I/O
trade-off explicit. See `agi-proof/benchmark-results/cluster/faults.public-report.json`.

## What's still open (honest ledger)

- **RDMA is modeled, not measured.** The network tax is a coefficient, not a
  RoCEv2/InfiniBand measurement. *Next:* benchmark real NCCL all-reduce across
  multi-GPU RunPod pods and calibrate the tax from observed bandwidth/latency.
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
python tools/run_cluster_sim.py --nodes 16 --gpus-per-node 8 --islands 2 \
    --jobs 400 --seed 7 --markdown
python tools/run_cluster_faultsim.py --nodes 16 --gpus-per-node 8 \
    --jobs 200 --seed 7 --mtbf 500 --checkpoints 60,180,300,600,1200 --markdown
python tests/test_cluster_scheduler.py
python tests/test_cluster_observability.py
python tests/test_cluster_faults.py
```
