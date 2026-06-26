# Distributed Training — FSDP/ZeRO Path (P3 — design)

> Status: design only. No training code in this pass. This is a months-long
> lift; this doc fixes the verified gaps, the seams, the modeling, and the
> build order so the work is de-risked before any trainer code is written.

## Why design-only now

The advisor's point ⑤: *"Software stack gap is larger than hardware gap. You
will need to stand up proper distributed training (FSDP, DeepSpeed ZeRO, or
Megatron-style) … integrate it with your gate, add checkpoint/restart that
respects R1–R4, and make benchmarking fully automated and local."*

This is correct and is the **only** advisor point that is genuinely net-new
engineering (points ① and ④ are mostly built; ② and ③ are data/protocol). But it
is also the largest lift and the easiest place to waste months. Per the agreed
sequencing: **P0 (capability delta) is implemented first; P3 starts only after P0
proves a delta worth scaling.** This doc is the design that waiting period
produces.

## The verified gap (grounded in the code, not assumed)

The codebase has **zero** distributed training today. Every supporting primitive
exists; the trainer is the missing piece.

### What exists (the head start)
- **Multi-GPU pods are rentable.** `tools/runpod_nccl_bench.py:176-177` rents
  `gpuCount>=2` SXM pods (default 2, A100-SXM4/H100/H100-SXM, `:55-59`) and
  rejects `<2`.
- **torchrun + NCCL bootstrapping is proven in-repo.**
  `tools/bench_nccl_allreduce.py:34-39` builds
  `torchrun --standalone --nproc_per_node=N …`; `:47-54` does
  `dist.init_process_group(backend="nccl")` and reads `RANK`/`WORLD_SIZE`/
  `LOCAL_RANK` from env. busbw = algbw·2(N−1)/N is reported per message size
  (`:74-83`).
- **The cross-node network cost is already modeled.** `clustersim/netcalib.py`
  derives `island_tax`/`node_tax` from ring-allreduce cost per tier
  (`netcalib.py:20-21`); committed modeled defaults
  (`clustersim/netcalib.json:8-11`): `island_tax=0.345`, `node_tax=1.036`,
  `comm_fraction=0.15`. `from_nccl_report` (`netcalib.py:157-183`) replaces the
  nvlink tier with a *measured* busbw.
- **Checkpoint/restart trade-offs are formalized.** `clustersim/faults.py`
  commits progress only at `checkpoint_s` boundaries (`_commit_progress`,
  `faults.py:184-197`), a failure loses the uncommitted tail, requeues after
  `recovery_s`, and `committed_s` carries durable progress across restarts
  (`:196`). `simulate_with_faults` (`:101-258`, defaults `checkpoint_s=300`,
  `recovery_s=60`) reports goodput vs wasted fraction.

### What is missing (the gap)
- **No distributed trainer.** `tools/train_lora.py` is a *manual single-process*
  loop (`train_lora.py:4-7`, `run_manual_train` at `:533-668`), loads with
  `device_map="auto"` (`:279`), no FSDP/DDP/device_mesh/ZeRO anywhere. Grep for
  those returns nothing.
- **GRPO hard-defaults to single-GPU.** `tools/run_rlvr.py:200-204` sets
  `RANK=0`/`LOCAL_RANK=0`/`WORLD_SIZE=1`/`MASTER_ADDR=127.0.0.1` via
  `setdefault` (never overriding a real launch) — these exist only because
  vLLM's colocate executor reads them (`run_rlvr.py:194-199`). The documented
  GPU usage is single-GPU (`:27-30`).
- **The adapter checkpoint is single-rank.** `save_adapter`
  (`train_lora.py:671-680`) writes a standard HF/PEFT adapter dir via
  `model.save_pretrained`. `--resume-adapter` (`:696-701`, `:299-301`) does
  `PeftModel.from_pretrained(..., is_trainable=True)` — "continue an existing
  adapter," **not** optimizer/step/RNG-state resume. No sharded checkpoint, no
  `resume_from_checkpoint`.
- **Launchers are single-pod.** Every RunPod launcher creates one pod, streams a
  script, scps artifacts back, deletes in `finally`. There is no multi-pod
  launch + rendezvous (`MASTER_ADDR`/`MASTER_PORT` across pods).

## The minimal-viable path (in build order)

### Build 1 — Intra-node multi-GPU first (torchrun + FSDP, one pod)
The cheapest step, and the one the existing primitives already support:
- Swap the `accelerate launch --num_processes 1` expectation
  (`run_rlvr.py:194-199`) for `torchrun --nproc_per_node=N` on a ≥2-GPU SXM pod
  (already rentable via `runpod_nccl_bench.py`'s path).
- Add an FSDP wrap to the GRPO trainer (`run_rlvr.py`) and the SFT trainer
  (`train_lora.py`). FSDP over a single node needs no code change to the reward
  (`gate_reward.make_grpo_reward` is rank-agnostic) or the gate.
- **Measure first**: run `bench_nccl_allreduce.py` on the target pod, feed the
  report through `netcalib.from_nccl_report` to get the *measured* (not modeled)
  intra-node tax before deciding FSDP vs ZeRO-3.

### Build 2 — Sharded checkpoint + real resume
- Replace the single-rank `save_adapter` with sharded optimizer + model + RNG
  state (FSDP `FULL_STATE_DICT` or sharded checkpoints), so a restart resumes
  from a step, not from scratch.
- Wire the cadence through the `clustersim/faults.py` model: pick `checkpoint_s`
  from the pod's MTBF so `wasted_fraction` stays bounded *before* running real
  jobs. The simulator already reasons about this trade-off; use it to choose the
  cadence rather than guessing.

### Build 3 — Multi-node (the real lift)
- Multi-pod launch + rendezvous: a new launcher that creates N pods, exchanges
  `MASTER_ADDR`/`MASTER_PORT` across them, and joins all ranks into one process
  group. This is the piece with **no existing precedent** in the repo.
- **Model the cross-node tax first.** `netcalib.node_tax=1.036` means a
  cross-node collective runs ~2× nominal — so multi-node only pays if the model
  does not fit on one node. Use `clustersim/simulator.py` (`:51-69`,
  `effective_runtime`) to decide whether a given model/size is worth spanning
  nodes before renting the second node.

### Build 4 — R1–R4 integration
R1–R4 (`docs/06-Roadmap/AI-Cluster-Reliability-Engineering.md`) are **cluster
ops** reliability tiers, not training internals:
- R1 (MTTR/fault-localization, `:43-55`) + R3 (observability, `:71-84`) surface
  node failures (XID/ECC via `agent/cluster/ssh_provider.py`) during a run.
- R4 (gated self-heal, `:86-96`, `agent/cluster/heal.py`) routes remediation
  through `agent/guarded.py`.
A distributed run surfaces its failures to R1/R3 and its recovery to R4; it does
not reimplement them.

## The provenance boundary (unchanged)

Any new GPU path inherits the discipline the DGX Spark lane established
(`docs/11-Platform/Spark-Local-GPU-Lane.md`, `.github/workflows/spark-gpu.yml`):
different numerics → `candidateOnly`/`registeredResult: false` until the
no-overclaim gate passes; x86 RunPod stays the source of record for registered
numbers. A distributed run is no different — its artifacts are
candidate-only until ≥3 seeds + bootstrap CIs + the promotion gate clear.

## Why this is sequenced last

Distributed training is **pure plumbing** until P0 shows the model+gate stack
produces a measurable capability delta worth scaling. Standing up FSDP before
that proof is the classic trap: months of infra for a model that has not yet
earned the scale. Once P0's panel shows D ≳ B (P2) and the fidelity flywheel
(P1) is feeding it, *then* the scale is justified — and this doc is the plan
that's ready to execute.

## Non-goals (this pass)

- No trainer code. No pod changes. No checkpoint-format change.
- Does not touch the reward, the gate, the panel, or the data pipeline.
- clustersim modeling experiments (sizing multi-node before renting) are a
  follow-up, not part of this doc.

## Verification target (when built)

Before any real multi-node job: (1) a measured NCCL report fed through
`netcalib.from_nccl_report`; (2) a `clustersim` run sizing the model against the
node/island tax to confirm multi-node is worth it; (3) a single-node FSDP run
that reproduces the single-GPU capability panel (P0) within seed noise — proving
the distributed path does not change the result before it is trusted at scale.
