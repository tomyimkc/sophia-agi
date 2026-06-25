# `kernels/` — HPC operator track (M1 skeleton)

A sibling portfolio track to Sophia's trust layer: high-performance GPU kernels measured
against the **hardware's physical limit**, not against a strawman baseline. See the
roadmap: [`docs/06-Roadmap/HPC-Operator-Compiler-Roadmap.md`](../docs/06-Roadmap/HPC-Operator-Compiler-Roadmap.md).

> **Honest status (M1 skeleton).** The *gate* exists; the kernels do not yet. The point of
> landing the harness first is that the first real kernel is born already measured.

## Layout

| Path | What |
|---|---|
| `bench/roofline.py` | Roofline harness — % of theoretical peak, regime, ridge point, over-100% guard. Offline; no GPU needed for the math. |
| `src/` | Kernels go here. Drop a `run_kernel.py` and the RunPod orchestrator will `ncu`-profile it. *(empty in M1)* |
| `reports/` | Profiling artifacts copied back from the pod (git-ignored). |
| `../tools/runpod_kernels.py` | Build + profile on a rented CUDA pod; `--dry-run` by default (no pod, no cost). |

## Use it now (offline)

```bash
python kernels/bench/roofline.py --self-test          # exercise the math
python kernels/bench/roofline.py --demo --device "NVIDIA H100 80GB HBM3"
python tools/runpod_kernels.py --dry-run              # print the remote script; no cost
```

## On a real GPU

```bash
RUNPOD_API_KEY=... python tools/runpod_kernels.py --yes --gpu-type "NVIDIA H100 80GB HBM3"
```

Always rents one pod, runs over SSH, copies reports back, and **deletes the pod even on
failure** (lifecycle reused from `tools/runpod_rlvr.py`).

## The reporting rule

Every kernel reports **% of roofline** (achieved / `min(compute_peak, intensity·bw_peak)`),
with ≥3 runs and dispersion — the same no-overclaim discipline as `RESULTS.md`. No
"Nx vs naive" headline. Anything above ~95% of roofline is treated as a FLOP/byte
accounting bug until proven otherwise.

## Next (M1 kernel)

Add `kernels/src/run_kernel.py` with a Triton tiled BF16 GEMM that prints its own
`roofline.analyze(...)` block, then profile it via `tools/runpod_kernels.py --yes`.
