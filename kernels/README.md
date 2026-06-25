# `kernels/` — HPC operator track (M1 skeleton)

A sibling portfolio track to Sophia's trust layer: high-performance GPU kernels measured
against the **hardware's physical limit**, not against a strawman baseline. See the
roadmap: [`docs/06-Roadmap/HPC-Operator-Compiler-Roadmap.md`](../docs/06-Roadmap/HPC-Operator-Compiler-Roadmap.md).

> **Honest status (M1).** The *gate* and the *first kernel* exist: a Triton tiled BF16 GEMM
> (`src/run_kernel.py`) that checks correctness vs `torch.matmul` and prints its own
> roofline block. It is a straightforward tiled GEMM (one fixed block config, no warp
> specialization / split-K / autotuned epilogue) — so its % of roofline is the M1 number
> to report and then close, not a tuned result. Real timing needs a CUDA GPU; on CPU/CI it
> skips cleanly.

## Layout

| Path | What |
|---|---|
| `bench/roofline.py` | Roofline harness — % of theoretical peak, regime, ridge point, over-100% guard. Offline; no GPU needed for the math. |
| `src/run_kernel.py` | Triton tiled BF16 GEMM (FP32 accumulate). Correctness-checked, self-rooflines. The RunPod orchestrator `ncu`-profiles it. |
| `reports/` | Profiling artifacts copied back from the pod (git-ignored). |
| `../tools/runpod_kernels.py` | Build + profile on a rented CUDA pod; `--dry-run` by default (no pod, no cost). |

## Use it now (offline)

```bash
python kernels/bench/roofline.py --self-test          # exercise the math
python kernels/bench/roofline.py --demo --device "NVIDIA H100 80GB HBM3"
python tools/runpod_kernels.py --dry-run              # print the remote script; no cost
```

## On a real GPU

Two ways, both dispatch-only and both with a confirm gate:

**A. GitHub Actions (no local secret needed).** Set the repo secret `RUNPOD_API_KEY` once
(Settings → Secrets and variables → Actions), then run the **`kernels-runpod`** workflow
(Actions tab → Run workflow), type `RUN` to confirm, pick the GPU and GEMM size. It
dry-runs the request, rents the pod, runs the roofline self-test + timed kernel + `ncu`
profile over SSH, uploads `kernels/reports/**` as a run artifact, and **always deletes the
pod**.

**B. Locally.**

```bash
RUNPOD_API_KEY=... python tools/runpod_kernels.py --yes --gpu-type "NVIDIA H100 80GB HBM3"
```

Both rent one pod, run over SSH, copy reports back, and **delete the pod even on failure**
(lifecycle reused from `tools/runpod_rlvr.py`). The pod also self-destructs via a remote
watchdog if the orchestrator dies.

## The reporting rule

Every kernel reports **% of roofline** (achieved / `min(compute_peak, intensity·bw_peak)`),
with ≥3 runs and dispersion — the same no-overclaim discipline as `RESULTS.md`. No
"Nx vs naive" headline. Anything above ~95% of roofline is treated as a FLOP/byte
accounting bug until proven otherwise.

## Next

1. Run `tools/runpod_kernels.py --yes` on an H100/A100 to get the **first measured % of
   roofline** for the tiled GEMM, with an `ncu` SM/memory-utilization report.
2. Close the gap toward the ceiling: autotune block configs, then warp specialization /
   split-K / a fused epilogue — re-reporting % of roofline at each step.
3. Add a FlashAttention-style fused kernel (M1 stretch) under `src/`.
