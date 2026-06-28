# GSS on the DGX Spark — pull-and-run runbook

Continue the Governed Speculative Sparsity work on the GB10. The Spark is the right box for
the one number we couldn't get on RunPod: the **on-silicon NVFP4 roofline** (the fp4 peak
exists only on Blackwell, and `kernels/bench/roofline.py` already carries the
`NVIDIA DGX Spark GB10` profile — 128 GB LPDDR5x unified, **273 GB/s**, FP4 ~500 TFLOP/s).

## 1. Pull the branch

```bash
git fetch origin claude/deepspec-theory-applicability-bmm7e2
git checkout claude/deepspec-theory-applicability-bmm7e2
git pull --ff-only origin claude/deepspec-theory-applicability-bmm7e2
```

## 2. One command

```bash
bash scripts/gss_spark_run.sh
```

It runs, in order: the offline invariants (Tier 0 + Tier 1), the CI suites, and the
**Tier-2 gather-GEMM roofline A/B** — dense (ρ=1.0) vs read-set (ρ=0.10) — each reported as
**% of the Spark's 273 GB/s**. Override the shape/read-set via env:

```bash
M=1 N=8192 K=8192 TILE=256 RHO=0.10 ITERS=50 bash scripts/gss_spark_run.sh
```

### What to read off it
- Each kernel run prints `bandwidth (% of 273 GB/s)` and a `traffic ratio` line. The
  **gather run should move ~ρ× the bytes of dense** (ρ=0.10 → ~0.11×) at `rel_err < 5e-2`
  vs the NumPy reference — the bandwidth lever in real bytes, certified correct.
- For a registered Tier-2 number, take **≥3 runs + dispersion** (raise `ITERS`, or re-run);
  report the % of roofline, never an "Nx vs a strawman".

## 3. Spark / aarch64 notes

- **Triton** is needed for the fused kernel. If `import triton` fails, the kernel prints a
  clean skip and only the NumPy reference runs — install a Blackwell-compatible Triton to get
  the real roofline.
- **Device name:** the script passes `--device "NVIDIA DGX Spark GB10"` explicitly so the
  roofline divides by the right 273 GB/s even if `torch.cuda.get_device_name()` returns a
  shorter string.
- **Real-checkpoint probe on the Spark:** use `--draft fakequant`, **not** `bnb` —
  bitsandbytes is x86/CUDA-fragile on aarch64. The 128 GB unified memory means you can hold a
  much larger MoE here than on a 24–48 GB RunPod card:
  ```bash
  python tools/gss_probe.py --backend hf --model <a-big-MoE> \
      --draft fakequant --campaign 5 --out agi-proof/benchmark-results/gss-spark.json
  ```

## 4. Where this sits

| Tier | Artifact | Status |
|---|---|---|
| 0 | `serving/gss_feasibility.py` | ✅ + registered GPU CI result (cost_ratio 0.253 [0.241,0.265]) |
| 1 | `serving/gss.py` | ✅ lossless proof (1e-12) + equivalence gate |
| 2 | `kernels/src/gss_gather_gemm.py` | ✅ CPU-validated — **this runbook gets its Spark roofline** |
| 4 | `moe/adapt.py` online bit-depth loop | open (acceptance-rate meter → live precision control) |

Honest scope unchanged: Tier 2 measures the **bandwidth** GSS saves (aggressive pruned-verify
lever); a *lossless* decode still pairs the gather with the dense-verify accept/reject in
`serving/gss.py` (~2× guaranteed-lossless vs ~3.9× aggressive ceiling). `canClaimAGI` stays
`false`.
