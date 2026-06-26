# Spark Local-GPU Lane — iteration, not registered results

The NVIDIA DGX Spark (Grace Blackwell GB10, aarch64, 128GB unified memory) is
wired as a **free, instant iteration lane** for GPU work — primarily RLVR/LoRA —
via [`tools/runpod_rlvr.py --local`](../../tools/runpod_rlvr.py) and the
[`spark-gpu.yml`](../../.github/workflows/spark-gpu.yml) workflow. It does **not**
replace GitHub Actions; it *augments* the GPU workflows so you can iterate without
renting a RunPod pod each time.

## The provenance boundary (load-bearing)

**The Spark is an iteration device, not a registered-results producer.** This is
not a limitation to work around — it is a deliberate boundary, and it exists for a
measurable reason:

- The Spark is **aarch64 / Grace Blackwell**, and the aarch64 wheel blockers
  (flash-attn, bitsandbytes, vLLM-colocate, unsloth — none ship aarch64 builds)
  force the `--quant bf16 --vllm none` path documented in
  [`config/inference.local.spark.json`](../../config/inference.local.spark.json).
- That path produces **different numerics** than the x86 RunPod A100 path
  (`--vllm colocate`, 4-bit or bf16 on a different arch).
- Therefore a Spark-produced training number is **not comparable** to a RunPod one
  and **must not be cited as the registered result** — per the existing
  REPLICATION.md discipline and the config's own `training` note.

**Source of record for registered numbers stays [`rlvr-runpod.yml`](../../.github/workflows/rlvr-runpod.yml) on x86 RunPod.** The Spark is for iterating fast and cheap; when a configuration is worth registering, run it through RunPod.

## Why the Spark is a *lane*, not a *replacement* for GitHub CI

- **17 of 18 workflows are CPU/lint/test** (Python compileall, the no-overclaim
  linter, pytest, Rust fmt/clippy). These finish in 2–15 min on free GitHub
  runners and gain nothing from Grace-Blackwell compute — but moving them to the
  Spark would make CI die whenever the Spark reboots, and would drop the
  `windows-latest` matrix leg. GitHub runners stay for all CPU work.
- **The 7 GPU workflows already externalize GPU work** to RunPod pods; the GitHub
  runner only orchestrates. The Spark replaces the *RunPod rental*, not the
  GitHub runner — and `--local` already implements exactly that.
- **Dispatch-only.** `spark-gpu.yml` never runs on push/PR/schedule because a
  personal server is not always up. You trigger it when you want a free iteration.

## Using the Spark lane

```bash
# Directly on the Spark (no GitHub involved):
python tools/runpod_rlvr.py --local --yes \
  --task provenance --remote-mode live \
  --quant bf16 --vllm none --seed 0

# Or via the workflow (dispatch from the Actions UI): spark-gpu.yml
# → artifacts land in agi-proof/benchmark-results/runpod-rlvr/local.*.json
```

Every Spark artifact is annotated `sparkIteration: true, registeredResult: false`
by the workflow so it cannot be mistaken for a registered number downstream.

## Registering the runner (one-time)

1. **Settings → Actions → Runners → New runner → self-hosted** on the repo.
2. On the Spark, run the generated `./config.sh` with labels:
   `--labels self-hosted,spark,aarch64`.
3. `sudo ./svc.sh install && sudo ./svc.sh start` to run as a service (survives reboot).
4. The `spark-gpu.yml` `runs-on: [self-hosted, spark, aarch64]` targets exactly these labels.

## Guardrails built into the workflow

- `--quant bf16 --vllm none` is **hard-coded** — no dispatch flag can override it,
  because the alternatives are un-runnable or un-registerable on aarch64.
- `concurrency` does **not** cancel in-flight runs — losing a 30-min GRPO job to a
  re-dispatch is wasteful on a single shared machine.
- A preflight step refuses to run if `nvidia-smi` fails (catches a mis-registered runner).
- `if: always()` on annotate + upload so a failed run still surfaces its logs.
