# Local benchmarks runbook — `scripts/run_local_benchmarks.sh`

One-command runner for the two already-wired benchmarks on the **DGX Spark + Mac Studio**
cluster. It only **orchestrates existing tools** — it never invents results. It **dry-runs by
default** (prints the plan, executes nothing); pass `--execute` to actually run.

> **Cost-guard (wisdom-gpu-prebaked):** owned hardware (Spark + Mac) is **free**; **RunPod is the
> only metered path** and this script never touches it. After any GPU session, **confirm zero
> leaked pods**. Cheap validation first (`limit=24, runs=1`); watch the first ~6 min for restart loops.

---

## What it runs

- **Benchmark A** — ≥2-family VALIDATED judging of the M3-SFT source-discipline uplift:
  smoke test (`tools/run_local_judge_eval.py`) → per-seed judging (`tools/judge_pilot_answers.py`,
  ≥3 seeds, 2 families) → aggregate + no-overclaim gate (`tools/run_lora_uplift_validation.py`).
- **Benchmark B** — low-RAM NVFP4 certification (Boundary-3): GPU-free self-test
  (`tools/certify_lowram.py --selftest`) → optional QAT train (`tools/train_lora.py --qat
  --qat-scheme nvfp4`, only with `--run-train`) → certify (`tools/certify_lowram.py --scheme nvfp4`).

---

## Prerequisites (start these BY HAND — the script does not)

**Benchmark A — two-box judge farm** (see `docs/11-Platform/Mac-Spark-Judge-Farm.md`,
`config/inference.local.mac-judge.json`):

```bash
# On the DGX Spark (CUDA, vLLM)  -> judge family 'qwen':
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
# On the Mac Studio (Apple Silicon, MLX) -> judge family 'mlx':
mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8080
```

Two distinct families (`qwen` + `mlx`); subject lineage is `allenai/olmoe`, so **judge != subject**.
Then set `SPARK_HOST` / `MAC_HOST` (and ports if non-default) when invoking the script.

You must also have **per-seed answer files** (`base_answer` + `adapter_answer` per case), one JSON
per seed (`judge_pilot_answers.py` has **no `--seed` flag** — seeds are separate files). Reproduce
them via the `wisdom-pilot-runpod` workflow if missing. For step A3 you must **assemble one
`judgments.json`** (`subjectModel`, `judges[]`, `seeds[].items[]` with per-family
`baseContent`/`adapterContent`) from the per-seed judge outputs — that is the upstream labelling
step the P6 preregistration specifies.

**Benchmark B — QAT/cert** (run on the Spark `.venv`): an existing PEFT adapter at
`training/lora/checkpoints/olmoe-qat-spark` (or pass `--run-train` to produce one), and calibration
rows at `training/lora/train.jsonl`.

---

## How to invoke

```bash
# Dry-run (default — prints the plan, executes nothing):
SPARK_HOST=… MAC_HOST=… scripts/run_local_benchmarks.sh --all

# Benchmark A only / B only:
SPARK_HOST=… MAC_HOST=… scripts/run_local_benchmarks.sh --bench-a --execute
scripts/run_local_benchmarks.sh --bench-b --execute

# Both, for real, INCLUDING the long QAT train:
SPARK_HOST=… MAC_HOST=… scripts/run_local_benchmarks.sh --all --execute --run-train
```

Flags: `--bench-a`, `--bench-b`, `--all` (default if none given), `--execute` (run for real),
`--run-train` (also run the long GPU QAT train in B), `-h/--help`.

Key env overrides: `SPARK_HOST MAC_HOST SPARK_PORT MAC_PORT JUDGES JUDGE_CONFIG SEEDS ANSWERS_DIR
ANSWERS_PREFIX JUDGE_OUT_DIR JUDGMENTS UPLIFT_OUT QAT_BASE QAT_ADAPTER QAT_DATA QAT_EPOCHS
QAT_LAMBDA CERT_CALIB CERT_OUT CERT_NEVAL PYTHON`.

---

## Pass bars (do not relax)

**Benchmark A — no-overclaim VALIDATED gate** (`tools/run_lora_uplift_validation.aggregate`;
`config/inference.local.mac-judge.json` `gate{}`; `RESULTS.md`). VALIDATED iff **all** hold:

- non-mock subject
- **≥2 distinct judge families** (`qwen` + `mlx`)
- **judge != subject** (subject lineage `allenai/olmoe` — clear of `qwen` & `meta-llama`)
- **mean pairwise Cohen's κ ≥ 0.40**
- **≥3 seeds**
- **95% bootstrap CI on the content-uplift delta excludes zero**

> Honest state (2026-06-29): κ measured **0.24 < 0.40** → gate **UNMET** → **CANDIDATE**, not
> VALIDATED. The win-rate panel (both families significant) is the honest headline; do not force
> a VALIDATED verdict.

**Benchmark B — low-RAM NVFP4 cert** (`serving/lowram_eval.LowRamGate` /
`certify_lowram.DEFAULT_CONTRACT`):

- overall: **mean KL ≤ 0.05** AND **top-1 agreement ≥ 0.97**
- protected slice: **KL ≤ 0.10** AND **agreement ≥ 0.95**

> Honest state (2026-06-29): best run (v3) mean_kl 0.045 (passes ≤0.05) but top1 0.906 (< 0.97)
> → **NO-GO** on the strict gate. On PASS you may claim **only**: "served-quant retains BF16
> next-token behavior to a measured bound" (`docs/11-Platform/Cheap-Compute-Boundary.md`).

---

## Artifacts written (per default paths)

| Step | Tool | Artifact |
|---|---|---|
| A1 | `run_local_judge_eval.py --config` | stdout only (print-only smoke test, CI-safe) |
| A2 | `judge_pilot_answers.py` | `…/m3-2family-judge/seed<N>-judge.json` (one per seed) |
| A3 | `run_lora_uplift_validation.py` | `…/m3-2family-judge/uplift-validation.json` (`validated` true/false) |
| B0 | `certify_lowram.py --selftest` | stdout only (GPU-free invariants) |
| B1 | `train_lora.py --qat` | PEFT adapter at `training/lora/checkpoints/olmoe-qat-spark/` |
| B2 | `certify_lowram.py` | `…/olmoe-qat-spark/lowram_report.json` (LowRamReport; exit 0 PASS / 2 FAIL) |

On a VALIDATED Benchmark-A result, promote the row in
`agi-proof/benchmark-results/published-results.json` and regenerate `RESULTS.md` via
`tools/build_results_page.py`. Otherwise keep it labelled **candidate**.
