# Spark-MoE Training & Serve Workflow (highest-intelligence-per-watt on a DGX Spark)

> Status: **plan + Phase-1 tooling**. Phases tagged by *where they run*:
> `[LOCAL]` = code/config, offline-testable now · `[CLOUD-PAID]` = rented x86 RunPod GPU
> (real money; manual dispatch) · `[SPARK-HW]` = needs the DGX Spark hardware ·
> `[MEASURE]` = no-overclaim-gated evaluation.

## Goal (reconciled with VISION)

Produce a model with **the most parameters that still runs smoothly** on the Spark **and
the highest intelligence** — *without* violating `VISION.md` ("don't out-train frontier
labs; innovate at the trust layer"; "no AGI claims").

The reconciliation: **a sparse Mixture-of-Experts (MoE) + the Sophia scaffold.**

- **Most parameters + smooth:** MoE gives a high total parameter count with low *active*
  compute per token, so an 80–106B model only reads ~3–12B of weights/token. That is the
  only architecture that survives the Spark's real bottleneck — **273 GB/s LPDDR bandwidth**
  (decode is bandwidth-bound; a dense 70B would crawl).
- **Highest intelligence:** per VISION, intelligence comes from the **verifier/council/
  provenance scaffold** around the model, not raw scale. A 3B-active MoE inside the full
  Sophia stack can outperform a much larger dense model on the verifiability axes that are
  the repo's success criteria.

## Target architecture

| Layer | Choice | Rationale |
|---|---|---|
| Student substrate | **Sparse MoE ~80–106B / 3–12B active** (e.g. Qwen3-Next-80B-A3B ~45 tok/s; GLM 4.6V 106B/12B quality pick) | Max params that decode smoothly on 273 GB/s |
| Intelligence source | **Distill a frontier teacher** (`model.py` presets: anthropic / openai / glm) → **RLVR-ground** | MOPD = the 2026 frontier recipe; borrows, doesn't out-train |
| Intelligence multiplier | **Full Sophia scaffold** (council, best-of-N, self-verify, conscience gate, verifier gates, RAG, graded abstention) | Where Sophia actually innovates |
| Serving | **vLLM on GB10**, quantized to ≤120 GB | `config/inference.local.spark.json` tiers |

## Phased workflow

### Phase 0 — Plan `[LOCAL]` ✅
This document.

### Phase 1 — Teacher-distillation corpus `[LOCAL]`
Generate verified reasoning trajectories from a frontier teacher over the three
machine-checkable domains (provenance / math / code), in the repo's `training/examples`
format. Offline-safe (`--teacher mock`); metered with a real spec.
```bash
python tools/build_distillation_corpus.py --teacher mock --domain all --n 4 --dry-run   # inspect
python tools/build_distillation_corpus.py --teacher glm:glm-5.2 --domain provenance --n 50
python tools/validate_attribution.py        # gate before merge (CONTRIBUTING.md "Phase 2")
```
Every trajectory is reviewable + provenance-gated. (Teacher-data tooling: `tools/claude_teacher.py`,
correction loop.)

### Phase 2 — Distill into the MoE student `[CLOUD-PAID]`
SFT/LoRA the sparse-MoE student on the Phase-1 corpus, then **on-policy distillation**
(sample student, teacher grades per-token — dense signal). Retarget the existing
`runpod-sophia-7b-sft.yml` / `tools/runpod_train.py` pattern to an MoE base.
```bash
gh workflow run runpod-sophia-7b-sft.yml -f confirm=RUN -f seed=0   # adapt to MoE base + distill corpus
```
> aarch64 wheel note: distillation *training* runs on x86 RunPod (flash-attn/bnb/vLLM-colocate/
> unsloth aarch64 wheels are not viable for the heavy trainer). Keep training on x86.

### Phase 3 — RLVR grounding `[CLOUD-PAID]` (the repo's core strand)
GRPO with **verifiable rewards** over provenance_faithful / math_equivalent / code-hidden-tests.
**Headline numbers stay on x86** (`REPLICATION.md`).
```bash
gh workflow run rlvr-runpod.yml -f confirm=RUN -f remote_mode=live -f task=provenance -f seed=0   # ×{0,1,2}
python tools/aggregate_rlvr_runs.py
```

### Phase 4 — Serve smoothly on the Spark `[SPARK-HW]`
Serve the trained MoE via vLLM on the GB10; point Sophia at it via `config/inference.local.spark.json`.
**Verify the aarch64+sm_100 vLLM MoE path runs your specific model+quant** (make-or-break).
Avoid NVFP4 (currently broken on GB10); use a working FP8/INT4 path. Target ≥20 tok/s decode.
```bash
vllm serve <org/<MoE-model>> --port 8000 --quantization <fp8|int4>     # on the Spark
export SOPHIA_MODEL_PROVIDER=vllm SOPHIA_MODEL_BASE_URL=http://localhost:8000/v1
python tools/run_local_judge_eval.py --provider vllm --base-url http://localhost:8000/v1   # smoke-test
```

### Phase 5 — Wrap in the scaffold `[LOCAL]` (already wired)
Every query routes through `council_deliberate` + `best_of` + self-verification + the
**conscience gate** (in `run_agent`) + verifier gates + RAG/provenance + graded abstention.
The verifier-synthesis flywheel + `governed_rsi` offline runner sharpen it continuously.

### Phase 6 — Iterate on Spark, run the eval farm `[SPARK-HW]`
Fast local LoRA iteration + always-on eval/judge farm (replaces the largest metered line item).
```bash
python tools/runpod_rlvr.py --local --quant bf16 --vllm none --task provenance   # local iteration, no pod/cost
python tools/run_local_judge_eval.py --provider vllm --judge-models <Qwen>,<Llama>   # 2-family local judges
```

### Phase 7 — Measure under the no-overclaim gate `[MEASURE]`
Any "intelligence" claim clears ≥2 judge families, ≥3 runs, κ≥0.40, CIs exclude chance → `RESULTS.md`.
```bash
python tools/build_results_page.py --check
```
If one GB10's 128 GB is tight, **cluster two Sparks via ConnectX (200G)** for bigger models / 128K context.

## Partition principle (must-not-cross)
- **Training (Phases 2–3) on x86 RunPod.** Spark = serve + iterate (LoRA) + eval/judge appliance.
- **Headline numbers stay on the platform they were registered on** (`REPLICATION.md`,
  `preregistered-thresholds.md`). Never cite a Spark-arch training number as the registered result.
- **No AGI/ASI claims** (VISION explicit non-goal). Intelligence is *measured*, not asserted.

## Coordination note (two-session awareness)
A parallel session owns the **Spark CI / serving / micro-GRPO lane** (branch
`claude/dgx-spark-integration`, "Spark-N" commits). This workflow deliberately delegates
Spark-serving/CI execution (Phase 4-exec, 6-exec, the self-hosted runner) to that session and
focuses this effort on the **strategy doc + distillation-data pipeline + scaffold**, which are
non-overlapping. Sync before touching shared Spark files.

## Honest limits
- 273 GB/s bandwidth → **MoE is mandatory** for "smooth"; dense models are out.
- **NVFP4 broken on GB10** today → the FP4 marketing number is not real throughput.
- **aarch64 vLLM MoE must be verified** for your exact model+quant before committing.
- You are not beating frontier dense models on raw capability — you're beating them on
  *verified intelligence per token-cost*. State it that way.
