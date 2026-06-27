# P6 — QLoRA content-uplift VALIDATION preregistration

**Status:** preregistration (written before the validating runs). `canClaimAGI` = false.

## Claim under test

> A 3-epoch QLoRA adapter of `Qwen/Qwen2.5-3B-Instruct`, trained on the curated provenance
> council traces, produces a **CONTENT-channel uplift** on the held-out 4-domain provenance
> eval (philosophy / psychology / history / religion) over the FP16 base.

The single-judge, single-seed run #7 measured a directional **+12.5pt** content uplift
(base 23/32 = 71.9% → adapter 27/32 = 84.4%). That is **candidate-only**. This document
preregisters the protocol that would move it to **VALIDATED**, so the seeds, judges, and
metric are fixed *before* the runs (no post-hoc selection).

## The VALIDATED bar (all five required)

Enforced by `tools/run_lora_uplift_validation.py` (reusing `provenance_bench` primitives;
identical thresholds to `provenance_bench/aggregate.py`):

1. **notMock** — real subject model, not a mock.
2. **≥2 independent judge families** — `provenance_bench.aggregate._distinct_families ≥ 2`.
3. **mean pairwise Cohen's κ ≥ 0.40** — inter-judge agreement on the adapter CONTENT labels.
4. **≥3 seeds** — independent training+eval replications.
5. **95% bootstrap CI on the uplift delta excludes zero**.

If any check fails, the result stays **candidate-only / illustrative** — never a capability claim.

## Fixed protocol parameters (preregistered)

- **Subject:** `Qwen/Qwen2.5-3B-Instruct`, QLoRA 4-bit, **3 epochs** (the config that showed
  the directional uplift). Optionally also the `--lora-rank-alloc` variant as a *separate*
  preregistered arm — do not pool the two.
- **Seeds:** `0, 1, 2` (three independent train+eval runs each, via `train-runpod.yml`).
- **Judges (≥2 families, each ≠ the subject's family `qwen`):** the subject is the `qwen`
  family, so **qwen-family judges are disqualified** (judge ≠ subject). Use e.g.
  `openrouter:deepseek/deepseek-chat` (deepseek) + `openrouter:meta-llama/llama-3.1-70b-instruct`
  (meta-llama). A local judge farm (two vLLM servers of distinct families) also satisfies the gate.
- **Metric:** consensus (majority-of-families) CONTENT pass-rate uplift `adapter − base`,
  per seed; aggregated mean with a 95% bootstrap CI over per-seed deltas.
- **Decontamination:** the eval set must be decontaminated from the training traces
  (overlap count 0 — already asserted by the W2 `contamination_zero` invariant).

## Pipeline

1. **Train** (×3 seeds) and **generate** base + adapter answers for all eval items
   (`train-runpod.yml` already runs the eval ladder; extend it to dump per-item transcripts).
2. **Judge** each transcript with the ≥2 judge families on the CONTENT channel
   (reuse `provenance_bench.llm_judge.make_llm_judge`; judge prompt = the content rubric of
   `agent/benchmark_checks.py`). Emit the judgments JSON schema in
   `tools/run_lora_uplift_validation.py`'s docstring.
3. **Aggregate + gate** with `tools/run_lora_uplift_validation.py --judgments …` → a
   no-overclaim report with `validated` and the five `validatedChecks`.

## What's built vs what remains

- **Built & tested (offline):** the aggregation + gate half — κ (Cohen), 95% bootstrap CI,
  family counting, and the 5-check VALIDATED gate, with a `--mock` self-test
  (`tests/test_lora_uplift_validation.py`, 6 tests). A mock subject can never validate
  (notMock=false), by design.
- **Remains (needs network + ≥3 paid GPU runs):** the upstream step — 3 seeded train+eval
  runs producing per-item transcripts, then labelling them with 2 non-qwen judge families.
  This spends real GPU + judge-API budget and is gated on explicit go-ahead.

## Honesty notes

- The single-judge eval ladder (`tools/eval_local_model.py`) uses a **lexical** content
  scorer, not an LLM judge. The VALIDATED protocol deliberately swaps in **independent LLM
  judges** — a lexical scorer cannot satisfy the ≥2-judge-family / κ gate.
- A VALIDATED uplift here is a **narrow, nano-scope** claim (one 3B model, one 32-item
  provenance eval). It does **not** move `canClaimAGI`; it is one honest measurement to the
  project's no-overclaim standard, nothing more.
