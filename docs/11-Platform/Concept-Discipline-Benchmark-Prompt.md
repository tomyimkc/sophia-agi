# Concept-Discipline Benchmark — prompt for an external AI runner

Hand this whole file to another AI (or a human) with repo + (optionally) GPU
access. It runs two measurements — **inference uplift** (offline, no GPU) and
**training uplift** (RLVR, GPU via GitHub Actions only) — under the no-overclaim
gate, and reports honestly. Nothing here may set `canClaimAGI`.

---

## Mission

Decide, with evidence, whether the concept-discipline machinery in this repo
**measurably improves a model's behaviour** on philosopher-style concept reasoning,
without introducing an over-abstention regression and without the gain being a
spurious-reward artifact. Report VERIFIED / NOT VERIFIED / INCONCLUSIVE per arm.
Do **not** claim improvement unless the stated bars are met.

## What "improvement" means (falsifiable)

The built modules add a fail-closed **concept gate** (no unscoped cross-tradition
identity, e.g. "ren is identical to agape") + a verifier-as-reward for RLVR. A
real improvement is, on a held-out split:

1. **concept-merge-violation rate DOWN** (bootstrap 95% CI of the delta excludes 0), AND
2. **over-abstention rate NOT up** beyond tolerance (the tripwire must not fire), AND
3. the gain **does not replicate under a spurious (random) reward**, AND
4. for the RLVR arm: ≥3 seeds, ≥2 base-model families, 95% CI excludes 0
   (`provenance_bench.aggregate._is_validated`).

If any bar fails, the honest verdict is NOT VERIFIED — say so.

---

## Arm 1 — Inference uplift (offline, runs on any machine incl. Apple Silicon)

```bash
# Sanity: offline invariants for every new module
python -m pytest tests/test_concept_testbench.py tests/test_concept_integration.py -q

# A/B: same policy, raw (arm A) vs concept-gated guarded loop (arm B)
python tools/run_concept_discipline_bench.py --model mock --reference naive   --seeds 3
python tools/run_concept_discipline_bench.py --model mock --reference disciplined --seeds 3

# Real model (any agent.model spec: openai-compatible endpoint, anthropic, grok, ...)
python tools/run_concept_discipline_bench.py --model "<your-model-spec>" --seeds 3
```

Read `agi-proof/benchmark-results/concept-discipline-inference.public-report.json`.
Report PASS only if `upliftReported == true` (violation-delta CI excludes 0 and < 0
on every seed, `overAbstentionTripwire.tripped == false`, and
`spuriousAblation.discriminates == true`). Include the per-seed violation deltas and
the over-abstain rate of both arms in your write-up — never hide the tradeoff.

## Arm 2 — Training uplift (RLVR; GPU **only** via GitHub Actions → RunPod)

> Repo rule (`.cursor/rules/runpod-github-actions.mdc`): never launch RunPod from a
> local shell. Use the workflow.

```bash
# Offline reward-wiring + spurious-ablation gate (no GPU; must pass before any GPU run)
python tools/run_rlvr.py --task concept --model mock
# -> agi-proof/benchmark-results/rlvr.public-report.json ; expect "RLVR REWARD WIRING VERIFIED ✓"
```

Then launch the GPU run from **GitHub → Actions → `rlvr-runpod` → Run workflow**:
`confirm=RUN`, `remote_mode=live`, `task=concept`, and vary `seed` across ≥3 parallel
dispatches (e.g. 0, 1, 2). Repeat with a second base model family for the ≥2-family
requirement. The workflow rents the pod, trains, copies the adapter-eval back, and
gates it through SSIL Layer-1 (`tools/ingest_rlvr_eval.py`); it always deletes the pod.

After the runs, aggregate and apply the no-overclaim gate (aggregate reads the
per-seed `*.adapter-eval.json` reports — it is task-agnostic via the ingest mapping):

```bash
python tools/aggregate_rlvr_runs.py \
  --reports agi-proof/benchmark-results/runpod-rlvr/**/*adapter-eval*.json \
  --registry agi-proof/benchmark-results/rlvr-replication/registry.jsonl --print
# canonicalizes only after >=3 seeds with a positive delta; pair with >=2 model families.
```

Report PASS for Arm 2 only if held-out grounded-correct / abstention rises vs the
untrained base adapter, the over-abstention tripwire does not fire, the spurious
ablation discriminates, and `_is_validated` is true.

---

## Deliverable (what to return)

A short report, one section per arm:
- the exact commands run and the report JSON paths;
- the headline numbers (violation rate baseline→treatment, grounded-correct,
  over-abstain rate, each delta's 95% CI);
- the spurious-ablation result and the over-abstention tripwire state;
- a one-line verdict per arm: **VERIFIED / NOT VERIFIED / INCONCLUSIVE**, with the
  single most decisive number;
- explicit confirmation that `canClaimAGI` is still `false` and that you made no
  capability claim beyond "concept-discipline behaviour improved on this eval".

## Hard rules

- Do not edit the eval set, the reward, or the gate to make numbers move.
- Do not run RunPod locally. GPU work goes through the `rlvr-runpod` workflow.
- A point estimate is not a result — every claim needs a CI that excludes 0.
- If you cannot get a GPU, complete Arm 1 and mark Arm 2 INCONCLUSIVE (not failed).
- Report the over-abstention cost even when the violation rate improves.
