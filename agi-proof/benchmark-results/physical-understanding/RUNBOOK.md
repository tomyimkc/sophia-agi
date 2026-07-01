# Physical-understanding real-VLM run — runbook

Pre-registered in `measurement_spec.json`. This is the **gated, human-triggered**
path from the offline physical/2.5D harness to a real measured number. Nothing
here has executed; the failure-ledger entry
`physical-spatial-verifier-real-vlm-not-run-2026-06-29` stays OPEN until it does.

> **Before any GPU/paid step:** read `.claude/skills/wisdom-gpu-prebaked/SKILL.md`
> (the anti-wastage runbook — currently git-crypt locked; unlock first) and honor
> the RunPod cost guardrail (human approval on paid pods). RunPod GPU work goes
> through GitHub Actions only, never local SSH. Cheap validation first
> (`--runs 1`, a handful of rows), watch the first ~6 min, confirm zero leaked pods.

## What is already built (offline, in `main`)

- The **physical trap split** — 34 rows across depth ordering, occlusion,
  real-vs-apparent size, and 3D distance (+ controls + numeric `measure`), every
  gold re-derived by a judge-free verifier (`multimodal_bench/verifiers.py`).
- The **answer eval** with a multi-family consensus judge + no-overclaim
  aggregation (`tools/run_multimodal_traps.py`, now with `--physical`).
- The **fail-closed metric gate** + **depth-source seam**
  (`multimodal_bench/metric_gate.py`, `depth_backend.py`): authored z offline,
  Depth Anything V2 for pixel depth (blocker until weights are present).

## Step 1 — VLM grounding on the physical axes (multi-family judged)

```
# offline sanity (illustrative; never validated):
python tools/run_multimodal_traps.py --physical --answer mock:credulous --runs 3
python tools/run_multimodal_traps.py --physical --answer mock:grounded  --runs 3

# real run (opt-in): a real VLM, judged by >=2 DISTINCT families (!= subject):
python tools/run_multimodal_traps.py --physical \
    --answer openai:<vlm-model> \
    --judge-spec <family-a-spec> --judge-spec <family-b-spec> \
    --runs 3 --json > physical-answer-eval.json
```

Bring up the judge families via the sanctioned farm
(`.github/workflows/open-judge-runpod.yml` / `docs/11-Platform/Mac-Spark-Judge-Farm.md`)
— first-party GPT/Claude/Gemini are egress-blocked in CI, so use an
OpenAI-compatible endpoint or the local/RunPod judges.

**Pass bar:** `validated:true` from `runner.aggregate_runs` — i.e. `notMock` +
`multiFamilyJudges` + `kappaAboveFloor (>=0.40, or AC1+CI)` + `atLeast3Runs` +
`ciComputed`. The split is small: treat the verdict as a coarse GO/NO-GO and
expand the suite before any powered claim.

## Step 2 — pixel-derived metric grounding gate

```
# offline: authored depth (harness check) vs the weights-gated pixel path
python tools/run_metric_gate.py                          # authored z: 2/2 accept, 3/3 block
python tools/run_metric_gate.py --depth depth-anything   # blocker until weights present

# real run (GPU + weights): Depth Anything V2 supplies per-object z from pixels
python tools/run_metric_gate.py --depth depth-anything:depth-anything/Depth-Anything-V2-Small-hf --json
```

**Pass bar:** accept-rate on grounded claims >= 0.95 AND block-rate on
hallucinated claims >= 0.95, with `depth-anything` as the depth source (authored z
is harness-only and never a headline).

## Step 3 — promote (only if both bars clear)

Add the row to `agi-proof/benchmark-results/published-results.json` (candidate or,
if Step 1 clears the full no-overclaim gate, validated), regenerate `RESULTS.md`
via `tools/build_results_page.py`, and flip the ledger entry to CLOSED. **Claim
ceiling:** `candidate_only; canClaimAGI:false` — authored scenes are not natural
images, so this is corpus-bound feasibility, not a general physical-understanding
claim.
