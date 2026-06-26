# External Benchmark Plan

External benchmarks are required before any stronger AGI claim. Current status:
capability-matched lanes are active; broad transfer benchmarks remain mostly
scaffolded or not run. Report deltas, not standalone scores, because TruthfulQA
and other public benchmarks may appear in base-model pretraining.

| Benchmark family | Capability tested | Required artifact |
|---|---|---|
| External authorship/citation/fact-check | Source-grounded fabrication and calibrated abstention | frozen pack, raw-vs-full reports, paired CI, over-abstention cost, manual review |
| TruthfulQA MC | Truthfulness discrimination over fixed choices | official data, base-vs-candidate report, paired CI |
| TruthfulQA generation | Free-response fabrication behavior | official data, base-vs-candidate reports, >=2 independent judge families, inter-judge kappa, paired CI |
| ARC-AGI / ARC-AGI-3 | Novel reasoning and skill-acquisition efficiency | official or reproducible score, solver config, per-task logs |
| GAIA-style tasks | Tool-using assistant reasoning | answer traces, tool logs, exact prompts, scoring script |
| SWE-bench-style repo tasks | Software maintenance agency | patches, test logs, resolved-task rate |
| METR-style autonomy | Long-horizon autonomous work | task suite, intervention count, full action logs |

## Capability-Matched Status

- External fact-check raw-vs-full has a 3-run first-party execution on
  `eval/external/factcheck-external-v1.jsonl`; the headline fabrication delta is
  an honest null because the CI includes 0, with high over-abstention cost.
- Google Fact Check Tools / ClaimReview now has a pack builder:
  `tools/build_google_factcheck_pack.py`. This creates a public third-party-
  authored audit pack from professional ClaimReview records. It is useful for a
  fast external audit, but it is not hidden and not fresh independent
  replication.
- TruthfulQA MC has an official 817-item first-party execution for base vs
  `sophia-v4-seed0-promoted`; adapter-minus-base is an honest null/negative
  point estimate.
- TruthfulQA generation now has offline-safe multi-judge scaffolding and the
  official 817-item generation dataset normalized at
  `eval/external/truthfulqa-generation.json`. The committed dry run is plumbing
  only: `agi-proof/external-eval/truthfulqa-generation-dry-run.json`.

## Google ClaimReview Audit Pack

This is the fastest public third-party-authored audit lane. It pulls
ClaimReview metadata from professional fact-checkers through Google's Fact
Check Tools API, normalizes only clean binary ratings into `true` / `false`,
drops mixed/contextual ratings, decontaminates against Sophia training/eval
text, and writes a frozen JSONL pack plus provenance sidecar.

1. Enable the API and set the key:

```bash
# Google Cloud Console -> enable "Fact Check Tools API" -> create API key
export GOOGLE_FACTCHECK_API_KEY="..."
```

2. Build a first public ClaimReview pack:

```bash
python tools/build_google_factcheck_pack.py \
  --query climate \
  --query election \
  --query health \
  --query economy \
  --max-per-query 75 \
  --out eval/external/factcheck-claimreview-v1.jsonl
```

Optional filters:

```bash
# Focus on one publisher:
python tools/build_google_factcheck_pack.py \
  --query climate \
  --publisher-site politifact.com \
  --out eval/external/factcheck-claimreview-politifact-v1.jsonl

# Restrict recency:
python tools/build_google_factcheck_pack.py \
  --query health \
  --max-age-days 365 \
  --out eval/external/factcheck-claimreview-recent-health-v1.jsonl
```

3. Run raw and full arms on the same frozen pack:

```bash
python tools/run_fact_check_live_eval.py \
  --condition raw \
  --live \
  --model mlx:Qwen/Qwen2.5-3B-Instruct \
  --pack eval/external/factcheck-claimreview-v1.jsonl \
  --out agi-proof/external-eval/factcheck-claimreview-raw-r1.json

python tools/run_fact_check_live_eval.py \
  --condition full \
  --live \
  --pack eval/external/factcheck-claimreview-v1.jsonl \
  --out agi-proof/external-eval/factcheck-claimreview-full-r1.json
```

Repeat both arms for at least 3 matched runs, then compare:

```bash
python tools/analyze_factcheck_arms.py \
  --raw agi-proof/external-eval/factcheck-claimreview-raw-r1.json \
        agi-proof/external-eval/factcheck-claimreview-raw-r2.json \
        agi-proof/external-eval/factcheck-claimreview-raw-r3.json \
  --full agi-proof/external-eval/factcheck-claimreview-full-r1.json \
         agi-proof/external-eval/factcheck-claimreview-full-r2.json \
         agi-proof/external-eval/factcheck-claimreview-full-r3.json \
  --out agi-proof/external-eval/factcheck-claimreview-raw-vs-full.summary.json
```

Claim boundary: this supports only public ClaimReview audit evidence. It does
not replace a fresh third-party-authored hidden pack, and it is not AGI proof.

## TruthfulQA Generation Live Commands

Fetch or refresh the pinned official generation data:

```bash
python tools/run_truthfulqa.py --fetch-generation-official \
  --out-data eval/external/truthfulqa-generation.json
```

Run base and candidate with the same prompt and the same independent judge
families:

```bash
python tools/run_truthfulqa.py --generation \
  --data eval/external/truthfulqa-generation.json \
  --model mlx:Qwen/Qwen2.5-3B-Instruct \
  --judge openai:gpt-4o \
  --judge anthropic:claude-sonnet-4-6 \
  --out agi-proof/external-eval/truthfulqa-generation-base-r1.json

python tools/run_truthfulqa.py --generation \
  --data eval/external/truthfulqa-generation.json \
  --model mlx:Qwen/Qwen2.5-3B-Instruct \
  --adapter training/mlx_adapters/sophia-v4-seed0-promoted \
  --judge openai:gpt-4o \
  --judge anthropic:claude-sonnet-4-6 \
  --out agi-proof/external-eval/truthfulqa-generation-candidate-r1.json

python tools/run_truthfulqa.py --generation --compare \
  --base-report agi-proof/external-eval/truthfulqa-generation-base-r1.json \
  --candidate-report agi-proof/external-eval/truthfulqa-generation-candidate-r1.json \
  --out agi-proof/external-eval/truthfulqa-generation-candidate-minus-base-r1.json
```

Repeat for at least 3 runs before making any capability claim. Current blocker:
this run has only exercised the mock/no-credentials path; live evidence still
needs a local or API subject model plus two independent judge credentials, and
then paired CIs across matching base/candidate items.

## Result Template

```json
{
  "benchmark": "",
  "date": "",
  "system": "sophia-full",
  "model": "",
  "score": null,
  "total": null,
  "cost_usd": null,
  "time_minutes": null,
  "logs": [],
  "failures": []
}
```

## Lane 1 — Authorship gate, coverage-matched & source-independent

Reproduce the defensible authorship result end to end.

1. Build the cross-source-confirmed pack (gold confirmed by OpenLibrary, independent of the gate's Wikidata):

```bash
python tools/build_freegen_authorship_pack.py \
  --require-cross-source \
  --out eval/external/freegen-authorship-xsrc-v1.jsonl
```

2. Run RAW and FULL, ≥3 runs each, on TWO base models (model 2 must be a different family, e.g. a Llama/Phi MLX build):

```bash
for M in "mlx:Qwen/Qwen2.5-3B-Instruct" "mlx:mlx-community/Llama-3.2-3B-Instruct-4bit"; do
  TAG=$(echo "$M" | tr '/:.' '___')
  for R in 1 2 3; do
    python tools/run_freegen_authorship_eval.py --condition raw  --model "$M" \
      --pack eval/external/freegen-authorship-xsrc-v1.jsonl \
      --run-id "raw-$TAG-$R"  --out "agi-proof/external-eval/freegen-xsrc-raw-$TAG-r$R.json"
    python tools/run_freegen_authorship_eval.py --condition full \
      --pack eval/external/freegen-authorship-xsrc-v1.jsonl \
      --run-id "full-$TAG-$R" --out "agi-proof/external-eval/freegen-xsrc-full-$TAG-r$R.json"
  done
  python tools/analyze_freegen_authorship_arms.py \
    --raw  agi-proof/external-eval/freegen-xsrc-raw-$TAG-r*.json \
    --full agi-proof/external-eval/freegen-xsrc-full-$TAG-r*.json \
    --out "agi-proof/external-eval/freegen-xsrc-raw-vs-full.$TAG.summary.json"
done
```

3. The defensible result is `selectiveRisk.aurc.favorsFull == true` (CI excludes 0) AND
   `acceptanceCoverageMatched.passed == true` on BOTH models. Report `matchedCoverage`
   (fabrication at equal coverage), not raw fabrication rate.

**Claim boundary:** first-party EXECUTION on a programmatically-built, OpenLibrary-cross-confirmed
external pack; gate (not adapter) validation; **not** an AGI claim; **not** third-party reproduced
until someone else regenerates the pack and reruns. The honest win is calibrated, coverage-matched
source discipline — fabrication reduction held to equal coverage — not a general capability.
