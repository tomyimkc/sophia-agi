# Benchmark Results

This folder stores proof-package benchmark artifacts that are safe to publish.

Current visible religion benchmark target:

```bash
python tools/score_benchmark.py benchmark/reference/responses-religion.json \
  --domain religion \
  --out agi-proof/benchmark-results/religion-visible-reference-2026-06-19.report.json
```

Hidden benchmark results must not expose prompts before evaluation disclosure.
Publish only salted commitments, aggregate scores, and reviewer signatures until
the hidden pack is intentionally revealed.

## Current Public Artifacts

| Artifact | Model | Result | Notes |
|---|---|---:|---|
| `religion-visible-reference-2026-06-19.report.json` | sophia-teacher-reference | 5/5 strict pass | Visible religion benchmark |
| `hidden-prepared-pack-2026-06-19-grok-cli.public-report.json` | grok-cli | 28.75/40 auto score; 2/8 strict pass | Prepared hidden pack is now used/spent |
| `hidden-fresh-reviewer-pack-2026-06-19-sophia-grok.public-report.json` | sophia-full-grok | 0/40 auto score; 0/8 strict pass | Full Sophia runner executed, but Grok backend produced 0/8 nonempty answers because session/network setup failed |
| `hidden-fresh-reviewer-pack-2026-06-19-sophia-deepseek.public-report.json` | sophia-full-deepseek | 27.5/40 auto score; 0/8 strict pass | Full Sophia runner executed with DeepSeek backend; 8/8 nonempty answers and 0 backend failures, but manual semantic review remains pending |
| `hidden-fresh-reviewer-pack-2026-06-19-sophia-deepseek-coding-council-repair3.public-report.json` | sophia-full-deepseek-coding-council-repair3 | 31.9/40 auto score; 0/8 strict pass | Diagnostic spent-pack rerun with coding council routing, operational evidence prompts, empty-answer repair, and two-pass manual review template |
| `hidden-prepared-pack-2026-06-19-diagnosis.md` | analysis | diagnosis | Explains why the hidden score was weak and how to improve |

The hidden-pack report uses automatic keyword/avoidance scoring with partial
credit. Manual rubric scoring is still required before making strong claims.

## Full Sophia Hidden Runner

Use the full runner for proof-quality hidden evaluation:

```bash
python tools/run_hidden_eval_sophia.py private/hidden-evals/<pack>/pack.json \
  --responses-out private/hidden-evals/<pack>/sophia-responses.json \
  --private-report-out private/hidden-evals/<pack>/sophia-private-report.json \
  --public-report-out agi-proof/benchmark-results/hidden-<pack>-sophia.public-report.json \
  --manual-review-out private/hidden-evals/<pack>/manual-review-template.json \
  --repair
```

This path exercises Sophia retrieval, prompt modes, gate checks, one bounded
repair attempt, actual tool logs, append-only learning memory diffs, and a
manual semantic review template. Do not run a fresh sealed pack until the
reviewer is ready, because running it spends the pack.

The runner fails before exposing hidden prompts if backend preflight fails. For
already-spent smoke packs only, `--skip-preflight` may be used to test failure
reporting.
