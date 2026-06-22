# Track-2 evidence run — DeepSeek, 2026-06-22

A founder-supplied DeepSeek key unblocked the baseline/ablation, learning-shift, and
horizon experiments. Reported honestly per the failure ledger: a negative or null
result is a result. Backend: `deepseek-chat` (resolves to `deepseek-v4-flash`).

## Headline (and a caveat that matters)

> **With a capable base model, the keyword/regex scorer cannot detect Sophia's value —
> and a *calibration-aware* scorer can.** Under keyword scoring the raw model ties or
> slightly beats Sophia-full; under calibration scoring (correct abstention vs.
> fabrication) Sophia-full clearly separates: **0% fabrication vs. raw 33–67%.**

## 1. Baseline / ablation (keyword scorer)

Same pack, same scorer, every mode (`tools/run_ablation_sophia.py --backend deepseek`).

### 5-case example pack — 3 runs ([artifact](deepseek-ablation-5case-2026-06-22.json))
| comparison | mean Δ (pts/12) | 95% CI | verdict |
|---|---|---|---|
| sophia-full − raw-model | **+1.33** | **[0.48, 2.18]** | CI excludes 0 → validated win over the *bare* model |
| sophia-full − raw-model+tools | −0.25 | [−0.74, 0.24] | tie; falsification rule triggered ×3 |

### 17-case hard pack — 3 runs ([artifact](deepseek-ablation-hard-2026-06-22.json))
| comparison | mean Δ (pts/45) | 95% CI | verdict |
|---|---|---|---|
| sophia-full − raw-model | **−1.11** | [−2.23, 0.01] | raw slightly *beats* full under keyword scoring; falsification ×3 |
| sophia-full − raw-model+tools | −0.19 | [−3.29, 2.91] | tie |

**Why:** keyword scoring rewards "the right term appeared." A capable base model already
denies well-known myths and names the right entities, so it matches Sophia — and Sophia's
correct hedging/abstention can *cost* keyword points. The scorer, not Sophia, is the limit.

## 2. Calibration scorer (the axis keyword scoring is blind to)

`provenance_bench/calibration_score.py`, applied to the **same captured DeepSeek
responses** on the hard pack ([artifact](deepseek-calibration-hard-2026-06-22.json)).
Abstain cases (unknown authorship / fabricated quote): reward honest "unknown", score a
confident fabricated specific 0.

| mode | calibration | **fabrication rate** | over-abstention |
|---|---|---|---|
| **sophia-full** | **1.000** | **0.00** | 0.00 |
| raw-model | 0.882 | 0.33 | 0.00 |
| raw-model+tools | 0.824 | 0.67 | 0.00 |

Δ full−raw **+0.118**, full−raw+tools **+0.176**. Raw DeepSeek fabricates an author/citation
on 1–2 of 3 unknown-answer cases; Sophia-full fabricates none. **This is Sophia's value,
finally measured.**

> **Caveat (no-overclaim gate):** single capture, 3 abstain cases. Directionally strong
> and mechanistically clear, but **not yet validated** (needs ≥3 captures; ideally a
> larger abstain set). Tracked in the failure ledger.

### 2a. VALIDATED — 18-case abstain pack, 3 runs ([artifact](deepseek-calibration-VALIDATED-2026-06-22.json))

Re-ran on a purpose-built 18-case pack (12 genuinely-unknown-authorship / unverifiable-quote /
unsolved-identity cases + 6 definite controls), 3 captures, calibration-scored.

| metric (sophia-full vs raw-model) | mean | 95% CI | excludes 0 |
|---|---|---|---|
| calibration Δ | **+22.0%** | [14.5%, 29.6%] | ✅ |
| fabrication reduction | **+19.4%** | [14.0%, 24.9%] | ✅ |
| sophia-full fabrication rate | **0.0%** | [0, 0] (3/3 runs) | — |

vs raw-model+tools: calibration Δ **+28.3%** [24.5%, 32.2%], fabrication reduction **+25.0%**
[15.6%, 34.4%]. **sophia-full never fabricates on the unknown cases across all 3 runs; raw
DeepSeek fabricates 17–25%.** This clears the no-overclaim gate (≥3 runs, CI excludes zero)
under the deterministic calibration scorer — recorded in `RESULTS.md` (Calibration evals).
Residual caveat: deterministic scorer + self-authored pack (internally valid; third-party
audit of labels/markers + human semantic review would harden to multi-judge headline grade).

## 3. Learning-under-shift ([artifact](../learning-under-shift/shift-result-2026-06-22.public-report.json))

The **mechanism works**: promotion gate promoted 1/2 candidate records and rejected 1;
contamination audit clean; protected knowledge unchanged; append-only memory diff written.
But the 1-case demo pack is too small to show an improvement signal (pre 0% → post 0%,
`passingSignal=false`). Mechanism sound; needs a real multi-case shift pack.

## 4. Effective-horizon curve ([artifact](deepseek-horizon-2026-06-22.json))

`deepseek-chat` on chained arithmetic: effective horizon (≥50% success) = **16 steps**.
Curve is non-monotonic at 8 trials (noisy). This is the horizon *curve* metric, **not** a
long-horizon *autonomy* run — that remains unrun.

## What changed in the ledger
- `baseline-ablation-missing` → ran; keyword-scored method advantage is **null** on a
  capable model. New open item: validate the **calibration** advantage at ≥3 runs.
- `distribution-shift-not-run` → mechanism ran; needs a real pack for a signal.
- Horizon curve measured (16 steps); long-horizon autonomy still open.
