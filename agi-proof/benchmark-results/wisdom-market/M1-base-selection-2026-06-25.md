# M1 — Same-size base selection run (2026-06-25)

**What this is:** the M1 instrument (`tools/run_same_size_market_baselines.py`) run for the
first time against *real* same-size open models over the full held-out wisdom-market benchmark
(`data/wisdom_market_benchmark/heldout_v1.jsonl`, N=324, EN 179 / ZH 145). Conditions
`raw,prompt,prompt_gate`, **3 runs each**, stdlib bootstrap 95% CIs. No training. The two
semantic metrics are still marker-based (ILLUSTRATIVE) — no LLM judge was wired, so **no
semantic headline claim** is made here; the structural metrics are deterministic.

> This is base SELECTION evidence, not a capability claim. `canClaimAGI` stays False.

## Model lineup — and why it changed from the plan

The plan named `Qwen3-4B / Phi-4-mini / Llama-3.2-3B / small-Gemma`. Verified against the live
OpenRouter catalogue on 2026-06-25, the slugs had drifted:

| Plan target | OpenRouter status (2026-06-25) | Used |
|---|---|---|
| `qwen/qwen3-4b` | **Removed** — smallest dense Qwen3 is now `qwen3-8b` (8B, 2× target) | — (dropped; see ledger) |
| `microsoft/phi-4-multimodal-instruct` | removed | `microsoft/phi-4-mini-instruct` (the plan's "Phi-4-mini") |
| `meta-llama/llama-3.2-3b-instruct` | present | `meta-llama/llama-3.2-3b-instruct` |
| `google/gemma-2-9b-it` | removed (and 9B is not same-size) | `google/gemma-3-4b-it` (4B; the plan's "small Gemma") |

Per an explicit human decision, the Qwen slot was **dropped** (no same-size Qwen3 exists on
OpenRouter; using 8B would break the "same-size" win condition) rather than substituted. The
three bases run are all ~3–4B, honoring the same-size constraint.

Cost: ~$0.78 of OpenRouter spend for the full 3-model × 3-condition × 3-run sweep
(SOPHIA_MAX_TOKENS=512, concurrency 12).

## Results — the source/moral axes (prompt_gate vs raw, Δ with 95% CI; * = CI excludes 0)

`Δ` is the *improvement* (higher-is-better metrics: cond−raw; lower-is-better: raw−cond).

| metric | llama-3.2-3b | phi-4-mini | gemma-3-4b |
|---|---|---|---|
| provenance_accuracy (raw)        | 0.993 | 0.978 | 0.992 |
| false_attribution_rate Δ         | +0.000 | +0.0058* | +0.0097* |
| contested_fabrication_rate Δ     | +0.120* | +0.022 | +0.027 |
| citation_fidelity (raw / Δ)      | 1.000 / +0.000 | **0.694 / +0.250\*** | 1.000 / +0.000 |
| qualification_rate_on_contested Δ| **−0.328** | −0.027 | **+0.311\*** |
| tradition_merge_rate Δ (儒/道)    | +0.137* | +0.000 | +0.042 |
| moral_route_accuracy Δ           | +0.059 | +0.078 | +0.098 |
| tool_route_accuracy (raw / Δ)    | 0.429 / **−0.429** | 0.619 / +0.143 | **0.000 / +0.857\*** |
| over_abstention_rate (prompt_gate)| 0.037 | 0.077 | 0.050 |
| **protected_history_regression** (raw→pg) | 0.000→0.000 | 0.000→0.000 | **0.000→0.222** |
| protected_religion_regression (raw→pg) | 0.314→0.020 | 0.039→0.010 | 0.147→0.010 |
| useful_correctness (raw→prompt→pg)| 0.415→0.012→0.006 | 0.465→0.452→0.386 | 0.464→0.651→0.594 |

(Full per-run values and CIs in the three `baselines_*_2026-06-25.json` files.)

## Reading per base

**llama-3.2-3b-instruct — NO.** The *prompt scaffold itself* (not the gate) collapses it:
`useful_correctness` 0.415 → **0.012** the moment the route-first JSON instruction is added,
`qualification_rate_on_contested` 0.328 → 0.0, `tool_route_accuracy` 0.429 → 0.0. The 3B base
cannot follow the structured advisor format and emits unusable output. The gate's
fabrication/merge wins (contested_fab Δ+0.12, tradition_merge Δ+0.137) are real but moot when
the scaffold has already destroyed usefulness. **Disqualified.**

**gemma-3-4b-it — strongest uplift, but trips the protected wire.** The scaffold *raises*
usefulness (0.464 → 0.594) and the gate delivers the largest source/moral gains of any base:
`qualification_rate_on_contested` Δ **+0.311\***, `tool_route_accuracy` 0.0 → **0.857\***,
`moral_route_accuracy` +0.098, religion regression cut 0.147 → 0.010. **But** the gate induces
a **protected-history regression**: `protected_history_regression` 0.000 (raw & prompt) →
**0.222 (prompt_gate)** — i.e. the gate fails *closed* on ~22% of protected history cases it
should answer. Under the plan's mandatory "NO protected-suite regression" rule this
**disqualifies gemma as written** — even though the regression is caused by the *gate's
fail-closed behavior on history*, not by the base.

**phi-4-mini-instruct — the only base clearing every hard M1 constraint, but a modest/narrow
win.** It handles the scaffold gracefully (useful_correctness 0.465 → 0.386 under the gate — a
real but bounded cost). CI-clean prompt_gate improvements: `citation_fidelity` **+0.250\*** (raw
0.694, genuine headroom), `false_attribution_rate` +0.0058*, `provenance_accuracy` +0.0067*.
`over_abstention_rate` 0.077 ≤ 0.10 ✓. **No protected-suite regression** (history flat, religion
improved). The catch: its biggest clean win is citation fidelity, **not** the headline
儒/道 differentiator (`tradition_merge_rate` Δ0, raw already low at 0.095), and it shows a small
CI-clean `qualification` dip (−0.027).

## Go / No-Go (per plan lines 129–133 / 214–218)

- **llama**: FAIL (scaffold collapse).
- **gemma**: FAIL the strict gate (gate-induced protected-history regression) — *despite* the
  largest source/moral uplift.
- **phi-4-mini**: **PASSES the literal criteria** — ≥1 CI-clean source/moral improvement,
  over-abstention ≤ 0.10, no protected regression, raw headroom present — but the pass is
  **modest and narrow** (citation-fidelity-led, not differentiator-led).

## Honest caveats

1. **Benchmark saturation.** `provenance_accuracy` (0.98–0.99) and `citation_fidelity`
   (1.0 for llama/gemma) are already near-ceiling raw — the plan warned uplift cannot be proven
   on saturated axes. The genuine headroom is in qualification / tradition-merge / routing.
2. **The gate's fail-closed behavior is the pivotal variable.** The strongest base (gemma) is
   blocked *only* by a gate-induced history regression; this looks more like a gate-tuning
   issue than a base verdict.
3. **No semantic judge.** `useful_correctness` / qualification quality are marker-based here —
   ILLUSTRATIVE, not validated. A ≥2-family judge pass is required before any semantic headline.

## Decision surfaced to the human (borderline → STOP per decision protocol)

phi-4-mini technically passes M1 but modestly; the larger win (gemma) is gated out by a
protected-history fail-closed issue. This is a borderline call the plan says to escalate rather
than headline. See the failure-ledger row `sophia-wisdom-4b-m1-base-selection-ran-2026-06-25`.
