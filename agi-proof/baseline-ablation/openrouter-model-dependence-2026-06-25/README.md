# Calibration / anti-fabrication is model-dependent (OpenRouter, 2026-06-25)

First **working-backend** run of the calibration-abstention comparison through a
real model (the prior level-3 hidden smoke used *deterministic fixtures*, not model
runs — see `whyNotLevel3`). Run with the repo's OpenRouter transport
(`tools/run_hidden_eval_openrouter.py`) and scored by the authoritative
`tools/run_calibration_score.py --json` on the reviewer-authored abstain pack
(18 cases: 12 abstain + 6 definite).

## Result — fabrication rate (over the 12 abstain cases)

| Base model | raw | sophia-full (prompt) | gate-only (prompt) |
|---|---|---|---|
| **weak** — `meta-llama/llama-3.1-8b-instruct` | **0.25** (3/12) | **0.00** | 0.17 |
| **strong** — `deepseek/deepseek-chat` (v3) | **0.00** | 0.083 (1/12) | 0.00 |

(calibrationScore: weak raw 0.717 / sophia 0.689; strong raw 0.883 / sophia 0.906.)

## Finding

**The Sophia system-prompt's anti-fabrication advantage is model-dependent.**
- On the **weak** model it reproduces the ledger's effect: raw fabricates on 3/12
  abstain cases; the Sophia prompt drops that to 0/12.
- On the **strong** model there is **no headroom**: the raw model already fabricates
  0/12, and the Sophia prompt *added* one fabrication (slight regression) in this run.

## Why this matters for the AGI question

Provenance/abstention prompting behaves like a **crutch for weak models**. As the
base model strengthens — which is the actual path toward AGI — its marginal value on
the calibration axis decays toward zero (here, slightly negative). Real AGI value has
to come from **capability gains** (e.g. the verifier-gated RLVR learning that cleared
its rung), not from abstention scaffolding alone. This is a boundary condition on the
project's flagship "calibration advantage" claim, not a refutation of it.

## Caveats (this clears nothing)

N=18, **single run per model (no CI)**, deterministic regex scorer, **prompt-level**
modes (NOT the full gated pipeline), self/reviewer-authored pack. Does **not** change
`canClaimAGI` (False) and does not clear any verification gate. To harden: ≥3 runs for
CI, the full pipeline (`run_hidden_eval_sophia.py`) not just the prompt, and a
third-party-authored pack.

Reproduce: `OPENROUTER_API_KEY=… python tools/run_hidden_eval_openrouter.py --pack
agi-proof/baseline-ablation/abstain-pack-2026-06-22.json --mode {raw,sophia_full} --model <m>`
then `tools/run_calibration_score.py <pack> <dump> --json`.
