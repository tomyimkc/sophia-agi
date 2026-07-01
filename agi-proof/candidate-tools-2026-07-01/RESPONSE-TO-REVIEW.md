# Response to the adversarial review (2026-07-01)

The review was accurate and high-value. I independently re-verified its findings against the
working tree (`feat/realtime-grounding-loop` @ `537279f9`) — **7 of 7 spot-checked structural
findings confirmed**, zero corrections to the reviewer. This document records disposition of
every defect: what I fixed in code (with tests), what needs a maintainer module change, and
what needs a human.

## One correction to the review's premise

The review's **section 0** ("the tools are not in the repo; could not be import-tested") was
**true when the review ran but is now stale.** The tools were copied into the working tree, and
**D1 (the fabricated `Model(spec).complete(prompt)` binding) was already fixed** before the
review arrived — rebound to `default_client(spec).generate(system, user)` + a mock-provider
fail-closed guard, and verified to import against the live `agent.model`. So D1 is **resolved**,
not open. Everything else in the review stands.

## Disposition table

| Defect | Severity | Status | What changed |
|---|---|---|---|
| D1 — fabricated model API | CRITICAL | **FIXED (prior turn)** | `default_client(spec).generate(system,user)`; mock→fail-closed; verified against live `agent.model` |
| D3 — WS-A validates wrong contract (schema.json vs runner's `validate_pack`) | CRITICAL | **FIXED** | WS-A now imports `tools.hidden_eval_protocol.validate_pack` and binds to the runner's real contract; fail-closed if unimportable (no divergent fallback) |
| D8 — `reward>=0.0` accepts abstain/vacuous as "grounded" | MEDIUM | **FIXED** | Bound to real `REWARD_CLEAN`/`REWARD_ABSTAIN`; substantive pass now requires `>= REWARD_CLEAN` and a gold-reference match; no-reference items never credited |
| D5 — protected-suite regression silently ignored | HIGH | **FIXED** | `run_round(..., protected={suite:[items]})` builds `EvalMetric(protected=True)` per suite so `evaluate_update` can reject on protected regression |
| D7 — `require_artifacts` is a bare `len()` | MEDIUM | **FLAGGED in-code** | Kept count-satisfying paths but added an explicit `NOTE(D7)` in candidate `notes` that real verifier RUN artifacts are required before a live claim |
| Q-C(1) — ARC parser aborts at first non-grid line | HIGH | **FIXED** | Robust parser: strips ``` fences, scans for the longest equal-width integer block, ignores prose; 4 new tests incl. "correct grid behind prose scores correct" |
| D2 — WS-D no-executor confounded, no single dispatch site | CRITICAL | **REDESIGNED (doc)** | Patch doc rewritten: single-flag plan WITHDRAWN; two defensible options (reuse existing `sophia-no-tools`, or multi-site gate + exclude `requiresToolLog`/`requiresMemoryDiff` cases) |
| D6 — Ablation field-count drift | HIGH | **DOCUMENTED** | Patch doc records the two extra fields (`use_context_packing`, `context_packing_policy`) and mandates keyword-only construction |
| D4 — `--minutes` cannot stop a runaway | HIGH | **NEEDS MAINTAINER (module change)** | WS-D docstring now states the limitation honestly and gives the exact `deadline_monotonic` recipe for `agent/long_horizon.py`; the wrapper cannot fix it |
| Q-B1 — transfer classifier decorative until a trainer is wired | — | **ACKNOWLEDGED** | `gen_after=None` dry-run is honestly marked `trained:false`; promote path is unreachable without a real SFT/DPO step + a genuinely disjoint `heldout_shifted` (not a paraphrase) |
| Q-B2 — reward-hacking surface (abstain-on-answerable) | — | **MITIGATED + ACKNOWLEDGED** | D8 fix stops labelling abstain as grounded; full mitigation needs a per-row answerability/gold label, documented |
| Q-A — third-party stamp unverifiable; decontam can't see pretraining | — | **ACKNOWLEDGED** | Stays gated (`canClaimAGI:false`); real external-signature verify + pretraining-corpus caveat noted as the close condition |

## Tests

24 offline tests pass — up from 17 — via (the four package test files by name; `tests/`
holds ~529 repo tests, so do NOT use a bare `pytest tests/`):

```
PYTHONPATH=. python3 -m pytest \
  tests/test_make_independent_hidden_pack.py tests/test_run_t1_gated_self_training.py \
  tests/test_run_arc_agi_sophia.py tests/test_ws_d_free_wins.py -q
```

Per-file: 4 + 6 + 10 + 4 = 24. The 7 new
tests lock in the review-driven behaviors (D8 abstain-not-credited, no-reference-not-credited,
D5 protected-regression-rejects, and four Q-C robust-parser cases).

## What still needs a maintainer (not doable from a drop-in)

1. **D4** — add `deadline_monotonic` to `run_long_horizon` (module change) so `--minutes` binds.
2. **D2** — decide Option 1 (`sophia-no-tools`) vs Option 2 (multi-site gate) for the ablation.
3. **Q-B1** — wire the SFT/DPO step so the promote path is exercised; ensure `heldout_shifted` is a genuinely disjoint distribution.
4. **WS-A** — a human external reviewer (independence cannot be self-satisfied).

## Run-first recommendation (agreeing with the review)

**WS-C, scoped to ARC-AGI-1, gate-off vs gate-on**, after the D1 binding + the parser fix
(both now done). It is the only purely compute-gated, external, judge-free evidence among the
five; cheap (minutes, a few dollars of API); and framed as accuracy-at-matched-coverage it
reinforces the repo's honest calibration headline rather than overclaiming.