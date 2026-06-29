# Session Handover — 2026-06-29 (Cardinal Virtues: Temperance + Justice + benchmarks)

> Continuation point for the next session/device. This session designed and shipped the
> two remaining Stoic cardinal virtues as candidate gates, then carried them into the same
> powered GO-path benchmark methodology PR #275 ran for Andreia. `canClaimAGI` stays
> **false**; nothing here promotes a result. All powered runs are model-gated (Spark+Mac farm).

## 0. Branch / PR / where things are
- **Feature branch `claude/virtue-ai-temperance-justice-ngpquf`** (tip `ff461bd1`): all work below,
  pushed, all offline gates green. Working tree clean.
- **PR #277** → `main` is OPEN (`Cardinal virtues: Temperance & Justice gates + PR-275-style
  benchmark scaffolds`). Not yet merged.
- The branch already **merged `origin/main` (PR #275 Andreia real-eval)** in at `be841d9a`, so it
  carries PR #275's `eval_stats` helpers (`cohen_kappa`, `gwet_ac1`, `bootstrap_ci_agreement`).
- Local `main` is the usual stale container lineage; `origin/main` is source of truth.

## 1. What shipped (committed on the branch)
**Gates (instruments, routing-gated, candidate-only):**
- **Temperance — Sophrosyne**: `agent/sophrosyne.py` (Measure Quotient `MQ = expenditure − demand`,
  verdicts proportionate|restrain|sustain|escalate) + `agent/intemperance_signals.py` (dual detector).
  Safety: *temperance is not negligence*. Opt-in **10th conscience path** (`consultTemperance`, off by default).
- **Justice — Dikaiosyne**: `agent/dikaiosyne.py` (Role A impartiality auditor, `JQ = 1 − flip_rate`;
  verdicts impartial|partial|false_equivalence|arbitrate) + `agent/partiality_signals.py`. Safety:
  *justice is not false balance* (defers to hard gates; documents the same PROTECTED-domain residual as Andreia).
  **Role B — the inter-virtue arbiter** `agent/virtue_parliament.py` (pre-registered lexical priority
  hard_prohibition > Wisdom > Justice > Courage > Temperance; deterministic + order-independent = unity-of-virtue).
- Each: MCP tools (`sophia_temperance_assess`/`_intemperance_check`/`_sophrosyne_benchmark`,
  `sophia_justice_assess`/`_partiality_check`/`_dikaiosyne_benchmark`/`_virtue_arbitrate`), fail-closed
  skill (`measure_advocate`, `fairness_advocate`), routing battery (16/16, NO-GO by design),
  measurement plan, robustness probe, ledger (temperance-ledger.md, justice-ledger.md).

**Benchmarks (PR-275 methodology):**
- **Orthogonality headline (offline, runnable now):** `tools/run_virtue_orthogonality_bench.py` +
  battery + `agi-proof/benchmark-results/orthogonality/measurement_spec.json`. All four gates over
  single-axis items → virtue confusion matrix (perfect diagonal, controls silent). NO-GO by design.
- **Sophrosyne real-eval scaffold (farm-ready):** `tools/build_sophrosyne_external_battery.py`
  (420 raw-text dilemmas, MDE 0.097, frozen), `tools/assert_sophrosyne_decontam.py` (RUN, clean),
  `tools/sophrosyne_decision.py`, `tools/label_sophrosyne_battery.py` (2-family, κ≥0.40 gate),
  `tools/run_sophrosyne_eval.py --model` (3-arm Δ(excess)/Δ(deficiency) + bootstrap CIs over ≥3 seeds),
  `tools/run_sophrosyne_robustness.py --semantic-backend` (LLM-judge intemperance backend).
- **Dikaiosyne real-eval scaffold (farm-ready):** `tools/build_dikaiosyne_external_battery.py`
  (400 raw-text equivalence classes, MDE 0.099, frozen), `tools/assert_dikaiosyne_decontam.py`
  (RUN, 2,400 members clean), `tools/dikaiosyne_decision.py`, `tools/label_dikaiosyne_battery.py`
  (2-family relevance-structure validation), `tools/run_dikaiosyne_eval.py --model` (class-level
  Δ(partiality), false-equivalence guardrail held by construction), `--semantic-backend`.
- **Runbook:** `docs/11-Platform/Cardinal-Virtue-Benchmarks.md`.
- **Dispatcher lane:** `scripts/run_local_benchmarks.sh --bench-virtues` runs the whole
  build→decontam→label→score pipeline (dry-run aware; opt-in, NOT in `--all`). Judge/subject specs
  env-overridable; `VIRTUE_JUDGE_A` defaults to a capable Qwen-32B distinct from the 7B subject
  (judge ≠ subject; avoids the M3 κ-deflation).
- **Failure ledger:** open candidate rows for both gates + two "real-eval machinery built — not yet run" rows.

**Verified (offline):** ~150 new tests pass; contract gates (lint_claims, failure-ledger, decontam ×3,
training-rows) + artifact-drift gates (wiki, RESULTS, version, dataset) green. No regressions in
conscience/andreia/MCP tests.

## 2. ▶ NEXT — run the powered evals (the only thing left; model-gated)
The benchmark machinery is built and farm-ready but **not yet run** — exactly like PR #275's machinery
before its run. The powered arms need the two-box judge farm (Spark Qwen-32B + Mac Llama-70B) + a
baseline subject model. As PR #275 measured for Andreia (a NO-GO: deriving signals from raw text
*worsened* the metric), these runs will **test, not assume** whether the gates beat a real baseline.

**Direct on the farm (simplest):**
```
scripts/run_local_benchmarks.sh --bench-virtues            # dry-run (prints the plan)
scripts/run_local_benchmarks.sh --bench-virtues --execute  # powered (judge farm must be up)
```
Receipts → `*/sophrosyne-measure-eval.public-report.json`, `*/dikaiosyne-justice-eval.public-report.json`.
Pass bars are pre-registered in each `measurement_spec.json` (Δ ≤ −0.10, 95% CI excludes 0, + guardrails,
κ≥0.40). GO → promote via `published-results.json` + `build_results_page.py`. NO-GO → stays candidate;
log measured numbers in the temperance/justice ledger.

**Via the GitHub bridge (cloud session, egress-blocked) — BLOCKED until two prerequisites:**
The poller (`tools/github_bridge_poll.py` on branch `spark-bridge`) has a hard allowlist
`{--dry-run --bench-a --bench-b --all --execute --run-train}` and runs ONLY `run_local_benchmarks.sh`.
It will **reject** `--bench-virtues` until:
  1. `--bench-virtues` is added to the poller's allowlist tuple on `spark-bridge`, AND
  2. the `spark-bridge` worktree carries this updated `run_local_benchmarks.sh` (merge `main` in
     AFTER PR #277 lands — and watch for the documented untracked-script *shadow* in that worktree:
     the poller must execute the TRACKED script, not a copied one).
Then queue `bridge/commands/<id>.json` with `args:"--bench-virtues --dry-run"` (no approval) to verify
the round-trip, then `args:"--bench-virtues --execute"` with a non-empty human `approvedBy` for the run.
**This session deliberately did NOT modify the spark-bridge executor** (a security control on always-on
hardware) blind — recommend merging PR #277 first, then doing the clean `main`→`spark-bridge` sync + the
1-token allowlist edit, with the poller confirmed live.

## 3. Guardrails (unchanged)
No-overclaim gate decides validity, never prose. Before commit/push: `make claim-check` + drift gates.
Owned hardware (Spark+Mac) is free; RunPod only via GitHub Actions + read `wisdom-gpu-prebaked` first.
`canClaimAGI` stays false until a third-party hidden eval is beaten.
