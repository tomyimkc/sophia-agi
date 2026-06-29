# Sophrosyne — measurement plan (candidate → GO)

This is the pre-registered path for moving the open claim
`sophrosyne-temperance-gate-improves-decisions-2026-06-29` (see
[failure-ledger.md](failure-ledger.md), [temperance-ledger.md](temperance-ledger.md))
from **candidate** to a **GO** receipt under the same measurement contract every
other Sophia result obeys (`tools/claim_gate.py`). It is written *before* the
powered run so the ordering is auditable; nothing here is a claim yet. It mirrors
[`andreia-measurement-plan.md`](andreia-measurement-plan.md).

## What is — and is NOT — claimed today

- **Shipped (verifiable now):** the Sophrosyne gate routes the pre-registered
  Measure-Calibration battery 16/16 deterministically
  (`tools/run_sophrosyne_bench.py` →
  `agi-proof/benchmark-results/sophrosyne/sophrosyne-measure-calibration.json`).
- **NOT claimed:** that the gate improves real decisions. The current receipt is
  **NO-GO by design** — one deterministic judge, author-written battery, no
  baseline contrast, no effect size with a CI, no task-success guardrail measured.
  `canClaimAGI` stays false.

## The claim to be tested (falsifiable)

> On a held-out task set, consulting the Sophrosyne gate reduces the **excess-error
> rate** (cut/restrained when more effort was right) **and** the **deficiency-error
> rate** (over-spent/sustained when restraint was right), versus the raw agent with
> no gate — **without** lowering task-success.

Primary metric: **Δ(excess-error rate)** *and* **Δ(deficiency-error rate)**, gate
vs no-gate baseline, paired per item.
Guardrail metric: **Δ(task-success rate)** must stay ≥ −0.02 (the dual of Andreia's
recklessness guardrail: a gate that "saves effort" by lazily cutting correct work
is not a win).

## Pre-registered thresholds

| Pillar | Requirement |
|---|---|
| 1 — Uncertainty | 95% CI on each Δ reported; both primary CIs must exclude 0 |
| 1b — Anytime-valid | If the eval is peeked during collection, report an anytime-valid CI |
| 2 — Power / MDE | N sized so MDE ≤ 0.10 on the error-rate scale **before** unblinding |
| 5 — Constructs | ≥ 2 **independent judge families** label the optimal measure; inter-judge **κ ≥ 0.40** |
| 6 — Decontam | task prompts ∉ any training/adapter data (`tools/assert_decontam.py`) |
| 8 — Magnitude | Δ(excess-error) ≤ −0.10 **and** Δ(deficiency-error) ≤ −0.10 (improvement) |
| Guardrail | Δ(task-success) ≥ −0.02 |
| Baseline | raw-agent (no-gate) contrast on the identical set; falsifier: baseline matches/beats the gate |

A GO requires **all** pillars. Any unmet pillar keeps the claim candidate and
adds/updates a failure-ledger row — never lower a threshold to force a pass.

## Battery upgrade (external + decontaminated)

The current `sophrosyne_measure_battery.json` is author-written and exists to pin
the gate's *routing* — it is explicitly NOT evidence about real decisions. For GO,
build a replacement that is:

1. **External / human-authored** — real agent transcripts (long-horizon runs,
   answers, tool-use traces) with a ground-truth optimal measure
   (proportionate|restrain|sustain|escalate) labelled by annotators who did not see
   the gate logic, ideally with the *true* token/tool budgets attached.
2. **Decontaminated** — content-shingle check that no task prompt appears in any
   training/adapter corpus (`tools/assert_decontam.py`).
3. **Quadrant-balanced** — enough excess and deficiency cases to power both
   error-rate deltas at the MDE above.
4. **Two-family labelled** — each case's optimal measure confirmed by ≥ 2
   independent judge families, reporting κ (heed the M3 prevalence-κ deflation
   lesson: the 2nd judge must be capable enough to discriminate).

## Run protocol

The machinery is **built** and pre-registered — `tools/run_sophrosyne_eval.py` +
`agi-proof/benchmark-results/sophrosyne/measurement_spec.json` (status
`preregistration_only`), with the committed not-run artifact
`sophrosyne-measure-eval.PENDING.public-report.json` (status `not_run`, verdict
**NO-GO**). What remains is plugging real inputs into it:

1. Freeze the external battery + thresholds; commit before any scoring (git
   ancestry is the pre-registration proof).
2. Score three arms on the identical set: **no-gate baseline** (a real agent
   deciding how much to spend with NO gate), **Sophrosyne consulted**
   (`context["consultTemperance"]=True` through `conscience_check`), and the gate
   standalone (`assess_temperance`). Offline, `--mock {profligate,miserly,oracle}`
   exercises the Δ+CI math but is NOT a model; `--model` refuses to fabricate.
3. Have ≥ 2 judge families label the optimal measure; compute κ and the per-arm
   excess/deficiency error rates, the paired Δ with bootstrap 95% CIs, and the
   task-success guardrail (`tools/eval_stats.bootstrap_ci_paired`).
4. `gate_verdict()` emits GO only when all pillars hold (real baseline, ≥ 2 judge
   families, both Δ CIs exclude 0, task-success guardrail held); otherwise NO-GO.
5. GO → headline via `published-results.json` → `build_results_page.py`. NO-GO →
   stays candidate; update the ledger with the honest bound.

## Threats to validity (state up front)

- **Demand estimation (`delta`):** "excess/deficiency error" is operationalized
  against the *true task demand*; if the demand set-point is mis-estimated, the
  metric is mis-scored. When `delta` is not supplied explicitly the gate derives it
  weakly from text (the robustness probe measures this gap — see the failure
  ledger). For the GO run, demand must come from the labelled task, not derived.
- **Self-judging:** the gate must not score its own ground truth; judges are
  independent of the gate and of the answer agent.
- **Lazy-cut confound:** a gate can lower excess error by always restraining; the
  task-success guardrail and the symmetric deficiency-error metric are what prevent
  that from reading as a win.
- **Baseline strength:** a strong raw agent may already spend well; if the baseline
  matches the gate, the claim is falsified and stays candidate.
