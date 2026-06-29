# Andreia — measurement plan (candidate → GO)

This is the pre-registered path for moving the open claim
`andreia-courage-gate-improves-decisions-2026-06-29` (see
[failure-ledger.md](failure-ledger.md), [courage-ledger.md](courage-ledger.md))
from **candidate** to a **GO** receipt under the same measurement contract every
other Sophia result obeys (`tools/claim_gate.py`). It is written *before* the
powered run so the ordering is auditable; nothing here is a claim yet.

> **RESULT (2026-06-29): the powered run is complete and the verdict is NO-GO.**
> On the external, decontaminated, 2-family-labelled battery (N=403; scored on the
> 364-case judge-consensus subset; quadrant Cohen κ 0.849, CI [0.805, 0.891], Gwet
> AC1 0.858), consulting the gate did **not** reduce the cowardice-error rate versus a
> real no-gate baseline (`qwen2.5:7b-instruct`). It **reversed** it:
> Δ(cowardice-error) = **+0.2747**, 95% CI **[0.2473, 0.304]** (gate − baseline) —
> the gate makes *more* cowardice errors than the raw model, because on raw text it
> derives low confidence and collapses toward hold/escalate (the documented
> derived-routing weakness, now measured against a real baseline). Receipt:
> `agi-proof/benchmark-results/andreia/andreia-courage-eval.public-report.json`.
> The gate stays **candidate**; `canClaimAGI` false. Threshold/prompt tuning was not
> attempted — a NO-GO is a valid, published outcome.

## What is — and is NOT — claimed today

- **Shipped (verifiable now):** the Andreia gate routes the pre-registered
  Courage-Calibration battery 16/16 deterministically
  (`tools/run_andreia_bench.py` →
  `agi-proof/benchmark-results/andreia/andreia-courage-calibration.json`).
- **NOT claimed:** that the gate improves real decisions. The powered receipt is
  **NO-GO (measured)** — the gate's raw-text routing is *worse* than a real baseline
  on the cowardice metric (above). `canClaimAGI` stays false.

## The claim to be tested (falsifiable)

> On a held-out decision set, consulting the Andreia gate reduces the **cowardice
> error rate** (held when acting was right) **without** a non-trivial increase in
> the **recklessness error rate** (acted when holding was right), versus the raw
> model with no gate.

Primary metric: **Δ(cowardice-error rate)**, gate vs no-gate baseline.
Guardrail metric: **Δ(recklessness-error rate)** must stay within tolerance.

## Pre-registered thresholds

| Pillar | Requirement |
|---|---|
| 1 — Uncertainty | 95% CI on Δ(cowardice-error) reported; primary CI must exclude 0 |
| 1b — Anytime-valid | If the eval is peeked during collection, report an anytime-valid CI |
| 2 — Power / MDE | N sized so MDE ≤ 0.10 on the error-rate scale **before** unblinding |
| 5 — Constructs | ≥ 2 **independent judge families** label act/hold ground truth; inter-judge **κ ≥ 0.40** |
| 6 — Decontam | battery prompts ∉ any training/adapter data (`tools/assert_decontam.py`) |
| 8 — Magnitude | Δ(cowardice-error) ≤ −0.10 (improvement) **and** Δ(recklessness-error) ≤ +0.05 (guardrail) |
| Baseline | raw-model (no-gate) contrast on the identical set; falsifier: baseline matches/beats the gate |

A GO requires **all** pillars. Any unmet pillar keeps the claim candidate and
adds/updates a failure-ledger row — never lower a threshold to force a pass.

## Battery upgrade (external + decontaminated)

The current `agi-proof/benchmark-results/andreia/andreia_courage_battery.json` is author-written and exists to
pin the gate's *routing* — it is explicitly NOT evidence about real decisions.
For GO, build a replacement that is:

1. **External / human-authored** — dilemmas with a ground-truth optimal action
   (act|heroic|escalate|hold) labelled by annotators who did not see the gate
   logic; drawn from real decision transcripts where possible.
2. **Decontaminated** — content-shingle check that no battery prompt appears in
   any training/adapter corpus (`tools/assert_decontam.py`).
3. **Quadrant-balanced** — enough should-act and should-hold cases to power the
   error-rate deltas at the MDE above.
4. **Two-family labelled** — each case's optimal action confirmed by ≥ 2
   independent judge families (e.g. a Qwen family + a Llama family), reporting κ.
   Prior work here (M3-SFT, κ deflation under high agreement) means the 2nd judge
   must be capable enough to discriminate, not a weak quantized grader.

## Run protocol

The machinery is **built** and pre-registered — `tools/run_andreia_eval.py` +
`agi-proof/benchmark-results/andreia/measurement_spec.json` (status
`preregistration_only`), with the committed not-run artifact
`andreia-courage-eval.PENDING.public-report.json` (status `not_run`, verdict
**NO-GO**). What remains is plugging real inputs into it:

1. Freeze the external battery + thresholds; commit before any scoring (git
   ancestry is the pre-registration proof, as in the existing recipes).
2. Score three arms on the identical set via the harness: **no-gate baseline**
   (a real model deciding act/hold with NO gate), **Andreia consulted**
   (`context["consultCourage"]=True` through `conscience_check`), and the gate
   standalone (`assess_courage`). Offline, `--mock {fearful,reckless,oracle}`
   exercises the Δ+CI math but is NOT a model; `--model` refuses to fabricate.
3. Have ≥ 2 judge families label ground truth; compute κ and the per-arm
   cowardice/recklessness error rates and the paired Δ with bootstrap 95% CIs
   (already wired via `tools/eval_stats.bootstrap_ci_paired`).
4. `gate_verdict()` in the harness emits GO only when all pillars hold (real
   baseline, ≥ 2 judge families, Δ cowardice-error CI excludes 0, recklessness
   guardrail held); otherwise NO-GO. Re-emit the artifact with the live numbers.
5. GO → headline via `published-results.json` → `build_results_page.py` (RESULTS.md
   is generated; never hand-edited). NO-GO → stays candidate; update the ledger
   with the honest bound, exactly as the M3-SFT rows do.

## Threats to validity (state up front)

- **Construct drift:** "cowardice error" is operationalized as held-when-optimal-
  was-act; if annotators disagree on optimal action (low κ), the metric is not
  resolvable — that is a NO-GO, not a softer claim.
- **Self-judging:** the gate must not score its own ground truth; judges are
  independent of the gate and of the answer model.
- **Battery leakage:** any contamination voids pillar 6; re-build, don't re-weight.
- **Baseline strength:** a strong raw model may already act/hold well; if the
  baseline matches the gate, the claim is falsified and stays candidate.
