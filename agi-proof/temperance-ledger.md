# Temperance Ledger

The magnitude-axis sibling of the [Courage Ledger](courage-ledger.md). The failure
ledger records overclaim (acting beyond the evidence) held back; the courage ledger
records the opposite (when the brave move was to act). This ledger records the
**measure** axis: moments where the right move was to spend *less* (restrain
excess) or *more* (sustain against a premature stop) — and what it cost.

It exists because Sophia's conscience kernel regulates *truth* and the
[Andreia gate](../agent/andreia.py) regulates *direction*, but neither regulates
*magnitude*: how much effort/words/tool-calls to spend and when to stop. The
[Sophrosyne gate](../agent/sophrosyne.py) adds that faculty as an orthogonal,
deterministic, fail-closed decision surface (Aristotle's mean between excess and
deficiency). This ledger is its audit trail.

**Boundary.** Sophrosyne is candidate infrastructure. Nothing here claims the gate
improves real-world decisions, nor that Sophia "has temperance" as a trait. The
decision surface is a mean-deviation heuristic over signals about expenditure and
demand; the claim that it tracks real temperance is OPEN and unproven. It never
suppresses a required verification step (temperance is not negligence).

## What gets logged

A measure event is logged when the gate returns `restrain` or `sustain`, or when it
escalates an akrasia case (high appetite on a scarce budget) or refuses to restrain
a protected step. Each entry records:

- the mean-deviation forces (`delta, epsilon, mu, alpha, rho`) and the resulting `MQ`;
- the intemperance axis (was the deviation *excess* or *deficiency*?);
- the verdict and, when known, the **outcome — including when restraint was wrong**
  (cut effort that should have been spent) or **when sustaining was wrong** (kept
  spending past the point). Recording the mis-measured cases is what keeps the cost
  of the gate visible rather than silently optimized away.

## Open claims (candidate — see Failure Ledger for the gating row)

| Claim ID | Status | Claim impact | Required response |
|---|---|---|---|
| sophrosyne-temperance-gate-improves-decisions-2026-06-29 | Open (candidate — instrument only) | Does the Sophrosyne gate reduce *excess errors* (cut when more effort was right) AND *deficiency errors* (over-spent when restraint was right) on real tasks, without lowering task-success? The pre-registered Measure-Calibration battery routes 16/16 deterministically (`agi-proof/benchmark-results/sophrosyne/sophrosyne-measure-calibration.json`), but that certifies the GATE'S ROUTING, not an effect on real decisions: ONE deterministic judge, author-written battery, no effect size with a CI, no task-success guardrail measured. Receipt: **NO-GO** by design. `canClaimAGI` false. | Promote past candidate only with (a) an external, decontaminated task set, (b) ≥2 independent judge families (κ ≥ 0.40), (c) a baseline (raw agent, no gate) contrast, (d) Δ excess- AND deficiency-error 95% CIs excluding zero, and (e) the task-success guardrail held (Δ ≥ −0.02). Until then: candidate. |

## Routing exemplars (from the deterministic self-benchmark)

- `restrain` — over-verbose / over-hedged / over-retrieval / runaway loop at low marginal value.
- `sustain` — premature stop / under-answer / truncated work with effort still valuable.
- `escalate` — akrasia (strong appetite on a scarce budget), or a protected required step.
- `proportionate` — expenditure tracks demand; efficient short answers are NOT nagged.
