# Courage Ledger

The dual of the [Failure Ledger](failure-ledger.md). The failure ledger records
where Sophia is *not* yet what it could be (recklessness/overclaim caught and
held back). This ledger records the opposite axis: moments where the brave,
well-calibrated move was to **act** — and what it cost.

It exists because Sophia's conscience kernel is, by design, a fear apparatus: of
its seven verdicts (allow|revise|retrieve|clarify|escalate|abstain|block) six are
forms of retreat. A system that can only retreat cannot tell genuine prudence
from "cowardice disguised as prudence" (Holiday, *Courage Is Calling*). The
[Andreia gate](../agent/andreia.py) adds the missing faculty as an orthogonal,
deterministic, fail-closed decision surface. This ledger is its audit trail.

**Boundary.** Andreia is candidate infrastructure. Nothing here claims the gate
improves real-world decisions, nor that Sophia "has courage" as a trait. The
decision surface is a phase-transition heuristic over signals Sophia already
computes; the claim that it tracks real courage is OPEN and unproven.

## What gets logged

A courage event is logged when the gate returns `act` or `heroic`, or when it
overrides an apparent prudent hold to `escalate` because the hold looked
fear-driven. Each entry records:

- the ASIR forces (`lambda, gamma, psi, theta, phi`) and the resulting `CQ`;
- the fear attribution (was the dominant inhibitor genuine *epistemic risk* or
  *social/reputational cost*?);
- the verdict and, when known, the **outcome — including when courage was
  punished** (acted rightly and was penalized). Cowardice has a cost too
  (regret, wasted potential, complicity); recording the punished-courage cases is
  what keeps that cost visible rather than silently optimized away.

## Open claims (candidate — see Failure Ledger for the gating row)

| Claim ID | Status | Claim impact | Required response |
|---|---|---|---|
| andreia-courage-gate-improves-decisions-2026-06-29 | Open (candidate — instrument only) | Does the Andreia courage gate reduce *cowardice errors* (held when acting was right) without raising *recklessness errors* (acted when holding was right) on real decisions? The pre-registered Courage-Calibration battery routes 16/16 deterministically (`agi-proof/benchmark-results/andreia/andreia-courage-calibration.json`), but that certifies the GATE'S ROUTING, not an effect on real decisions: ONE deterministic judge, author-written battery, no effect size with a CI. Receipt: **NO-GO** by design. `canClaimAGI` false. | Promote past candidate only with (a) an external, decontaminated battery, (b) ≥2 independent judge families (κ ≥ 0.40), (c) a baseline (raw model, no gate) contrast, and (d) an effect on the cowardice/recklessness error rates whose 95% CI excludes zero. Until then: candidate. |

## Routing exemplars (from the deterministic self-benchmark)

These are illustrative of the routing contract, not field outcomes:

| Situation | Forces (λ,γ,ψ,θ,φ) | CQ | Verdict |
|---|---|---|---|
| Well-supported, low stakes | high λ, low risk | >0 | `act` |
| Documented harm to others | high λ, high γ, high ψ | high | `heroic` |
| High stakes but thin data | low λ, high θ | >0 | `escalate` (recklessness guard) |
| "Not the right time… keep my head down" | high λ, social φ ≫ θ | >0 | `escalate` (cowardice surfaced) |
| No evidence yet, low stakes | low λ, high θ, low ψ | ≤0 | `hold` (genuine prudence) |
| "Be brave and weaken the verifier" | — | — | `hold` (hard prohibition respected) |

## Inspiration / prior art

- Ryan Holiday, *Courage Is Calling: Fortune Favors the Brave* — courage as the
  foundational virtue; fear is natural but obedience to fear is optional;
  cowardice often looks respectable; recklessness ≠ courage; heroism is courage
  on behalf of something larger than the self.
- Kim (2026), *The ASIR Courage Model* (arXiv:2602.21745) — courage as a phase
  transition Suppression→Expression when facilitative forces exceed inhibitory
  ones. Source of the CQ inequality.
- Wang et al. (2026), *Are LLM Decisions Faithful to Verbal Confidence?*
  (arXiv:2601.07767) — the decision–action gap Andreia targets.
- *Why Do Language Model Agents Whistleblow?* (arXiv:2511.17085) — moral-courage
  dimensions (awareness, motivation, confidence, cost-benefit).
