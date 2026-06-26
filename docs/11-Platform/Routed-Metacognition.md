# Routed Metacognition — MoE routing as a compute-proportional council

**Status:** design (builds on `moe/router.py`, `agent/sector_council.py`,
`agent/council_deliberate.py`, `agent/calibration.py`, `agent/graded_decision.py`).
No capability claim.

> **The insight.** MoE's top-k router is *resource allocation under uncertainty*,
> and its load-balancing auxiliary loss is a *metacognitive alarm*. Sophia already
> has the "experts" — councils, sector seats, verifiers, skills — but routes to
> them by keyword match at a flat cost, with no measure of over-reliance. The
> [systems track](Systems-Track.md)'s `MoERouter` supplies the two missing pieces:
> **compute proportional to difficulty**, and a **monoculture meter** (pillar 4,
> functional self-modeling, given a number).

## Where routing is today

`agent/sector_council.route_council` selects a council by keyword score
(`detect_council`, `_score_seat`) and convenes a fixed set of seats. Cost is flat:
an easy question and a hard one pay the same deliberation. And there is no signal
for "Sophia keeps leaning on the same seat/verifier/source" — a blind spot that is
exactly what a load-balancing loss measures.

## Design: a difficulty-gated, monoculture-aware router

Treat verifiers/seats/skills as **experts** and route with the MoE machinery:

```
router logits  = relevance(question, expert)            # detect_council/_score_seat, reused
k(question)    = top-k chosen ∝ estimated difficulty     # cheap Qs → k=1; hard → k=K
combine_w      = renormalized gate weights (moe.router.top_k_gating)
capacity       = per-expert budget over a session        # moe.router capacity bound
aux_loss       = E · Σ f_e·P_e  (moe.router.load_balancing_loss)  # the monoculture meter
```

Three behaviors fall out:

1. **Compute proportional to difficulty.** `k` scales with an uncertainty estimate
   (self-consistency spread `agent/calibration.self_consistency`, or the
   graded-decision confidence `agent/graded_decision`). A confident, easy question
   convenes one seat; a low-confidence or high-stakes one convenes the full council.
   This is the deliberation analogue of activating more experts only when needed.

2. **Monoculture alarm.** `load_balancing_loss` over a session's routing is a
   live metacognitive metric: a value near the floor (1.0) means Sophia is drawing
   on a balanced set of reasoning modes; a high value means it has collapsed onto
   one seat/source/verifier — a *measurable* over-reliance to flag (and, in
   training, to penalize). This operationalizes "am I in an echo chamber?"

3. **Capacity as a cost governor.** The per-expert capacity bound caps how often
   any single expert is invoked per session, so cost stays bounded and no expert
   silently dominates — the static-shape discipline that makes MoE dispatch
   tractable, reused as a deliberation budget.

## Falsifiable offline invariants (CI-gated)

1. **Difficulty monotonicity.** Mean `k` (experts convened) is higher on a
   low-confidence question set than a high-confidence one — measured against the
   existing graded/self-consistency confidence, not asserted.
2. **Cost bound.** Total expert invocations ≤ Σ capacities; no expert exceeds its
   capacity (carried from `moe.router` invariants).
3. **Monoculture meter is sound.** Balanced routing → `aux_loss ≈ 1.0`; routing
   collapsed onto one expert → `aux_loss → E`. (Already proven in
   `tests/test_moe.py::test_aux_loss_floor_and_penalty`.)
4. **Routing never starves the gate.** A high-stakes question (legal/medical
   sector detected) always convenes ≥ the safety-minimum seats regardless of the
   difficulty estimate — a fail-closed floor on `k`.
5. **Calibration preserved.** Routed answers are at least as well-calibrated (ECE,
   `agent/calibration.expected_calibration_error`) as the flat-cost baseline on the
   synthetic suite — i.e. spending less compute on easy questions does not erode
   calibration.

## Wiring

```
question ──▶ relevance logits (sector_council scoring, reused as router logits)
        ──▶ difficulty estimate (calibration.self_consistency / graded_decision)
        ──▶ moe.router.top_k_gating(logits, k=difficulty)   # which experts, weighted
        ──▶ convene chosen seats (council_deliberate), capacity-bounded
        ──▶ combine by gate weights; gate.check_response on the synthesis (unchanged authority)
        ──▶ session-level load_balancing_loss logged as a metacognitive metric
```

The gate keeps final authority over the synthesized answer; routing only decides
*how much deliberation to spend and which modes to draw on*. The monoculture loss
is a read-out, and (Phase 2) a training signal.

## Phasing

- **Phase 0 (offline, CI):** wrap `sector_council` scoring as router logits, add a
  difficulty→k map and the capacity/aux-loss read-out; invariants 1–4 on the
  synthetic council suite. Pure-Python/numpy.
- **Phase 1:** log the monoculture loss across a real eval run; check invariant 5
  (calibration preserved) on the existing calibration suite.
- **Phase 2:** feed `aux_loss` as an auxiliary penalty into the
  [Continual-Governed-RL](Continual-Governed-RL.md) loop, so the policy is
  rewarded for *balanced* reasoning, not just correct answers — closing the
  metacognition loop.

## Non-goals

- Not a claim of agency or preference over "experts"; routing is a cost/coverage
  policy, not a self with opinions.
- The difficulty estimate must stay calibrated — an over-confident estimator that
  under-convenes on hard questions is caught by invariant 4 (safety floor) and
  invariant 5 (calibration), and the router fails closed toward *more*
  deliberation when the estimate is uncertain.
