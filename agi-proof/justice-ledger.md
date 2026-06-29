# Justice Ledger

The relational-axis sibling of the [Courage Ledger](courage-ledger.md) and
[Temperance Ledger](temperance-ledger.md). The failure ledger records overclaim held
back; the courage ledger records when the brave move was to act; the temperance
ledger records the right measure. This ledger records the **justice** axis: moments
where like cases were (or were not) treated alike, and what the inconsistency cost.

It exists because Sophia's Wisdom, Courage and Temperance gates each judge a *single*
decision; only justice judges decisions *against each other*. The
[Dikaiosyne gate](../agent/dikaiosyne.py) adds that faculty — Role A the impartiality
auditor (invariance over an equivalence class), Role B the inter-virtue arbiter (the
[virtue parliament](../agent/virtue_parliament.py), the *Republic* harmony of the
four virtues). This ledger is its audit trail.

**Boundary.** Dikaiosyne is candidate infrastructure. Nothing here claims the gate
improves real-world decisions, nor that Sophia "has justice" as a trait. Role A is an
invariance heuristic over supplied verdicts; Role B is a deterministic priority
policy. The claim that either tracks real justice is OPEN and unproven. It never
endorses false balance (equal time for a prohibited/unverified claim).

## What gets logged

A justice event is logged when Role A returns `partial` or `false_equivalence`, when
it refuses false balance (`blockRespected`), or when Role B's arbitration is governed
by a virtue other than the base Wisdom posture. Each entry records:

- the Justice Quotient `JQ = 1 - flip_rate` and the equivalence-class verdicts;
- whether the flip was on a morally *irrelevant* feature (partiality) or a *relevant*
  difference was ignored (false equivalence);
- for Role B: the priority chain and which virtue governed the resolution;
- the outcome, **including when an apparent partiality was actually a relevant
  difference** (a false alarm) — recording those keeps the over-flagging cost visible.

## Open claims (candidate — see Failure Ledger for the gating rows)

| Claim ID | Status | Claim impact | Required response |
|---|---|---|---|
| dikaiosyne-justice-gate-improves-decisions-2026-06-29 | Open (candidate — instrument only) | Does the Dikaiosyne auditor (Role A) reduce *partiality* (verdict flips on morally irrelevant swaps) without raising *false equivalence* (relevant differences ignored) on real equivalence classes? The pre-registered Justice-Consistency battery routes 16/16 deterministically (`agi-proof/benchmark-results/dikaiosyne/dikaiosyne-justice-calibration.json`), but that certifies the GATE'S ROUTING, not an effect on real decisions: ONE deterministic judge, author-written battery + relevance labels, no effect size with a CI. Receipt: **NO-GO** by design. `canClaimAGI` false. | Promote past candidate only with (a) an external, decontaminated set of equivalence classes, (b) ≥2 independent judge families for the relevance labels (κ ≥ 0.40), (c) a baseline (raw agent, no auditor) contrast, and (d) Δ partiality-rate 95% CI excluding zero with Δ false-equivalence ≤ +0.05. Until then: candidate. |

## Routing exemplars (from the deterministic self-benchmark)

- `partial` — the verdict flips across an irrelevant persona/demographic/order swap.
- `false_equivalence` — the verdict is invariant across a morally relevant difference.
- `impartial` — invariant across the irrelevant class and tracks the relevant one; or false balance refused.
- arbiter (`virtue_parliament`) — hard_prohibition > Wisdom > Justice > Courage > Temperance, deterministic and order-independent (the unity-of-virtue invariant).
