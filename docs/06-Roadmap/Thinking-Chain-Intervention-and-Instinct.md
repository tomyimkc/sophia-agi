# Thinking-Chain Intervention & an Instinct System

> **Status:** research brainstorm + one falsifiable prototype (`reasoning/instinct_gate.py`).
> `candidateOnly: true`, `canClaimAGI: false`. This note proposes theory and a *model* of it;
> it does not assert a measured capability. Graduating any claim here to VALIDATED requires the
> repo's no-overclaim gate (≥2 judge families, ≥3 seeds, CI, pre-registration). See
> [`agi-proof/measurement-thesis.md`](../../agi-proof/measurement-thesis.md).

## 0. The operator's thesis, stated precisely

Two intuitions, restated as falsifiable claims:

1. **Injection.** "AI nowadays operates inside a *thinking chain* — can we inject that chain
   and make it change its mind?" → *Is there a control point, during generation, where an
   external (or self-) signal can flip the model's conclusion rather than only re-prompting it
   from scratch?*
2. **Instinct.** "When the AI senses something is wrong it should change its mind
   **immediately**, instead of going the wrong way and then trying to fix it." → *A fast,
   always-on error reflex should trigger a discrete **re-route/backtrack** early, not a slow
   forward patch at the end.*

The second is the load-bearing one. The first is the *mechanism*; the second is the *policy*.
This repo already has most of the policy machinery — it is missing the **reflex** and the
**benchmark that says when the reflex is worth trusting**. That gap is what this note fills.

---

## 1. What the literature actually says (grounding, not vibes)

Three findings from current work bear directly on the thesis, and two of them are
*uncomfortable* — which is exactly why they matter here.

**(a) The chain is often not the cause.** Chain-of-thought text frequently fails to
faithfully reflect the computation that produced the answer; linear probes can predict the
answer *before* the explanation is generated, and models can be trained to emit irrelevant
reasoning yet stay accurate — i.e. much CoT is *post-hoc rationalisation*, and it carries
confirmation bias. ([CoT in the wild is not always faithful](https://arxiv.org/pdf/2503.08679);
[CoT is not explainability](https://www.alphaxiv.org/overview/2025.02v1);
[Confirmation bias in CoT](https://arxiv.org/pdf/2506.12301).) *Implication:* you cannot trust
the chain's self-report that "something is wrong." A reflex has to read a **separate** signal
(activations, agreement, a verifier) — not the prose.

**(b) Intrinsic self-correction is marginal and can hurt.** The canonical result is that LLMs
**cannot reliably self-correct reasoning** without external feedback, and accuracy sometimes
*drops* after a self-correction pass — the "accuracy–correction paradox" where a forward "fix"
turns a right answer wrong. ([LLMs Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798);
[Decomposing LLM Self-Correction](https://arxiv.org/pdf/2601.00828).) *Implication:* the
operator's instinct is correct — "go forward and patch it" is the weak move. The win has to
come from **catching the error early and re-routing**, not from a late repair pass.

**(c) You really can inject mid-stream — two ways.**
   - *Token-level:* training/decoding with **backtrack markers** ("Wait", "Alternatively",
     reflection tokens, an explicit "I made an error" token at the end of a bad step) measurably
     improves recovery. ([Self-Backtracking](https://arxiv.org/pdf/2502.04404);
     [Learn to reason from trial and error](https://arxiv.org/pdf/2510.26109).)
   - *Activation-level:* **activation steering / representation engineering** adds a contrastive
     "direction" vector to the residual stream at a layer *during generation* — changing
     behaviour with no weight update, cheaply, at inference time. This is how you literally
     "inject the chain and change its mind." It is established for refusal/truthfulness/safety —
     **but** steering vectors are unreliable: their effect is geometry-dependent and the linear
     approximation breaks at strength. ([Contrastive Activation Engineering](https://arxiv.org/pdf/2505.03189);
     [Unreliability of steering vectors](https://arxiv.org/pdf/2602.17881).)

**Net:** injection is real (both kinds). Self-correction-by-default is not the answer. The open
design problem is the *trigger* — a reflex that fires early, on a trustworthy non-prose signal,
and whose false alarms don't cost more than its catches. That is precisely an **instinct
system**.

---

## 2. The instinct system (architecture)

Think **two systems**, in the dual-process sense, but the repo's idiom keeps it deterministic
and fail-closed.

```
                 reasoning chain  s_1 → s_2 → s_3 → … → s_n → answer
                       │     │     │            (System 2: slow, serial deliberation)
   ┌───────────────────┴─────┴─────┴──────────────────────┐
   │   REFLEX BUS  (System 1: fast, always-on, parallel)   │
   │   cheap detectors, one scalar "wrongness" each:       │
   │     • self-consistency disagreement   (agent/calibration.py)
   │     • verifier / type mismatch         (agent/*_verifier.py)
   │     • provenance/grounding loss        (okf.revise, agent/grounded_gate.py)
   │     • activation probe direction       (steering-vector dot-product)
   │     • novelty/surprise spike           (okf/surprise_consolidation.py)
   └───────────────────────────┬──────────────────────────┘
                               aggregate → scalar + confidence
                                          │
                          INTERRUPT CONTROLLER (graded)
                                          │
        ┌──────────┬──────────────┬──────────────┬─────────────┐
     continue    reroute       backtrack       escalate       abstain
   (signal low) (fresh path)  (to pre-error)  (ko / human)  (fail-closed)
```

Design commitments (each maps to something the repo already enforces):

- **The reflex reads signals, not prose.** Because the chain is unfaithful (§1a), detectors
  consume activations / agreement / verifier verdicts / grounding state — never the model's own
  "I think this is wrong" sentence.
- **The response is a *discrete operator*, not a forward patch.** The five verdicts are the
  repo's existing conscience vocabulary (`allow|hedge|abstain|escalate|block`, see
  `agent/consequence_gate.py`). "Change its mind" = `reroute`/`backtrack`; "go forward and fix"
  is deliberately *not* an option.
- **Oscillation is escalation, never silence.** If re-routing keeps revisiting a failing belief
  state, that's a **ko** — terminate with `escalate` (needs a human or new information), never a
  silent loop and never a forced pick. This rule already exists in
  [`reasoning/consequence/ko_detector.py`](../../reasoning/consequence/ko_detector.py) and
  [`reasoning/consequence/revise_loop.py`](../../reasoning/consequence/revise_loop.py); the
  instinct system reuses it verbatim.
- **The decision is graded, not binary.** `(reflex_score, confidence)` maps onto a confidence
  curve exactly like [`agent/graded_decision.py`](../../agent/graded_decision.py) — a strong
  signal commits to `reroute`; a weak one `hedge`s or keeps going.
- **Belief edits are revisions, not overwrites.** When the re-route changes a *belief* (not just
  a step), do it AGM-style — give up the weaker belief with its cascade, abstain on ties — via
  [`agent/belief_revision_policy.py`](../../agent/belief_revision_policy.py). This is the
  "change its mind" operation on the belief graph, and it is already non-destructive.

So the instinct system is **~80% assembly of existing parts** + **two new pieces**: (1) the
fast reflex bus (a portfolio of cheap detectors with a calibrated aggregator), and (2) the
*interrupt controller* that turns a reflex spike into one of the five operators early in the
chain.

---

## 3. The core question: *when is the reflex worth trusting?*

A reflex that fires too eagerly throws away good chains; one that fires too late is useless.
This is the real research risk, and it is benchmarkable **today, offline**, in the repo's
`reasoning/` idiom. That is what [`reasoning/instinct_gate.py`](../../reasoning/instinct_gate.py)
does: a pure-stdlib, seeded, planted-ground-truth *model* of three policies over the same
trajectory distribution —

| policy | what it does | analogue |
|---|---|---|
| `commit` | run to the end, never intervene | today's default |
| `late` | run to the end, then one self-correction pass (fixes wrong w.p. `p_fix`, **breaks** right w.p. `p_break`) | intrinsic self-correction (§1b) |
| `instinct` | reflex fires the first step its noisy signal crosses `tau` → **backtrack & re-route**; ko-bounded | the proposed system |

### Verdict (seed 1234, 4000 trials — reproduce with `python reasoning/instinct_gate.py --run`)

```
good reflex (snr=3): commit 0.533   late 0.552   instinct 0.726   (escalate 0.13)
poor reflex (snr=0): commit 0.533                instinct 0.526   (false-interrupt 0.04)
break-even reflex SNR (d′) = 1.0   [stable across seeds/trials with a noise-margin estimator]
```

Four falsifiable hypotheses, all confirmed in `--self-test` / `tests/test_instinct_gate.py`:

- **H1 — Compounding ("don't plough ahead").** Under `commit`, the longer an error runs
  uncorrected the lower the final correctness (monotone in error lateness; MC tracks the
  closed form `0.93·0.80^(remaining)` to <0.01). *This is the formal content of the operator's
  intuition.*
- **H2 — Early ≫ Late.** With a usable reflex, `instinct` (0.73) beats `late` self-correction
  (0.55) decisively — and `late` is barely better than doing nothing (commit 0.53), matching §1b.
- **H3 — The ceiling is the reflex, not the policy.** There is a **finite, positive
  break-even SNR = 1.0** (a detectability d′ of 1.0 between "wrong" and "fine"). Below ~0.5
  the reflex *under-performs `commit`* (snr=0: 0.526 < 0.533); the 0.5–1.0 band is within
  Monte-Carlo noise (no reliable gain — which is *why* the break-even estimator requires a
  margin beyond noise, otherwise it reports an unstable 0.5). The whole gain is bounded by
  the reflex's ROC — the direct analogue of `deliberation_roofline`'s result that the ceiling
  is the *verifier's* SNR, not compute. **This is the honest boundary: an instinct is only as
  good as the reflex behind it, and a bad reflex is worse than none.**
- **H4 — Bounded.** The ko guard makes re-route terminate in a clean `escalate` (bounded
  re-routes), never an endless patch-forward loop.

The negative result (H3) is the most valuable part — it tells you *not* to ship a reflex whose
detector SNR you haven't measured, and it gives a concrete bar (SNR > break-even) for when the
"change its mind" instinct is a net win.

### 3a. Measuring the first real reflex against that bar

H3 raises exactly one question for any candidate reflex: *does its separation between
"chain is wrong" and "chain is fine" clear the break-even bar?*
[`reasoning/instinct_reflex_eval.py`](../../reasoning/instinct_reflex_eval.py) is the go/no-go
**harness** that answers it. It wires the first concrete reflex — **self-consistency
disagreement** (`agent/calibration.py`, `1 − agreement`) — to the planted belief-revision
oracle (`eval/belief_revision/belief_revision_50_v1.jsonl`) and reports a **d′** (the same
unit as `instinct_gate`'s SNR) and **AUC**:

```
self-consistency reflex (synthetic sampler, N=50):  d′ 0.96   AUC 0.73   clears d′=1.0? NO
competence sweep:  comp 0.45 → d′ 0.74   0.62 → d′ 0.96   0.80 → d′ 1.54   0.95 → d′ 1.74
no-signal control:  d′ 0.06   AUC 0.53   (the harness manufactures no separation)
```

**Honest finding (candidate):** self-consistency disagreement is a *directionally real* reflex
(errored items score higher, AUC 0.73, control collapses to chance) but at a *moderately
competent* reasoner it sits **just under** the break-even bar (d′ 0.96 < 1.0) — it only clears
once the underlying reasoner is fairly competent (d′ 1.54 at competence 0.80). So self-
consistency alone is a *borderline* first reflex; the reflex bus (§2) will likely need a
second, independent detector (verifier mismatch / grounding loss) to clear the bar reliably at
realistic competence.

**Critical caveat:** the d′ above is the *synthetic sampler's*, present to validate the harness
end-to-end and demonstrate the go/no-go — it is **not** a measured claim about a real model. The
sampler is one pluggable function; a real model drops in via
`run_reflex_eval(..., sampler=my_model_sampler)`, and that real number is the gated next step
below, not asserted here. The *harness* is the deliverable; `canClaimAGI` stays `false`.

**Real-model run — attempted, blocked (honest record).** The real sampler exists
([`tools/run_reflex_openrouter.py`](../../tools/run_reflex_openrouter.py), OpenAI-compatible,
key from env only). A run against OpenRouter (gpt-4o-mini + a second family) was attempted and
**blocked at the provider**: the key authenticated (a valid `user_id` resolved) but every
request returned `HTTP 403 "prohibited due to a violation of provider Terms Of Service"` — the
sandbox's policy-enforcing egress IP is ToS-blocked by the provider, independent of key or
model. So **no real-model d′ exists yet** (the synthetic number is the only one, and is not a
model claim). The attempt also surfaced and fixed a harness-honesty bug: the tool originally
folded API failures into the answer stream, which made an all-failed run masquerade as
`base_error=1.0` (the model "getting everything wrong") — exactly the fabricated-metric trap the
[rlvr-harness-traps](../../.claude/skills/rlvr-harness-traps/SKILL.md) discipline warns about.
The tool now **fails loud** on any 4xx and never records a failed call as data. To get the real
number: run the tool from a non-blocked network (your own machine) with a rotated key in
`OPENROUTER_API_KEY` — `python tools/run_reflex_openrouter.py --model <m> --samples 5 --yes`.

---

### 3b. Two detectors beat one — and *independence* is the whole game

§3a said self-consistency is borderline and the bus needs a second, independent detector.
[`reasoning/instinct_fusion.py`](../../reasoning/instinct_fusion.py) builds and measures that.
Detector **A** is self-consistency (keys on *uncertainty*); detector **B** is a real `okf`
grounding-closure check (`okf.revise`/`claims_to_abstain`) that fires when a proposed abstain
set wrongly includes a claim that still has live grounding — it is deliberately *partial*
(blind to under-abstention), so it catches the **confident structural errors A misses**, not
the oracle.

```
A self-consistency : d′ 0.87   AUC 0.67   clears d′=1.0? NO
B okf-grounding     : d′ 0.97   AUC 0.66   clears d′=1.0? NO
A+B fused (z-sum)   : d′ 1.86   AUC 0.86   clears d′=1.0? YES     ρ(A,B) ≈ −0.22
```

**Finding:** *neither detector clears the bar alone, but their fusion does* — two weak
reflexes make one good one. And it is governed by a clean law (verified MC vs closed form):
for two detectors of detectability `d_A,d_B` with correlation `ρ`,
**`d′_fused = (d_A + d_B) / √(2 + 2ρ)`** — which is `√(d_A²+d_B²)` (quadrature) at `ρ=0` and
collapses to the mean (no gain) at `ρ=1`. So a second detector helps **only to the extent it
is independent**: at d′≈0.96 each, fusion clears the bar for any correlation below
**ρ* ≈ 0.84**. The complementarity is explicit — among the errors A lets through, B still
separates at AUC 0.79. (Honest scope: B is real `okf` and A is real self-consistency, but the
*answer distribution* is a seeded synthetic reasoner over the real graphs; the d′s validate
the fusion law and bus design, not a real model — the real-model fusion d′ is still gated.)

**Architectural payoff:** the reflex bus in §2 should fuse detectors chosen for *independence
of failure mode* (uncertainty-based + structure-based + provenance-based), not redundancy.
A pile of correlated detectors buys almost nothing; two uncorrelated borderline ones clear the
bar.

## 4. From model to measured claim (pre-registration sketch)

To graduate this from `candidateOnly` to a real eval under the Instrumented Evaluation Contract
([`agi-proof/measurement-thesis.md`](../../agi-proof/measurement-thesis.md)):

- **Datasets that already exist here:**
  [`eval/belief_revision/belief_revision_50_v1.jsonl`](../../eval/belief_revision/belief_revision_50_v1.jsonl)
  (transitive retraction — the "change its mind" ground truth) and
  [`eval/consequence_cascade/consequence_cascade_40_v1.jsonl`](../../eval/consequence_cascade/consequence_cascade_40_v1.jsonl)
  (when a re-route should `escalate`). These give planted oracles for re-route *correctness*.
- **The real reflex to wire first:** self-consistency disagreement
  (`agent/calibration.py`) — label-free, cheap, already built. The measurement harness for it
  already exists — [`reasoning/instinct_reflex_eval.py`](../../reasoning/instinct_reflex_eval.py)
  (§3a) — and reports d′/AUC against the break-even bar. The gated step is to swap its synthetic
  `sampler` for a real multi-sample model run and read off the *real* d′ before trusting the
  reflex. §3a/§3b show self-consistency is borderline alone but a **second independent detector**
  (real `okf` grounding-closure) fuses with it to clear the bar — so the gated real-model run
  should sample *both* detectors and report fused d′.
- **Primary metric:** final-answer correctness uplift of `instinct` over both `commit` and
  `late`, with token-cost as a co-primary (the re-route tax must be reported, not hidden).
- **Pass bar (no-overclaim):** ≥2 independent judge families, judge ≠ subject, ≥3 seeds, 95% CI
  excluding zero, pre-registered MDE. Until then every number stays labelled `candidate`.
- **Activation-steering arm (stretch):** if a local model is available, build a contrastive
  "error-direction" steering vector and test the *injection* claim directly — does adding it at a
  mid-layer flip a wrong-path chain? Report it against the steering-unreliability caveats (§1c),
  never as a guaranteed control.

---

## 5. Honest risks / failure-ledger candidates

1. **Reflex SNR unknown on real traces.** The whole thesis is gated on H3's break-even; we have
   *modelled* it, not *measured* it on a real detector. Until measured, "instinct helps" is
   `candidate`.
2. **Faithfulness trap.** If a detector secretly reads the (unfaithful) chain prose, it will
   look good and generalise badly. Detectors must be auditable as prose-independent.
3. **Steering-vector brittleness.** Activation injection is geometry-dependent and breaks at
   strength; it cannot be the *only* control path.
4. **Re-route cost can dominate.** At high reflex sensitivity, false-interrupt and escalate
   rates climb (see the SNR sweep); the token tax is real and must stay a co-primary metric.

---

## 6. TL;DR

- "Inject the chain to change its mind" is **real** — token-level (backtrack markers) and
  activation-level (steering vectors) — with caveats.
- "Change its mind early instead of patching forward" is the **right** instinct: the model
  confirms early re-route (0.73) ≫ late self-correction (0.55 ≈ commit 0.53, i.e. doing
  nothing), matching the published weakness of intrinsic self-correction.
- The hard part is the **reflex**: there is a measurable break-even SNR (d′ = 1.0) below which
  an instinct *hurts*. Ship the reflex only after its ROC is measured against that bar — and the
  measurement harness for that go/no-go already exists (`reasoning/instinct_reflex_eval.py`).
- First reflex measured (synthetic, harness-validation): **self-consistency disagreement is
  borderline** — d′ 0.96 at moderate competence, just under the bar.
- **Fusion result:** a second *independent* detector (real `okf` grounding-closure) fused with
  self-consistency clears the bar (d′ 1.86) though neither does alone — governed by
  `d′_fused=(d_A+d_B)/√(2+2ρ)`, so the bus must pick detectors for **independence, not
  redundancy**. Real-model fused d′ is the gated next step.
- This repo already has the *policy* substrate (ko-escalate, graded decision, AGM belief
  revision, consequence gate). The missing pieces are the **reflex bus** and the **interrupt
  controller** — and a no-overclaim eval, datasets for which already exist.

**Sources:**
[CoT not always faithful](https://arxiv.org/pdf/2503.08679) ·
[CoT ≠ explainability](https://www.alphaxiv.org/overview/2025.02v1) ·
[Confirmation bias in CoT](https://arxiv.org/pdf/2506.12301) ·
[LLMs cannot self-correct yet](https://arxiv.org/abs/2310.01798) ·
[Accuracy–correction paradox](https://arxiv.org/pdf/2601.00828) ·
[Self-backtracking](https://arxiv.org/pdf/2502.04404) ·
[Reason from trial and error](https://arxiv.org/pdf/2510.26109) ·
[Contrastive activation engineering](https://arxiv.org/pdf/2505.03189) ·
[Unreliability of steering vectors](https://arxiv.org/pdf/2602.17881)
