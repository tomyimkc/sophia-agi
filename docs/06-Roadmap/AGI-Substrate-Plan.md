# AGI-Substrate Plan — the honest, achievable path

**Premise.** This repo (solo author, no GPU fleet, no foundation model) will **not** build
"true AGI." Continual learning without forgetting, grounded causal world models, and safe
recursive self-improvement are **unsolved by everyone** — see the three-tier gap analysis
in `docs/11-Platform/AGI-Missing-Pillars.md` and the honest limits in the README's
"What Sophia cannot do (yet)".

**Reposition.** The valuable, defensible, *achievable* goal is to become the
**verification + reward + measurement substrate the first AGI gets built on** — not the AGI.
Every workstream below serves that, is grounded in modules that already exist, and ships
under the no-overclaim gate (`tools/lint_claims.py`).

> Nothing here is an AGI claim. Each item lists its honest bound and what it does **not** prove.

---

## W1 — Distill regex gates → learned verifiers *(cheap, do first)*

**Why:** today's gates (`agent/verifiers.py`, `agent/moral_ontology.py`,
`agent/constitutional_classifier.py`) are regex/keyword rules. Distilling them into a small
neural classifier tests whether the *concept* generalizes to unseen domains the regex never saw.

**Build:**
1. Use the deterministic gates to label a large **synthetic corpus** (reuse the synthesis
   seam in `selfextend/verifier_synthesis.py`).
2. Train a small classifier (DistilBERT-scale, or a LoRA on a 1–3B open model — low end is CPU-feasible).
3. Evaluate on **held-out domains** disjoint from the training labels.

**Acceptance:** held-out-domain accuracy reported vs the regex baseline, with ECE and an
adversarial-robustness slice; result recorded in the failure ledger.
**Resource:** none → a small GPU (optional).
**Honest bound:** a classifier that generalizes a rule is a *more robust gate*, **not** a step
toward general intelligence.

---

## W2 — Verified auto-promotion *(the novel, provably-safe contribution — highest leverage)*

**Why:** the only honest way to touch "recursive self-improvement" is a **bounded, proved**
slice. You already have every piece:
- governance contract + `schema/golden-vectors.json` (conformance),
- `agent/continual_plasticity.py` (promotion scorecard),
- `agent/formal_verifier.py` + `tests/test_formal_verifier.py` (property checks),
- `selfextend/flywheel.py` (synthesize → validate → cover),
- held-out anti-gaming (`agent/steering/anti_gaming.py`).

**Build:** a synthesized gate auto-promotes to default **iff ALL** hold:
1. passes **every** golden vector (no conformance regression),
2. **strictly dominates** the incumbent on a held-out adversarial set — zero regressions,
3. a `formal_verifier` property proof of a safety invariant (e.g. *"never accepts an
   unsourced claim"*) passes,
4. contamination + leakage checks clean.
Otherwise: **abstain / quarantine**, fail-closed (the existing default).

**Acceptance:** a demo where a candidate gate auto-promotes only when (1–4) pass and is
rejected when any fails; all four predicates enforced in code + a test.
**Resource:** none (offline).
**Honest bound:** this is *self-improvement with a proof* on a narrow gate — **not** an
intelligence explosion, and not architecture self-redesign.

---

## W3 — Process Reward Model over reasoning traces

**Why:** `agent/planner_mcts.py` is a toy (MCTS over a scripted simulator). The real version
scores *partial* reasoning traces for epistemic risk and lets search minimize fabrication
risk at horizon N.

**Build:** train a PRM on the gate's verdicts as **process labels**; wire its score as the
MCTS reward so plans optimize "lowest fabrication risk at horizon N" before acting. Reuse
`selfextend/env_verifier.py` for execution-verified steps.

**Acceptance:** on a held-out trace set, PRM-guided planning lowers end-of-trajectory
fabrication vs greedy, reported with CIs.
**Resource:** modest (small model / CPU-feasible at low N).
**Honest bound:** planning over a *scored* space — **not** open-world planning.

---

## W4 — One small-model RLVR run *(the only honest "filter → engine" step)*

**Why:** today the Conscience Kernel *filters* outputs from frontier models. RLVR turns the
verifier into a **reward** so a policy natively fabricates less. `tools/run_rlvr.py` already
has the offline invariants; the open item is `rlvr-live-run-not-yet-gated-2026-06-21`.

**Build:** rent ~1 GPU-day; RLVR post-train a **1–3B open model** with the Sophia
verifier-reward. Pre-register the falsifiable question: *does the post-trained model fabricate
less on an entity-disjoint held-out set than base / than SFT?*

**Acceptance:** clears `aggregate._is_validated` (≥2 judge families, κ≥0.40, ≥3 runs, CI
excludes 0) on a held-out split + manual spot-check.
**Resource:** **GPU budget (the binding constraint).** This is the gate on the whole tier.
**Honest bound:** a 1–3B model that fabricates less is **not** AGI — it's evidence the reward
shapes behavior at small scale.

---

## W5 — Close the measurement gap with *existing* public benchmarks *(cheap, alongside W1)*

**Why:** world-class *epistemic* benchmarks (SEIB-100, provenance delta), **zero**
general-reasoning ones. You can't measure a gap without the instrument.

**Build:** run the gate/agent on public reasoning sets via `tools/run_external_eval.py` and
`tools/run_gpqa_provenance.py` — **GSM8K (done), MMLU, ARC, GPQA, BIG-Bench-Hard,
SWE-bench-lite**. Report honestly, including the **coverage cost** the gate imposes.

**Acceptance:** a "general-reasoning" results tier in `RESULTS.md`, with the gate's
help-or-hurt trade-off stated plainly.
**Resource:** low (offline / small inference).
**Honest bound:** these measure *base-model* reasoning through the harness — **not** a
Sophia-specific capability claim.

---

## Out of scope (concede honestly)

- **Multimodal grounding** — no sensory surface; out of scope. (A narrow "figure/citation
  provenance" slice is the only ever-on-brand piece, and not now.)
- **Tier-3 unsolved science** (continual learning w/o forgetting, causal world models, safe
  RSI) — **do not claim.** The contribution we *can* make is the **measurement/safety
  harness**: e.g. a catastrophic-forgetting benchmark with provenance (measure, don't solve);
  W2 is the bounded, proved RSI slice.

---

## Sequence

| Order | Workstream | Resource | Ships |
|---|---|---|---|
| 1 | **W1** distill verifiers + **W5** reasoning benchmarks | none / low | robustness + the measuring instrument |
| 2 | **W2** verified auto-promotion | none | the novel, provably-safe contribution |
| 3 | **W4** small-model RLVR | **GPU-day** | the honest filter→engine proof |
| ∞ | **W3** PRM | modest | trajectory-scoring layer, as capacity allows |

**Prerequisite under all of it:** the third-party-validated result (open in the ledger:
`calibration-self-authored-pack-2026-06-22`, `hidden-review-third-party-not-run`). Without one
independent number, none of these land as credible.

**One-line:** stop trying to *be* the AGI; become the verification, reward, and measurement
layer it can't be safely built without.
