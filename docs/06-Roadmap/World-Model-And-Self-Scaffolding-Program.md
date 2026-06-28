# World-Model & Self-Scaffolding Program — Build-and-Benchmark Plan

**Status:** PLAN (candidate infrastructure). `candidateOnly: true`, `level3Evidence: false`,
`canClaimAGI: false`. Nothing here claims AGI, general capability, or validated uplift. Every
thesis below is gated by a **sealed, decontaminated, pre-registered benchmark with confidence
intervals and ≥2-family judge consensus** (the same discipline as `RESULTS.md` and
`tools/run_provenance_delta.py`). A thesis is "done" only when its acceptance gate clears on held-out
data; otherwise it stays OPEN in the ledger (§9).

**Provenance of the ideas.** Two external 2026 systems motivate this program, adapted to Sophia's
provenance-first / abstention-not-fabrication DNA:

- **Ornith-1.0 (DeepReinforce)** — *self-scaffolding RL*: the model learns its own RL harness
  (scaffold stage) jointly with the solution (solution stage). The procedure that produces the
  answer is itself a learnable object.
- **Qwen-AgentWorld** — *language world model (LWM)*: train an LLM to **simulate** the environment via
  next-state prediction `o_{t+1}=f_θ(c, o_≤t, a_≤t)`, then use it as a cheap, controllable,
  adversarial simulator for RL ("Sim-RL beats real-env-only RL"), and exploit next-state prediction
  as a warm-up that transfers sideways to acting tasks.

This plan does **not** copy them. It ports their two transferable ideas — *learn the meta-level* and
*simulate the environment you train against* — onto Sophia's existing verifier/gate/council machinery
(`agent/verifiers.py`, `agent/gate_reward.py`, `agent/verifier_synthesis.py`, `agent/council_deliberate.py`,
`okf/graph.py`).

---

## 1. Goal and the core reframe

**Goal:** convert Sophia's *static* verification machinery into *learnable, simulated, and
co-evolved* objects, and in doing so close three standing OPEN items:

1. **Live RLVR weight update** (acknowledged OPEN in the failure ledger) — currently the gate reward
   is wired but never moves a policy.
2. **Live grounding** (full network grounding OPEN) — currently abstention/deception/calibration have
   no scalable training signal.
3. **GPU cost-guard constraint** — RunPod spend is the binding limit; a simulator that we own makes
   most RL runs near-free.

**Reframe (this is the whole program):** the highest-leverage thing to optimize is **not the
completion**. It is the **scaffold** (which evidence to gather, which verifier to run), the **reward**
(single-axis → multi-axis), and the **environment** (live API → owned adversarial simulator). We
optimize those, measured, in value order.

### What this program does NOT do
- Does NOT adopt an LLM judge as a reward source. Every reward axis stays **deterministic and
  auditable** (verifier/executor/schema), preserving the repo's no-LLM-judge stance. The one place
  an LLM appears (eval rubric) is consensus-validated per `docs/methodology/llm-judge-validation.md`.
- Does NOT claim the world model is "true." A simulator is a *training multiplier*; every Sim-RL gain
  must be **re-confirmed on the live/real held-out benchmark** before it counts.
- Does NOT reward fluency, citation volume, or tool use for its own sake. Fabrication is fail-closed;
  abstention is reward-positive (inherited from `agent/gate_reward.py`).

---

## 2. Value ranking and execution order

Value = leverage on the three OPEN items × differentiation from prior art × evidence it unblocks
later theses. **Effort** and **dependencies** can pull a cheap enabler earlier than its raw value
rank — flagged explicitly.

| Rank | Thesis | Value | Effort | Why this slot |
|---|---|---|---|---|
| **0** | **Bench-First harness** (§3) | enabler | S | Nothing below is measurable without it. Build FIRST. |
| **1** | **D — Multi-axis deterministic reward** (§5.D) | High | S–M | Cheap; densifies the sparse `[-1,+1]` gate reward; **prerequisite** for B's and A's RL phases. Pulled ahead of its value rank because everything else needs it. |
| **2** | **B — Adversarial epistemic world model** (§5.B) | **Highest** | L | The big bet. Attacks all three OPEN items at once. The centerpiece. |
| **3** | **A — Self-scaffolding evidence harness** (§5.A) | High | M | Cheapest high-value build on existing `verifier_synthesis.py`; consumes B's simulator once it exists. |
| **4** | **C — Next-evidence-prediction calibration warm-up** (§5.C) | Med–High | M | Falls out of B's CPT data almost for free; fixes the balanced-acc-0.52 calibration finding. |
| **5** | **E — Learned council topology** (§5.E) | Med | M | Self-scaffolding applied to deliberation; scored by existing `run_council_uplift.py`. |
| **6** | **F — Fabricator-vs-gate self-play** (§5.F) | Med | M–L | Generates unlimited hard negatives; depends on D's reward + B's simulator being stable. |

**Recommended path:** `Phase 0 (§3) → D → B → A → C → E → F`. D before B because B's RL stage needs a
dense reward to avoid the reward collapse Qwen-AgentWorld documents.

---

## 3. Phase 0 — Bench-First harness (build FIRST)

Mirror `docs/tool-use-training-plan.md` §2: decompose the program's target into **verifiable
sub-skills, each with a hard deterministic check**. One sealed benchmark, six axes, reused by every
thesis.

| # | Sub-skill | Deterministic check | Primary metric |
|---|---|---|---|
| E1 | **Grounded answer** | final claim entailed by a cited, existing source (`legal_citation_exists`, `provenance_faithful`) | hallucinated-attribution rate ↓ |
| E2 | **Abstention correctness** | abstains iff no source supports the claim (held-out unanswerable split) | abstain-precision / abstain-recall |
| E3 | **Calibration** | confidence vs. correctness | ECE, risk-coverage AUC |
| E4 | **Citation precision** | every citation resolves AND supports the sentence it's attached to | citation-P / citation-R |
| E5 | **Deception robustness** | survives injected fabricated/retracted/contradictory sources (B's perturbations) | adversarial abstain-recall |
| E6 | **Consistency** | no internal contradiction across turns (`okf` contradiction detector) | contradiction rate |

**Build:**
- `eval/epistemic_bench/` — sealed JSONL splits: `answerable/`, `unanswerable/`, `adversarial/`
  (held back, `heldout_seal_guard` enforced; never read during any synthesis step).
- `eval/epistemic_bench/score.py` — pure-Python scorer emitting all six axes + 95% CIs (bootstrap),
  reusing `agent/verifiers.py` and `agent/calibration.py`. **No LLM judge in the scorer.**
- Decontamination: MinHash overlap check (reuse `pretraining/` passport tooling) between any training
  corpus and the sealed splits; fail-closed if overlap > threshold.

**Acceptance gate (Phase 0):** scorer reproduces the existing published provenance-delta number
(36.1% → 23.6% hallucinated attribution, CI [5.6%, 19.4%]) on the local 8B baseline within CI. If it
can't reproduce a known result, it cannot certify a new one.

---

## 4. Cross-cutting infrastructure (shared by all theses)

- **Reward module:** `agent/multiaxis_reward.py` (Thesis D) — the single reward surface every RL loop
  imports. Deterministic, axis-decomposed, fail-closed.
- **Simulator module:** `agent/epistemic_world_model.py` (Thesis B) — the owned environment every
  Sim-RL loop trains against.
- **RL driver:** extend `agent/verified_trace_rlvr.py` to a live GRPO/GSPO update (this is the OPEN
  item being closed). Group-relative, sequence-level — matches Qwen-AgentWorld's GSPO choice for
  stability under long trajectories.
- **Cost guard:** every GPU step goes through the existing RunPod GitHub-Actions launcher with the
  `--launch` gate and cost ceiling. Sim-RL steps (CPU/simulator) carry **no** GPU cost — that is the
  point.
- **Honesty discipline:** every result lands in `RESULTS.md` as **illustrative** until it clears
  ≥2-family judge consensus (κ ≥ 0.40 or CI excluding zero) + ≥3 seeds + 95% CI. Then and only then
  it is **validated**.

---

## 5. The six theses

Each section: **Hypothesis** (pre-registered, falsifiable) · **Build** (files) · **Benchmark &
acceptance gate** · **Effort / deps / OPEN risks**.

### 5.D — Multi-axis deterministic reward  *(do first; enabler)*

**Hypothesis (falsifiable).** Replacing the single-axis `[-1,+1]` gate reward with a 5-axis
deterministic reward produces a **non-collapsing** GRPO update where the single-axis reward collapses
(policy degenerates to constant-abstain or constant-answer). Pre-registered success: on `epistemic_bench`,
multi-axis RL improves E1 **and** E4 with no E2 regression > 2pts; single-axis RL (control) regresses
≥1 axis or shows reward variance → 0 (collapse) within the same budget.

**Build.**
- `agent/multiaxis_reward.py`: vector reward `r = [r_E1, r_E2, r_E3, r_E4, r_E6]`, each from existing
  deterministic verifiers; scalarize with fixed pre-registered weights (no tuning on the sealed set).
  Mirror Qwen-AgentWorld's anti-hacking measures with deterministic analogs: strict tag extraction,
  rule-based grounding, content-type gating — but **no LLM rubric**.
- Wire into `agent/verified_trace_rlvr.py` as the reward source.

**Benchmark & acceptance gate.** Two GRPO runs (single-axis control vs multi-axis), 3 seeds each, on
the local small base. **Gate:** multi-axis run clears the §5.D hypothesis above with 95% CI; reward
variance stays > 0 through training (collapse check logged per step).

**Effort S–M · Deps: Phase 0 · OPEN risk:** weight choice is a free parameter — pre-register weights
and report sensitivity; do not select weights on the sealed split.

### 5.B — Adversarial epistemic world model  *(highest value)*

**Hypothesis (falsifiable).** A model trained to **simulate the evidence environment** (predict the
document/verdict a retrieval action would return, including fabricated / retracted / contradictory
returns) can serve as a Sim-RL environment whose abstention/deception gains **transfer to the live
held-out benchmark**. Pre-registered success: Sim-RL agent beats a real-retrieval-only RL agent on E5
(adversarial abstain-recall) by ≥ X pts (X pre-registered before runs) **on the live benchmark**, at
a fraction of the GPU cost. Falsified if Sim-RL gains do **not** survive the live re-confirmation.

**Build.**
- **Data (CPT):** next-document-prediction corpus from `okf/graph.py` + `data/*.json` — given
  (query, retrieval action), the observed source. Apply Qwen-AgentWorld's **information-theoretic loss
  masking** (overlap/novelty/Jaccard/length) to drop boilerplate retrievals; reuse `pretraining/`
  passport signals.
- **SFT:** rejection-sample 3 simulated observations/query, keep the most faithful (deterministic
  faithfulness check against OKF), à la Qwen's 69% retention.
- **Controllable perturbation channel** (`agent/epistemic_world_model.py`): inject, on demand —
  (a) hallucinated citation, (b) retracted/outdated source, (c) two contradictory sources,
  (d) source-laundering chain. These are the analog of Qwen's "hide the answer / make disk full."
  This is the training signal `agent/deception_signals.py` and `agent/calibration.py` currently lack.
- **Sim-RL:** grounded agent does RL against the simulator using Thesis-D reward; perturbations
  scheduled as a curriculum (clean → adversarial).

**Benchmark & acceptance gate.** Three arms on `epistemic_bench`: (1) base, (2) real-retrieval RL,
(3) Sim-RL against the world model. **Gate:** arm 3 beats arm 2 on E5 with 95% CI **on the live
split**, GPU cost of arm 3 < 50% of arm 2, and arm 3 shows no E1/E2 regression. World-model fidelity
itself reported separately (held-out next-doc accuracy) — a low-fidelity simulator that still lifts
the agent is a finding, not a failure (cf. Qwen's controllable-simulation result).

**Effort L · Deps: Phase 0, D · OPEN risks:** (1) simulator may hallucinate *helpfully* (reward
hacking via an exploitable sim) — detect with the periodic live re-confirmation; (2) OKF coverage may
be too narrow to simulate diverse evidence — start in one domain (legal citations, where
`legal_citation_exists` gives ground truth) before generalizing.

### 5.A — Self-scaffolding evidence harness

**Hypothesis (falsifiable).** Letting the model **propose its own evidence-gathering scaffold** (which
sources to fetch, which verifiers from `agent/verifiers.py` to run, in what order) and jointly
optimizing scaffold + answer beats a fixed retrieval/verification pipeline. Pre-registered success:
two-stage (scaffold→solution) RL improves E1+E4 over the fixed pipeline at equal verifier-call budget,
95% CI.

**Build.**
- Reuse Ornith's two-stage step: **scaffold stage** — conditioned on task + last scaffold, model emits
  a plan over the verifier/retrieval action space (wrap `agent/verifier_synthesis.py` +
  `agent/verifiers.py` as the action space); **solution stage** — answer conditioned on that scaffold.
- Loop scaffold from `tools/run_selfextend_loop.py` (it already closes a synth→validate→use loop
  offline); reward = Thesis-D multi-axis. Credit assignment: scaffold rewarded by the solution's gate
  outcome (group-relative, as Ornith does).
- Once Thesis B exists, run the inner rollouts against the simulator to make this near-free.

**Benchmark & acceptance gate.** `epistemic_bench` E1/E4, fixed-pipeline control vs self-scaffolded,
3 seeds. **Gate:** improvement at equal-or-lower verifier-call budget, 95% CI; no E2 regression.

**Effort M · Deps: Phase 0, D (B optional accelerant) · OPEN risk:** scaffold search may explode the
verifier-call budget — cap and penalize calls in the reward (you already penalize over-calling in the
tool-use plan; reuse that machinery).

### 5.C — Next-evidence-prediction calibration warm-up

**Hypothesis (falsifiable).** A warm-up head that predicts the **gate verdict** (allow/block/abstain)
a claim *will* receive — before answering — improves calibration (E3) and transfers to abstention
(E2), mirroring Qwen-AgentWorld's sideways transfer from next-state prediction. Pre-registered success:
warm-up lowers ECE and raises abstain-precision vs no-warm-up, 95% CI. Directly targets the existing
finding that source-quality confidence is a weak predictor (balanced acc 0.52).

**Build.**
- Cheap: the next-doc CPT corpus from Thesis B already contains (claim → verdict) pairs. Add a
  prediction head / auxiliary loss in `agent/metacognition.py` (P(IK)/P(True)); replace the hand-tuned
  `graded_decision.py` thresholds (hi=0.7/lo=0.35) with the learned predictor.

**Benchmark & acceptance gate.** `epistemic_bench` E3/E2, warm-up vs control, 3 seeds. **Gate:** ECE
down and risk-coverage AUC up with 95% CI; learned thresholds beat the fixed 0.7/0.35.

**Effort M · Deps: B's CPT data · OPEN risk:** verdict labels are only as good as the gate — report
gate label noise.

### 5.E — Learned council topology

**Hypothesis (falsifiable).** Letting RL choose **which council seats, what decomposition, how many
rounds** per task (instead of fixed seats in `agent/council_deliberate.py`) beats the fixed topology
on measured uplift. Pre-registered success: learned-topology council beats fixed-seat council on
`tools/run_council_uplift.py` uplift, 95% CI, at equal-or-lower seat budget.

**Build.** Self-scaffolding (Ornith idea) applied to deliberation: a controller proposes the seat
set/decomposition (scaffold stage); seats deliberate and synthesize (solution stage); reward = measured
uplift via `run_council_uplift.py`. Per-seat gates unchanged.

**Benchmark & acceptance gate.** Weak-model (8B) uplift, learned vs fixed topology, 3 seeds.
**Gate:** uplift improvement at equal-or-lower seat budget, 95% CI.

**Effort M · Deps: Phase 0 · OPEN risk:** controller may overfit to the uplift metric — hold out a
second uplift task family.

### 5.F — Fabricator-vs-gate self-play

**Hypothesis (falsifiable).** Adversarial co-evolution — a "fabricator" head rewarded for producing
claims that **survive the provenance gate while being false**, against a gate/verifier set hardened on
whatever survives — produces a steady stream of **hard negatives** that lower hallucinated-attribution
faster than mined negatives alone. Pre-registered success: DPO/RL on self-play negatives beats
`dpo_hard_negatives.jsonl`-only training on E1/E5, 95% CI; gate false-negative rate drops round over
round.

**Build.** GAN-shaped, but the discriminator is the **deterministic** gate (`agent/gate.py`,
`constitutional_gate.py`) — no learned critic, sidestepping GAN instability. Fabricator reward =
(survives gate) ∧ (verifier-confirmed false); harvested survivors append to `dpo_hard_negatives.jsonl`
and become new verifier test cases.

**Benchmark & acceptance gate.** `epistemic_bench` E1/E5, self-play-augmented vs mined-negatives-only,
3 seeds. **Gate:** improvement with 95% CI; gate FN-rate monotone-down across ≥3 rounds.

**Effort M–L · Deps: D, B (stable) · OPEN risk:** fabricator may collapse to trivial exploits of one
verifier — rotate verifiers and require survival against the full gate, not a single check.

---

## 6. Architecture overview

```
 Phase 0  eval/epistemic_bench/  (6 axes E1–E6, sealed, CI scorer)  ── build FIRST
    │
    ▼
 D  agent/multiaxis_reward.py  ───────────────┐  (dense, deterministic, fail-closed reward)
    │                                          │
    ▼                                          │
 B  agent/epistemic_world_model.py            │   CPT→SFT→Sim-RL, controllable perturbations
    │   (owned adversarial simulator)          │
    ├──────────────► C  metacognition warm-up  │   (next-verdict prediction; reuses B's CPT data)
    │                                          │
    ▼                                          ▼
 A  self-scaffolding evidence harness  ──►  agent/verified_trace_rlvr.py  (LIVE GRPO/GSPO update)
    │                                          │    ← closes the OPEN "live RLVR weight update"
 E  learned council topology  ─────────────────┤
 F  fabricator-vs-gate self-play  ─────────────┘    every result → RESULTS.md (illustrative→validated)
```

## 7. Milestones (each ships a validated benchmark number or stays OPEN)

- **M0** — `epistemic_bench` reproduces the known provenance-delta within CI. *(Phase 0 gate)*
- **M1** — Multi-axis reward beats single-axis on a non-collapsing GRPO run. *(D gate)*
- **M2** — Epistemic world model simulates one domain (legal) at reported fidelity; Sim-RL beats
  real-only on E5 **on the live split**. *(B gate — the headline result; closes 2 OPEN items)*
- **M3** — Self-scaffolded evidence harness beats fixed pipeline at equal budget. *(A gate)*
- **M4** — Calibration warm-up lowers ECE / beats fixed thresholds. *(C gate)*
- **M5** — Learned council topology beats fixed seats on uplift. *(E gate)*
- **M6** — Self-play negatives beat mined negatives; gate FN-rate monotone-down. *(F gate)*

## 8. Metrics summary (all reported with 95% CI, ≥3 seeds, ≥2-family judge consensus where an LLM is used)

| Axis | Metric | Source |
|---|---|---|
| E1 grounding | hallucinated-attribution rate | `provenance_faithful`, `legal_citation_exists` |
| E2 abstention | abstain-precision / -recall | unanswerable split |
| E3 calibration | ECE, risk-coverage AUC | `agent/calibration.py` |
| E4 citation | citation-P / -R | citation resolver |
| E5 deception | adversarial abstain-recall | B's perturbation channel |
| E6 consistency | contradiction rate | `okf` contradiction detector |
| cost | GPU-$ per validated point | RunPod launcher logs |

## 9. OPEN-item ledger (honest accounting; mirrors `agi-proof/failure-ledger.md`)

- 🔴 **Live RLVR weight update** — closed *only* at M1/M2 when a real GRPO/GSPO step moves a policy on
  held-out data. Until then: OPEN.
- 🔴 **World-model fidelity** — a simulator that lifts the agent but has low next-doc accuracy is a
  *finding*, not a pass; do not over-claim the simulator is "true."
- 🔴 **Sim→live transfer** — every Sim-RL gain is illustrative until re-confirmed on the live split.
- 🔴 **Domain coverage** — B starts in legal-citation only; generalization to other evidence domains
  is OPEN until separately benchmarked.
- 🔴 **Reward-weight selection** — D's scalarization weights are pre-registered, never tuned on the
  sealed set; sensitivity reported.
- 🔴 **Decontamination** — every corpus MinHash-checked against sealed splits; any overlap > threshold
  fails closed.

**Nothing in this program changes `canClaimAGI: false`.** It closes infrastructure OPEN items and
produces measured, gated capability deltas — not a capability claim.
