# From Un-0 to sophia-agi — oscillatory / dynamical-systems ideas for an LLM epistemics engine

**What this is.** You pointed at `unconv-ai/Un-0` as inspiration. This document reports what
Un-0 actually is (read from its source, not its tagline), surveys the real research line it
belongs to, and proposes five concrete directions that cross-pollinate that line into
`sophia-agi` — an LLM provenance / verification / calibrated-abstention system. It is grounded
in (a) Un-0's own code, (b) a literature survey of oscillatory/dynamical-systems computation,
and (c) a fresh read of the 7 commits your `main` gained since we last worked, so nothing here
re-proposes something you just built.

**Honesty boundary.** These are research *bets*, each with its single strongest objection
named. Nothing here has been run. Citations give title + venue as surfaced in a web search;
arXiv IDs are included only where the survey explicitly saw them and are marked accordingly —
treat any ID as "to confirm before you cite it in a paper."

---

## 1. What Un-0 actually is (from the code, not the marketing)

- **Mechanism:** an *image generator* — not an LLM — built on **Kuramoto dynamics**. A
  population of coupled phase oscillators evolves under `dθ/dt = ω + cos(θ)·(sin(θ)@Kᵀ) −
  sin(θ)·(cos(θ)@Kᵀ)` (the standard Kuramoto coupling, expanded to two matmuls), integrated by
  `torchdiffeq.odeint` (euler/rk4, ~10 steps). The settled phase state decodes to an image. No
  diffusion schedule, no adversary, no iterative denoising.
- **Training signal:** a contrastive **drift loss distilling a self-supervised DINO teacher**
  (a per-class memory queue of DINO feature "views", multi-temperature) plus a small pixel
  loss; muP parameterization; a class-conditional coupling tensor `K_drive[c]`.
- **Results:** competitive FID (CIFAR-10 8.86, ImageNet-64 6.74) at 1.3M–322M params.
- **The deeper bet (unconv.ai):** *dynamical systems as a computing substrate* that maps onto
  analog / physical hardware for ~1000× lower energy than digital accelerators.

**Why it matters to you even though it's not an LLM.** Un-0 computes an answer by letting a
system **settle to a low-energy / coherent state**, and the *degree of settling* is a signal.
sophia-agi's entire job is deciding **when to trust an answer and when to abstain**. The bridge
is exactly this: *the coherence/energy of a settling dynamical system is a natural confidence
and verification signal.* That is not a metaphor — the literature below makes it concrete.

---

## 2. The research line Un-0 sits in (surveyed, real works)

Six themes, most-relevant first. Confidence tags reflect whether the survey directly saw the
identifier.

1. **Oscillators as neurons; synchronization as binding/reasoning.**
   - *Artificial Kuramoto Oscillatory Neurons (AKOrN)* — Miyato, Löwe, Geiger, Welling, **ICLR
     2025 (Oral)**, arXiv 2410.13821 *(ID seen)*. Replaces threshold units with Kuramoto phase
     oscillators; iterating more steps and selecting lowest-energy states raises OOD reasoning
     (Sudoku) accuracy, and **oscillator energy is near-linearly calibrated to correctness**.
   - *KomplexNet* — Muzellec, Alamia, Serre, VanRullen, **TMLR 2025**, arXiv 2502.21077 *(ID
     seen)*. Complex-valued net whose phases synchronize features of the same object; **phase
     coherence tracks OOD robustness**.
2. **Deep Equilibrium / implicit models — "settle to an answer."**
   - *Deep Equilibrium Models (DEQ)* — Bai, Kolter, Koltun, **NeurIPS 2019**, arXiv 1909.01377
     *(ID seen)*. Output = fixed point of a weight-tied map; the **fixed-point residual
     ‖f(z)−z‖ is a stability/consistency score**.
3. **Neural ODEs / continuous-depth — iterative, adaptive-compute reasoning.**
   - *Neural ODEs* — Chen, Rubanova, Bettencourt, Duvenaud, **NeurIPS 2018 (Best Paper)** *(ID
     omitted — not seen)*. Solver tolerance is a per-input "deliberate longer" knob.
   - *Continuous Thought Machines (CTM)* — Darlow et al., **Sakana AI, 2025**, arXiv 2505.05522
     *(ID seen)*. Internal temporal axis; **neuron synchronization used directly as the
     representation**, with variable per-input "ticks."
4. **Energy-Based Transformers — thinking as energy minimization + built-in verification.**
   - *Energy-Based Transformers are Scalable Learners and Thinkers (EBT)* — Gladstone et al.,
     **2025**, arXiv 2507.02092 *(ID seen)*. A Transformer outputs an **energy scalar that
     verifies context–prediction compatibility**; prediction = gradient descent on energy;
     **min-energy selection = Best-of-N self-verification without an external reward model**;
     System-2 gains scale with distribution shift.
5. **Order parameter r as a consensus/confidence readout.**
   - Kuramoto's `r = (1/N)|Σ exp(iθ_j)| ∈ [0,1]` — a single scalar for how much a population
     agrees (established concept). *GASPnet* — Alamia, Muzellec, Serre, VanRullen, **2025**,
     arXiv 2507.16674 *(ID seen)* — "routing by agreement": phase-aligned units reinforce, mis-
     matched ones suppress (an agreement-gated attention analogue).
6. **Oscillatory / neuromorphic hardware — the low-energy substrate.**
   - Oscillator Ising Machines + Equilibrium Propagation (Gower, Nokia Bell Labs / Cambridge,
     **2025**, arXiv 2505.02103 *(ID seen)*); a follow-on OIM/EP work (arXiv 2510.12934, *title
     seen, authors not captured*); CMOS coupled-oscillator Ising machines (COBI, *medium
     confidence — secondary summaries*); and a peer-reviewed review, *Computing with
     oscillators*, **npj Unconventional Computing 2024** (DOI s44335-024-00015-z, *DOI seen*).

**The through-line:** across all six, a dynamical system **relaxes toward a low-energy /
high-coherence state, and how well it relaxes is information** — about correctness, consensus,
and how hard the input is. Un-0 uses that to *generate*; sophia-agi can use it to *verify and
abstain*.

---

## 3. Where sophia-agi is right now (so we don't re-propose)

From reading the 7 new `main` commits directly:

**Now present (offline-tested machinery):** verifier-gated real-time grounding loop with
fail-closed admission (`agent/realtime_grounding.py`); streaming decontamination
(`agent/streaming_decontam.py`); reversible consolidation with a revert ledger
(`agent/realtime_consolidation.py`); a Phase-0 C1 benchmark harness with Wilson intervals and a
control-sanity guard (`agent/realtime_benchmark.py`); a **council of disciplines** with per-seat
fail-closed verify (`agent/council_registry.py`); a **predictive verifiability world-model**
that decides answer-vs-abstain *before* answering (`agent/verifiability_model.py`); a
certificate-carrying **SMT rung** (`agent/smt_verifier.py`); a hash-chained **self-reliability
calibration store** with a `should_answer()` metacognition hook (`agent/calibration_belief_store.py`);
a **T3 calibration-verifier instrument** with AUROC/ECE + pre-registered GO/NO-GO
(`tools/run_calibration_verifier_eval.py`); and the cooperative long-horizon deadline you had
the terminal AI add.

**Still missing (the live-evidence gap, unchanged):** trained discipline adapters (the council
runs on stub seats); any executed GPU/GRPO run (offline machinery only); actual weight updates
from live data (consolidation is dry-run); a live/online data source (fixture backends only); a
real gold-labelled trace corpus and **≥2 independent judge families** above κ≥0.40 (the headline
bench-A verdict sits at κ=0.394, *below* the bar); an exercised SMT "pass" path (z3 not
installed).

**Reading of the gap:** you have an extraordinarily complete *epistemic-measurement* engine. The
five directions below add a *new kind of signal* to it — a convergence/coherence signal that is
complementary to your existing rule-based and verifier-based signals, and that you do **not**
currently compute anywhere.

---

## 4. Five cross-pollination directions

Ordered by payoff-per-cost. Each: the idea, why sophia uniquely enables it, what to reuse, and
the single strongest objection.

### O1 — Order-parameter consensus score for self-consistency (do first; lowest cost, no training)

**Idea.** You already do self-consistency selective prediction (your validated SimpleQA result).
Today that's a vote/agreement count. Replace it with the **Kuramoto order parameter**: embed the
*k* sampled reasoning traces (or the retrieved provenance sources) on a phase circle by semantic
similarity, run a few Kuramoto steps, and read `r ∈ [0,1]` as the confidence score — **abstain
when r is low** (incoherent evidence), answer when `r` is high. This is a *training-free* readout
computable from things you already sample.

**Why sophia enables it.** You already generate the traces and rank the sources; the order
parameter is a drop-in scalar over them, and you already have the calibration/abstention harness
(`calibration.py`, `abstention_scoring.py`, `conformal_gate.py`) to score whether `r` beats your
current agreement signal at matched coverage.

**Reuse.** `agent/calibration.py` (has a `self_consistency` helper), the conformal gate, and the
T3 evaluator to compare `r`-gating vs your current gate on AUROC/ECE.

**Strongest objection.** `r` may be a *fancier* coherence measure that adds nothing over a plain
agreement count — coherent-but-wrong traces (shared bias) will show high `r`. Mitigation: the
honest test is head-to-head against your existing self-consistency gate on the *same*
verifier-checkable set; keep it only if ΔAUROC's CI excludes 0. Cheap to falsify — which is why
it's first.

### O2 — Energy/consistency head as a learned verifier + abstention trigger (flagship)

**Idea.** Follow EBT: train a scalar **energy head over (answer, evidence)** that scores
compatibility, so that **min-energy selection is Best-of-N self-verification without an external
reward model**, and a **high converged energy is a principled abstain signal**. This is the
learned-verifier cousin of your W1 verifier-distilled PRM: W1 distills a *symbolic* step oracle;
O2 learns a *continuous compatibility energy* whose gradient also gives you refinement.

**Why sophia enables it.** You have the verifier stack to *label* energy targets (accepted →
low energy, rejected/abstain → high) and the calibration instruments to prove the energy is
calibrated. AKOrN's empirical near-linear energy↔accuracy calibration is the property you'd be
buying.

**Reuse.** `agent/verified_trace_rlvr.py` and the verifier stack for targets;
`tools/run_calibration_verifier_eval.py` (AUROC/ECE + GO/NO-GO) as the acceptance gate; your
`verifiability_model.py` is the natural place for the head to live (it already predicts
verifier-pass before answering — give it a differentiable energy).

**Strongest objection.** Goodhart: optimizing answers against a learned energy finds
low-energy-but-wrong states, and the energy inherits the coverage gaps of the verifier that
labeled it. Mitigation (same discipline as your PRM work): keep the symbolic verifier as a
periodic ground-truth audit of the energy head, and hold out whole domains to measure whether
the energy generalizes or just memorizes checkable ones.

### O3 — Fixed-point stability as a consistency check (DEQ-style verification)

**Idea.** Frame verification as: *does (claim + evidence) settle to a stable self-consistent
fixed point?* Iterate a weight-tied "re-express the claim from its evidence" map; a **large
fixed-point residual or non-convergence flags an unstable, unverifiable answer** → quarantine or
abstain. This is a convergence-based sibling to your SMT rung: SMT certifies a *decidable* band
exactly; O3 gives a *soft* stability score on the large undecidable remainder.

**Why sophia enables it.** Your real-time grounding loop already re-verifies beliefs
(`mark_stale`) and your consolidation is reversible — a stability iteration slots in as an
admission-time check before ingestion.

**Reuse.** `agent/realtime_grounding.py` (`ingest_one` admission pipeline) as the host for a
stability gate; the reversibility ledger to undo an ingestion whose fixed point later drifts.

**Strongest objection.** Defining the "re-express" map so its fixed point *means* factual
consistency (not just paraphrase stability) is the whole difficulty; a trivial map converges for
everything. Mitigation: make the map route through the *evidence set* (claim must be
reconstructible from sources), so non-convergence corresponds to evidence that doesn't actually
support the claim; validate against your labeled C1 fact pack.

### O4 — Convergence-driven adaptive test-time compute ("think longer" gated by coherence)

**Idea.** Use the *convergence behavior itself* to allocate deliberation: monitor an internal
iteration (self-consistency `r` stabilizing, energy descent flattening, or trace agreement
converging) and **spend more samples/steps on inputs that haven't converged, stop early on those
that have** — and use the convergence gap as a confidence proxy. EBT shows the benefit of
"thinking" scales with distribution shift, so this naturally spends compute where it helps.

**Why sophia enables it.** You already have the long-horizon loop with a cooperative deadline
and a step-level decision log — the exact place to insert a "converged? stop : sample more"
policy, turning your fixed-k self-consistency into an adaptive-k one.

**Reuse.** `agent/long_horizon.py` (the deadline/step loop just added) + the self-consistency
sampler; log the per-query step count as a new calibration feature.

**Strongest objection.** "Benefit-from-thinking" and convergence can be gamed or flat — some hard
queries never converge and you burn budget for nothing. Mitigation: cap adaptive-k with your
existing deadline, and treat *persistent non-convergence* as an abstain signal rather than a
reason to keep spending.

### O5 — Energy-based epistemics on a low-energy oscillatory substrate (longer-horizon, speculative)

**Idea.** The unconv.ai thesis, made relevant to you: an energy-minimization verifier/sampler
(O2) maps onto **oscillator Ising machines / CMOS coupled-oscillator hardware**, whose *native
dynamics perform the optimization* — so always-on verification and uncertainty sampling become
affordable at inference-time energy budgets. This is the "epistemics layer as cheap physical
computation" bet.

**Why sophia enables it.** If O2 produces a genuine energy-based verifier, its *inference* is
exactly the class of energy-minimization/sampling that OIM+Equilibrium-Propagation hardware runs
natively — a concrete bridge from your software epistemics to the 1000×-energy thesis.

**Reuse.** Nothing in-repo yet; this is a research-direction flag, not a near-term build. It
earns its place only *after* O2 shows a calibrated energy verifier exists in software.

**Strongest objection.** Large gap between "coupled-oscillator Ising machines solve MNIST-scale
energy problems" and "run an LLM-scale semantic verifier" — the hardware today handles small
combinatorial energies, not high-dimensional language compatibility. Honest status: a horizon
bet contingent on O2, not a 2026 deliverable.

---

## 5. Priority and honest framing

| # | Direction | Payoff | Cost | Training? | Kills-it-fastest test |
|---|---|---|---|---|---|
| O1 | Order-parameter consensus score | Better abstention gate, free | Low | No | Beat your self-consistency gate on ΔAUROC (CI excludes 0) |
| O2 | Energy head as learned verifier | Modality-general self-verification | Med (GPU) | Yes | Calibrated energy (AUROC/ECE) + held-out-domain generalization |
| O3 | Fixed-point stability check | Soft consistency on undecidable claims | Med | Maybe | Non-convergence ⇔ unsupported claim on the C1 pack |
| O4 | Convergence-adaptive compute | Spend deliberation where it helps | Low–Med | No | Adaptive-k beats fixed-k at matched cost |
| O5 | Oscillatory-hardware epistemics | ~1000× energy (horizon) | High | — | Contingent on O2; hardware scale gap |

**Recommended sequence.** **O1 first** — it's training-free, computable from what you already
sample, and cheaply falsifiable against your own validated self-consistency result. **O2 as the
flagship** — the learned energy verifier is the highest-upside and slots into
`verifiability_model.py`, but wait until a GPU backend is wired (same blocker as your GRPO/
consolidation lanes). O3 and O4 are inexpensive follow-ons that reuse the grounding and
long-horizon loops. O5 is a flag on the horizon, honest about the hardware gap.

**The one-line reason this is worth doing.** sophia-agi today measures trust with *rules and
verifiers*. Un-0's world contributes a *different, complementary signal you don't currently
compute anywhere*: how coherently an answer settles. O1/O2 add that signal to your abstention
gate; the rest are the ways it could deepen. None replaces your existing epistemics — they
extend it along the one axis (convergence/coherence) your current stack is silent on.

---

## Provenance & confidence

- Un-0 mechanism: read directly from `unconv-ai/Un-0` source (`README.md`, `un0/model.py`,
  `un0/losses.py`, `un0/eval.py`).
- sophia-agi state: read directly from the 7 new `origin/main` commits' added files.
- Literature: a web-search survey; titles/venues as surfaced, arXiv IDs only where explicitly
  seen (and still "confirm before citing"). AKOrN, EBT, DEQ, Neural ODEs, CTM, GASPnet, and the
  OIM/EP hardware line are the load-bearing references and were cross-checked for name+venue.
- These are directions with named failure modes, not results. Nothing here has been run.