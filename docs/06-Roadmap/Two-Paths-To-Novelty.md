# Two Paths to Novelty — research plan (world model + proof search)

**Status:** research plan, not a claim. `candidateOnly`, `level3Evidence: false` until run.
**Author:** GLM-5.2 advisory, 2026-06-26. Grounded in the repo's `deliberation_roofline`
result and the public methodology literature.

---

## Why these two, and not other things

The roofline result (`reasoning/deliberation_roofline.py`) is an **architectural
ceiling**: every Sophia output is bounded by (training data ∪ retrieved sources),
filtered by a verifier. That ceiling is what makes the system *safe* (cannot
hallucinate past the verifier) and is also what prevents *novelty*. Breaking it
legitimately — producing knowledge outside (train ∪ retrieved) — requires one of:

1. **A learned dynamics model that generalizes** to situations with no proximal
   training trace (soft novelty via interpolation-into-extrapolation). **Path A.**
2. **Proof search in a formal system**, where the proof *is* its own verification —
   the one domain where novelty is reachable *under* a ceiling. **Path B.**

Everything else (more retrieval, bigger LLM, better gate) raises how much of the
verifier's headroom you capture but cannot cross the ceiling. These two are the
only fundamental paths; the rest is support.

---

# Path A — A real learned world model (the AlphaGo-move experiment)

## The falsifiable question
**Can a neural outcome predictor, trained on Sophia's own harness traces,
generalize to held-out AND distribution-shifted (state, action) pairs — or does it
collapse to memorization?** This is the experiment that tells you whether the
AlphaGo recipe (learned model + search) is viable for your substrate at all.

The repo already has the scaffold (`agent/verified_world_model.py`) with the
canary discipline (promote only on held-out gain + bounded shift-degradation),
but the predictor is a toy feature-logistic. Path A fires it with a real model.

## Methodology (grounded in the literature)
- **DreamerV3** ([Hafner et al. 2023](https://huggingface.co/papers/2301.04104);
  [official repo](https://github.com/danijar/dreamerv3); [Nature 2025 control
  paper](https://www.nature.com/articles/s41586-025-08744-2)) — the canonical
  learned-world-model approach: encode state → discrete latent → predict next
  latent + reward → actor-critic *inside the imagined model*. Key property for us:
  it supports **discrete actions** and learns from a **replay buffer of traces**
  (offline-compatible).
- **RLVR-World** ([NeurIPS 2025](https://ise.thss.tsinghua.edu.cn/~mlong/doc/RLVR-World-NeurIPS25.pdf))
  — directly trains world models with **verifier rewards**, the exact alignment
  with Sophia's verifier-as-reward thesis.
- **Distribution-shift generalization** ([OOD survey 2025](https://arxiv.org/html/2507.21160v1);
  [offline-RL OOD](https://link.springer.com/article/10.1007/s00521-026-11966-8))
  — the failure mode to measure: a model that aces held-out but collapses under
  shift has *memorized*, not generalized. Sophia's `verified_world_model.py`
  shift-split + `shiftDegenerate` verdict already encodes this test.

## Implementation (extends existing seams; does NOT rewrite)
1. **New predictor** `agent/world_model_neural.py` implementing the existing
   `OutcomePredictor` protocol (`predict(state, action) -> p`). Architecture: a
   small MLP over a tokenizer embedding of the `(state, action)` strings (NOT a
   full DreamerV3 — that needs an environment Sophia doesn't have). Trained on
   `OutcomePair` traces via the injected `predictor_factory`. ~150 LOC + torch
   optional behind the same CUDA gate as RLVR.
2. **Trace corpus**: mine `agent/memory/agent_runs/*.jsonl` (the harness decision
   logs) into `(state-bucket, action, outcome)` triples. The state-bucketing
   adapter (`state_key` in `planner_learned_sim.py`) already exists.
3. **Fire the existing scaffold**: `train_verified_world_model(traces,
   predictor_factory=NeuralPredictor)`. Read the verdict:
   - `promote` → the model generalized; wire it into `planner_learned_sim.py`.
   - `hold-shift-degenerate` → it memorized; the substrate is bounded to
     lookup-table world models. **Either result is a real finding.**
4. **Test bench**: `tests/test_world_model_neural.py` — construct a synthetic
   trace corpus with a *learnable* signal + a *shifted* test split that reverses
   the signal; assert promote-on-learnable AND hold-on-shifted-reversed (the
   canary must fire on the degenerate case).

## Honest scope
This is NOT a DreamerV3 port. Sophia has no environment to dream in. This tests
the *narrow* question: does a neural predictor beat the lookup table at
generalizing over the harness's own action-outcome traces. If yes, the planner
gets a real model; if no, you've learned the substrate's ceiling is real.

---

# Path B — Wired proof search (the legitimate novelty pathway)

## The falsifiable question
**Can Sophia, in the math domain, produce a proof (or proof step) that is verified
correct by Lean AND was not trivially retrievable from its training data?** This is
the one place the verifier-ceiling permits novelty: a formal proof is
self-verifying, so "novel + verified" is achievable without breaking the fail-closed
discipline. The repo has `math_verifier.py` (sympy) with Lean *reserved-but-not-wired*
(returns `abstain`/`lean_unavailable`).

## Methodology (grounded in the literature)
- **LeanDojo + ReProver** ([NeurIPS 2023](https://neurips.cc/virtual/2023/poster/73510);
  [project](https://leandojo.org/); [repo](https://github.com/lean-dojo/reprover);
  [docs](https://leandojo.readthedocs.io/)) — the open, reproducible
  AlphaProof-style stack. LeanDojo provides: programmatic Lean 4 interaction
  (tactic application, premise extraction, proof-state trees). ReProver adds
  **retrieval-augmented premise selection** — the LLM proposes tactics, retrieval
  surfaces relevant library lemmas, Lean verifies. This maps 1:1 onto Sophia's
  generate→retrieve→verify idiom.
- **AlphaProof** ([Nature 2025](https://www.nature.com/articles/s41586-025-09833-y);
  [Schrittwieser walkthrough](https://www.julian.ac/blog/2025/11/13/alphaproof-paper/))
  — the closed-source reference: RL over Lean proof search, tactic generation,
  partial-proof "Lean state" representation. We use the *open* LeanDojo analogue.
- **LeanProgress** ([ICLR 2025](https://arxiv.org/html/2502.17925v2)) — proof-progress
  prediction to guide search and catch LLM-proof hallucination via formal check.
- **DL4TP** ([curated bibliography](https://github.com/zhaoyu-li/DL4TP)) — survey entry point.

## Implementation (extends existing seams; does NOT rewrite)
1. **Wire the Lean backend** in `agent/math_verifier.py` — replace the `abstain`/
   `lean_unavailable` stub with a real LeanDojo call. Lean 4 + elan + lean-dojo
   pip are an opt-in extra (`requirements-theorem.txt`); CI stays sympy-only,
   fail-closed abstention when Lean is absent (preserve the current default).
2. **New module** `agent/proof_search.py` — best-first search over Lean tactic
   steps: state = Lean proof state, actions = LLM-proposed tactics (via the
   existing model adapter) + ReProver premise retrieval, verifier = Lean itself.
   Bounded depth + node budget; fail-closed (no proof → abstain, never assert).
3. **Test bench**: `tests/test_proof_search.py` against a tiny bundled Lean
   library (e.g. the `Marlowe`/`Nat.add_comm` style examples from LeanDojo's
   tutorial). Assert: (a) a trivially-provable theorem is proved and Lean-verifies
   it; (b) an unprovable goal abstains (fail-closed); (c) CI mode (no Lean) abstains
   with `lean_unavailable` and never crashes.
4. **The novelty probe**: on a small held-out theorem set, log whether the produced
   proof appears verbatim in the training corpus / library (a near-duplicate check).
   A proof that is Lean-valid AND not a near-duplicate retrieval is the novelty
   signal — recorded honestly, candidate-only.

## Honest scope
This is NOT a bid to beat AlphaProof. It wires the **open, reproducible**
AlphaProof-style path (LeanDojo) into Sophia's existing math-verification seam,
gated behind an opt-in extra so the fail-closed default is untouched. The novelty
probe is a *measurement* (does the system ever produce non-retrieved verified
math?), not a claim of creative superintelligence.

---

## Cross-cutting discipline (both paths)
- **Opt-in extras**, never default-on. Core stays stdlib + numpy; CI unchanged.
- **Fail-closed** everywhere: no model/Lean → abstain, never fabricate.
- **`candidateOnly`/`level3Evidence: false`** on every artifact until a gated run.
- **The roofline is respected**: Path A raises the verifier-achievable headroom;
  Path B is the one domain where the ceiling permits novelty. Neither claims AGI.

## Sequencing
**Path A first** — it's the experiment that tells you whether the planner/world-model
substrate is viable at all (and it runs on the Spark, which the user is deploying).
**Path B second** — it's the deeper novelty pathway but heavier infra (Lean toolchain).
Both are independently shippable; they don't depend on each other.

## Open questions for the human (decide before implementing)
1. **Path A predictor scope**: small MLP over string embeddings (lightweight,
   runs on CPU, ~1 day) vs. a real DreamerV3-style discrete-latent model (heavy,
   needs GPU, ~1 week, may be overkill without an environment). **Recommend: MLP first.**
2. **Path B Lean version**: Lean 4 (LeanDojo-v2 target, modern) vs Lean 3 (legacy,
   more ReProver tutorial coverage). **Recommend: Lean 4.**
3. **GPU**: Path A's MLP is CPU-fine; Path B's premise-selection training wants a
   GPU (the Spark or RunPod). Inference (tactic proposal) is light. OK to start
   CPU-only and add GPU for the training phase only?
4. **Novelty-probe strictness** (Path B): "not verbatim in corpus" (loose) vs
   "not a near-duplicate by embedding similarity" (strict). **Recommend: strict.**

## DECISIONS (human, 2026-06-26)
1. **Path A: Full DreamerV3-style** discrete-latent world model (the real
   generalization question, not the lightweight proxy).
2. **Path B: Lean 4** via LeanDojo-v2.
3. **GPU: CPU-first, GPU later.** Dev/test runs on CPU (torch CPU backend); the
   full-scale real-trace training phases are the "GPU later" steps (Spark/RunPod).
4. **Path B novelty probe: strict** (embedding near-duplicate).

**Resolution of the DreamerV3-style + CPU-first tension:** build the real
DreamerV3-style **discrete-latent RSSM** architecture (the load-bearing component
for the generalization question + the planner), with **torch imported lazily behind
the CUDA gate** (same pattern as `run_rlvr.py` / `requirements-rl.txt`). The
discrete-latent mechanics live in `agent/world_model_dreamer.py`; torch is an
optional import that **abstains fail-closed** when absent (like `math_verifier.py`'s
Lean stub). Dev/test runs on torch-CPU; full real-trace training is the GPU phase.
This respects both choices: DreamerV3-style architecture as chosen; CPU-first
dev/test as chosen.
