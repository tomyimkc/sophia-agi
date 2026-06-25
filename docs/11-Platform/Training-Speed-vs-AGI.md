# Training speed vs AGI — the honest strategy note

> Synthesis of three research threads (capability ceilings, verifier scaling, generation-bound
> wall-clock). BRUTALLY HONEST. The repo's defensible value is a **fail-closed provenance layer**,
> not AGI. No fabricated numbers below — every quantitative claim is either labelled as a rule of
> thumb or left to the eval that measures it.

---

## 1. Top line: training SPEED is not on the AGI critical path

Making the LoRA/RFT loop *faster* buys you more iterations of the same loop. It does **not** raise
the capability ceiling. Three load-bearing facts:

- **Imitation copies style, not capability.** Fine-tuning a small model on a stronger model's
  outputs makes it *sound* like the teacher while closing little of the real capability gap
  (Gudibande et al., "The False Promise of Imitating Proprietary LLMs", arXiv:2305.15717). Our SFT
  / council-distillation pipelines inherit this limit — they buy style and format, not new reasoning.
- **The ceiling is set at pretraining.** Capability per parameter is bounded by compute-optimal
  pretraining (Hoffmann et al., "Chinchilla", arXiv:2203.15556). A 3B base has a 3B base's ceiling;
  no amount of post-training throughput moves it.
- **The real levers are elsewhere:**
  - **Test-time compute** — spending more inference search/sampling on hard problems often beats a
    bigger model (Snell et al., arXiv:2408.03314). Implemented here: `tools/test_time_search.py`.
  - **RL in real environments** with a *real* reward — but with the caveat that RL may mostly
    *sharpen* abilities already in the base rather than add new ones ("Does RL Really Incentivize
    Reasoning Capacity Beyond the Base Model?", arXiv:2504.13837 — 2025, pre-cutoff; confirm figures).
  - **Data** quality/coverage, and ultimately **a bigger base model.**

**Implication:** speed work is a *cost* optimization, not a *capability* one. Prioritize it
accordingly (see §5).

---

## 2. The verifier-scaling crack

Our gate is a **fixed, deterministic** provenance checker. That is a feature (reproducible,
fail-closed) and a liability: a static target gets **more hackable and more ceiling-limiting as the
policy strengthens**. The policy learns the gate's exact contour and Goodharts it — measured
pass-rate climbs while real epistemic quality stalls.

- Reward hacking is the expected failure mode (Skalse et al., "Defining and Characterizing Reward
  Hacking", arXiv:2209.13085).
- Optimizing against an imperfect checker overshoots and then degrades true quality (Gao &
  Schulman, "Scaling Laws for Reward Model Overoptimization", arXiv:2210.10760).
- Rule-based verifiers have characteristic, exploitable holes ("Pitfalls of Rule-based Verifiers",
  arXiv:2505.22203 — 2025, pre-cutoff; confirm figures).

**Prescriptions (implemented / planned):**

1. **A reference verifier the policy is NEVER trained against.** Held-out adversarial traps that
   only score, never teach: `data/reference_holdout_traps.json` (flagged `heldout: true`,
   "NEVER use for training/RL").
2. **A Prover-Verifier loop** to keep the verifier ahead of capability — pit a *helpful* and a
   *sneaky* prover at the same gate and track the **sneaky evasion rate** against the held-out
   reference checker (Kirchner et al., "Prover-Verifier Games", arXiv:2407.13692). Implemented:
   `tools/prover_verifier.py`.
3. **Retrieval-grounded, claim-decomposed SUPPORT checking** — decompose a response into atomic
   claims and check each is *supported* by retrieved evidence, rather than matching a brittle rule.
   Implemented: `tools/support_check.py`.

Related: weak supervisors eliciting strong models (Burns et al., "Weak-to-Strong Generalization",
arXiv:2312.09390) and debate as a scalable-oversight primitive (Irving et al., "AI safety via
debate", arXiv:1805.00899) are the conceptual backbone for "verifier ahead of capability".

---

## 3. Generation-bound fact

In an RL/RFT step the wall-clock is dominated by **sampling candidates**, not the optimizer step.
As a rule of thumb, **~70–80% of RL/RFT wall-clock is generation** (this is a planning heuristic,
not a measured number from this repo — measure it before quoting it). Consequences at our scale:

- **FSDP / FP8 / MoE are irrelevant at 3B + LoRA.** Those techniques pay off when the *training*
  step or model size is the bottleneck. Here the trainable footprint is tiny and the optimizer step
  is cheap; speeding it up speeds up the wrong 20%.
- **The lever is the generation backend.** Batch generation via vLLM attacks the dominant cost:
  `tools/run_rft.py --gen-backend vllm` (lazily loads vLLM once and batch-generates all N
  candidates per prompt; `--model mock` falls back to native deterministic generation offline).

---

## 4. Measurement gap: gate != capability

The deterministic gate measures **process integrity** (no fabricated citation, no false arithmetic,
no forbidden-lineage merge). It says **nothing** about whether the model can *reason*. Optimizing
gate pass-rate + wall-clock alone is Goodhart on a non-capability metric — a model that abstains on
everything trivially "passes" while being useless. (Abstention is a *correct* fail-closed output;
that is exactly why gate pass-rate cannot stand in for capability.)

The fix is a **held-out generality eval** with no provenance/attribution content: ARC-AGI-style
abstraction/pattern puzzles, multi-step arithmetic, logic, analogy, out-of-domain reasoning, scored
**deterministically against gold** (no LLM judge). Implemented: `tools/eval_generality.py` over
`data/generality_tasks.json`. This makes "are we getting more capable?" falsifiable.

---

## 5. The reprioritized ROI roadmap

Ordered so each step de-risks the next. Speed is deliberately *late* — you do not want to iterate a
loop faster until you can measure whether the loop is helping.

| # | Priority | Why it comes first | Tool |
|---|----------|--------------------|------|
| 1 | **Measurement** | Without a capability metric, every other optimization is Goodhart. | `tools/eval_generality.py` |
| 2 | **Test-time compute** | Highest capability-per-effort lever; no retrain needed. | `tools/test_time_search.py` |
| 3 | **Verifier hardening** | Keep the gate ahead of the policy before pushing the policy harder. | `tools/prover_verifier.py`, `data/reference_holdout_traps.json`, `tools/support_check.py` |
| 4 | **vLLM generation** | Now make the (verified, measured) loop cheaper where the cost actually is. | `tools/run_rft.py --gen-backend vllm` |
| 5 | **Confront the base ceiling** | When 1–4 plateau, the honest answer is a bigger/better base — not more speed. | (procurement / base-model decision, not a tool) |

---

## 6. Honest caveats

- **Citation hygiene (corrected).** arXiv ids encode year+month as `YYMM`: `25xx` = 2025,
  `26xx` = 2026. This repo's assistant knowledge cutoff is **January 2026**, so `25xx` papers
  are **pre-cutoff** (within the knowledge window) and `26xx` papers are **post-cutoff** (must be
  independently verified before quoting). Every citation in *this* doc is `23xx`–`25xx`, i.e.
  **pre-cutoff** — including `arXiv:2504.13837` (RL-ceiling, Apr 2025) and `arXiv:2505.22203`
  (Pitfalls of Rule-based Verifiers, May 2025), which are 2025, not post-cutoff. Still confirm the
  exact figures before they enter the proof package; recency ≠ post-cutoff.
- **Where genuine post-cutoff (`26xx`) ids DO appear** — in this session's chat research summaries
  and parts of `Training-Efficiency-Feasibility.md` — those (e.g. `2601.*`, `2604.*`, `2605.*`)
  are the ones that must be verified before any AGI proof-package use.
- **The load-bearing classics are real and pre-cutoff:** Gudibande (2305.15717), Chinchilla
  (2203.15556), Snell (2408.03314), Skalse (2209.13085), Gao & Schulman (2210.10760), Burns
  weak-to-strong (2312.09390), Irving debate (1805.00899), Kirchner Prover-Verifier (2407.13692).
- **No fabricated numbers.** The only quantity above is the ~70–80% generation-share *rule of
  thumb*, explicitly labelled as a planning heuristic to be measured — not a repo result.
- **No AGI claim.** Nothing here moves a 3B base toward AGI. The repo's defensible, honest value is
  a **fail-closed provenance layer**: it abstains when it cannot machine-verify, and abstention is a
  correct output. That discipline is the asset — training speed is not.
