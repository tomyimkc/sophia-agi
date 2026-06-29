# DeepSeek-Reasonix Integration + Physics/Math/Code Capability Roadmap

**Status:** planning doc (no capability claims). Analysis of
[esengine/DeepSeek-Reasonix](https://github.com/esengine/DeepSeek-Reasonix) and a
sequenced plan for folding its *useful* concepts into Sophia, in service of one
goal: train the physics, mathematics, and coding ability of a Sophia model to a
benchmarkable, top-tier level. Grounded in a repo audit of `provenance_bench/`,
`agent/`, `reasoning/`, `selfextend/`, `training/`, `eval/`, `benchmark/`, and
`agi-proof/`.

> **Scope discipline.** Per [VISION.md](../../VISION.md), the benchmark is
> trustworthiness and verifiability, not breadth. Nothing here is a capability
> claim until it passes the no-overclaim gate (≥2 independent judge families,
> κ reported, ≥3 seeds, 95% CI excluding zero). Every number below is a *target*,
> not a result, until it lands in `agi-proof/benchmark-results/` with a sealed
> holdout.

---

## 0. What DeepSeek-Reasonix actually is (myth vs. reality)

| Assumption | Reality |
|---|---|
| A reasoning-model *trainer* with R1-style RL advantages | **No.** It is a terminal coding **agent harness** (Go, single binary). |
| Has distillation / CoT-training / RLVR | **None.** It is purely **inference-time**. |
| "Reasonix" implies a reasoning model | Marketing. It drives DeepSeek (flash/pro) + MiMo via OpenAI-compatible endpoints. |

So there is **no training IP to import.** But four of its harness ideas are real
engineering wins, and — critically — a cheap, stable agent harness is *exactly the
engine you need to manufacture verifiable reasoning training data at scale*. That
is the bridge between Reasonix and your actual goal.

### The four reusable concepts

1. **Prefix-cache stability.** Conversations grow *prepend-only* so the KV-cache
   prefix is never invalidated → cheap long sessions. DeepSeek's prefix cache is
   billed at a fraction of fresh tokens; keeping the prefix stable is the whole game.
2. **Planner/Executor split in separate cache-stable sessions.** A planner model
   emits a structured plan as text; an executor (full tool-using agent) carries it
   out in its *own* session. The sessions never mix, so neither prefix is disturbed.
3. **Low-frequency compaction.** Compact once when `prompt_tokens` reach
   `compactRatio` (default 0.8) of the window — not every turn — to respect the cache.
4. **Config-over-code + MCP subprocess tools.** Providers, tools, plugins declared
   in TOML; external tools run as stdio JSON-RPC subprocesses (MCP-compatible).

### Why this matters for *training*, not just serving

RLVR/GRPO and your `selfextend/loop.py` are **data-hungry**: they need millions of
rollouts, each with a verifiable reward. The dominant cost is inference. Reasonix's
prefix-cache discipline + planner/executor split is a blueprint for a **rollout
generator that is ~3–10× cheaper per trace** when generating from a hosted DeepSeek
(or your own vLLM) endpoint. Cheaper rollouts → more RL steps per dollar → the
live GPU run in your failure-ledger becomes affordable. **That is the integration.**

---

## 1. The honest state of Sophia for this goal

What already exists and is strong (this is a *lot* of the way there):

- **Verifiable rewards, judge-free.** `provenance_bench/math_reward.py` (sympy),
  `provenance_bench/code_reward.py` (sandboxed exec), `agent/math_verifier.py`,
  `agent/code_verifier.py`. This is the moat — machine-checkable ground truth.
- **RLVR plumbing prepared.** `tools/run_rlvr.py --task {math,code}`, TRL + vLLM +
  QLoRA in `requirements-rl.txt`. Gated on CUDA (Open in `failure-ledger.md`).
- **Contamination safety.** `provenance_bench/rl_dataset.py` (entity-disjoint
  splits), `holdout_seal.py` (crypto sealing), `eval/contamination.py`.
- **Self-extension loop.** `selfextend/loop.py`: abstain → synthesize verifier →
  validate on held-out → promote. Currently *selection*, not parameter update.
- **Measurement rigor.** `provenance_bench/runner.py` + `score.py` (multi-judge,
  κ), no-overclaim gate, sealed reviewer packs.
- **Serving/infra.** MoE router + quant (`moe/`), FlashAttention + NVFP4 kernels,
  KV-cache/quant serving, MLA documented.

The three honest gaps for *this* goal:

| Gap | Evidence | Closes via |
|---|---|---|
| **No live GPU RL run** has happened | failure-ledger OPEN item; `run_rlvr.py` "prepared" | Phase 3 |
| **Physics is barely covered** | only `data/science.json` (17 records); no physics verifier, no Lean physics | Phase 2 + 4 |
| **No external SOTA benchmark numbers** | `agi-proof/external-benchmarks/` are *plans* (MATH, GSM8K, SWE-bench, etc.) | Phase 5 |

---

## 2. The connective thesis

**Verifiers are both the moat and the fuel.** Math and code give machine-checkable
ground truth with *no LLM judge* — exactly what RLVR and self-extension need.
Physics is the natural, under-explored extension (recent work: Lean4Physics /
PhysLib / LeanPhysBench, PhysProver). The plan:

```
  cheap rollouts (Reasonix harness ideas)
        │
        ▼
  verifiable reward (math sympy / code exec / physics dimensional+numeric+Lean)
        │
        ▼
  GRPO/RLVR post-training (TRL+vLLM+QLoRA)   ←─ the live GPU run
        │
        ▼
  selfextend promotes new verifiers on sealed holdout
        │
        ▼
  no-overclaim benchmark gate → numbers you can publish
```

Reasonix is the on-ramp's *fuel pump*; the verifiers are the *fuel*; GRPO is the
engine; the no-overclaim gate is the dynamometer.

---

## 3. Crazy-idea brainstorm (ranked by leverage × feasibility)

Ordered so the high-leverage, buildable ones come first. ★ = recommended for the
roadmap; ☆ = research bet.

**★ A. Cache-stable rollout factory.** Port Reasonix's prepend-only session +
0.8 compaction into a `pipeline/rollout/` generator that produces RLVR traces from
a DeepSeek/vLLM endpoint. Target: 3–10× cheaper traces. *This unlocks everything
else.*

**★ B. Planner/Executor as a reward-shaping signal.** Run Sophia's existing
`council_deliberate.py` in the Reasonix two-session pattern: a *planner* seat emits
a structured plan, an *executor* seat (tool-using) executes. Reward the executor on
verifier pass; reward the planner on *whether its plan led to a passing execution*
(credit assignment to the plan). This is a cheap proxy for a process reward model
without training a PRM.

**★ C. Verifiable Physics gate.** Build `agent/physics_verifier.py` with three
tiers: (1) **dimensional analysis** (units must balance — pure-Python, deterministic),
(2) **numeric check** against a reference solver / known constants, (3) **symbolic**
(sympy) for closed-form. Wire as `physics_sound` in `agent/verifiers.py` +
`provenance_bench/physics_reward.py`. This is the missing third RLVR domain.

**★ D. Lean physics path.** Stand up `formal_proofs/eval/` on PhysLib/Lean4Physics
so a subset of physics problems get *formally* verified (theorem-proving reward),
mirroring your existing Lean math hooks (`agent/lean_verifier.py`). Hardest reward,
highest trust.

**★ E. GRPO with a layered reward.** Per the SOTA (DeepSeek-R1, RLVR): binary
verifier reward + small shaping terms (format, brevity, calibration-from
`agent/uncertainty_scoring.py`). Use `provenance_bench/improvement.py` to measure
gate-on vs gate-off delta on a sealed split.

**☆ F. Generative Process Reward Model (GenPRM-style).** Train a PRM that scores
*each reasoning step* by re-deriving it, using your verifiers to auto-label step
correctness (formal-verification-synthesized PRM data is a 2025 technique). Turns
your binary outcome rewards into dense rewards → sample-efficient RL.

**☆ G. Self-extension → parameter update (close the loop).** Today `selfextend`
selects verifiers; wire its promoted verifiers as *new RLVR reward functions* so a
promotion triggers a fine-tune. This is the failure-ledger's "highest leverage"
item — turning a toy pillar real.

**☆ H. Belief-graph-guided MCTS rollouts.** Use `agent/verification_mcts.py` +
`reasoning/reasoning_compiler.py` to search reasoning trees at *generation* time,
keeping only verifier-passing branches as SFT data (expert iteration / STaR with a
real checker). Cross-pollinates your reasoning IP into data generation.

**☆ I. Curriculum from the data passport.** Your `pretraining/data_passport/`
already flags the math/code curriculum as ~60% near-duplicate + unlicensed. Use it
to build a *deduped, difficulty-sorted* curriculum (easy→hard by verifier
pass-rate of the base model) — curriculum RL is a known stabilizer.

**☆ J. Distill DeepSeek-R1 traces under your gate.** Generate long-CoT traces from
a strong open reasoner, *filter through your verifiers* (keep only correct,
provenance-clean traces), SFT a small Sophia model on the survivors. Cold-start
before RL, exactly the R1 recipe — but every trace is gate-checked, so no
contamination/hallucination leaks in.

**☆ K. Physics world-model micro-sim reward.** For mechanics/E&M problems, run a
tiny deterministic simulator (pure Python) and reward answers that match the
simulated trajectory. Verifiable reward for problems with no closed form.

### Addendum — mined from latest Reasonix content (v1.13, GUIDE/SPEC)

**★ L. Cache-shared best-of-N via branching** *(built — `Session.branch` +
`best_of_n`).* Reasonix `/branch <turn>`: fork from the plan checkpoint so N sampled
solutions reuse the cached prefix. Keep verifier-passing branches = expert-iteration
SFT data at a fraction of naive best-of-N cost (measured 4.5×).

**★ M. Stale-count repair loop** *(built — `generate_until`).* Reasonix AutoResearch
tracks `stale_count` and *pivots* instead of grinding. Stops re-rolling a problem the
model can't currently solve — saves rollout budget, surfaces the hard tail.

**★ N. User-preserving compaction + archive** *(built — `Session.compact`).* Keep user
turns verbatim, fold only assistant/tool work, archive dropped messages. Honest
context management that never loses the task intent or the trace.

**☆ O. Guardian subagent.** Reasonix v1.12 added a safety-oversight subagent. Wire
Sophia's `conscience.py` / `constitutional_gate.py` as a guardian pass over each
rollout's tool calls — reject reward-hacking or unsafe actions before they score.

**☆ P. Permission-gated tool execution.** Reasonix's `deny > ask > allow` policy with
arg-specifier matching (`Bash(npm run test:*)`). For the rollout factory's tool loop,
classify tools read-only vs. mutating (Sophia already has `dataflow/firewall.py`,
`security/labels.py`) so generated trajectories teach *safe* tool autonomy.

**☆ Q. Memory-v5 execution contract.** Reasonix compiles prior-turn outcomes into an
"execution contract" injected into the next turn *only when outcomes create actionable
constraints*, never mutating the cache-stable prefix. Maps onto Sophia's
`sophia_contract/` + `CONTRACT.md` — a cache-safe way to carry verified constraints
across rollout turns.

**☆ R. Requirement-by-requirement evidence audit.** Reasonix's multi-phase goal
evaluation checks each requirement against evidence. A natural rubric for the
`provenance_bench` runner on multi-step problems (partial credit per sub-goal), and a
dense-reward source for ☆F.

---

## 4. Feasible roadmap (phased, gated, falsifiable)

Each phase has an **entry/exit gate** and lands artifacts where the no-overclaim
CI can see them. CPU/offline phases first; the single expensive GPU phase is
isolated so it can be done in one rented-GPU burst (RunPod MCP is wired).

### Phase 0 — Baseline & scaffolding (CPU) — ✅ **landed**
- ✅ `tools/run_baseline.py`: pass@1 on the FAMILY-disjoint **sealed** eval split
  for `math` / `physics` / `code`, scored by the same verifiable reward the RLVR
  run optimizes (no LLM judge), with a Wilson 95% interval.
- ✅ Pre-registration reports sealed into `agi-proof/benchmark-results/baseline-*.json`
  (eval-split content hash recorded, so the Phase 3 "after" is provably on the
  identical unpeeked set). 4 tests in `tests/test_run_baseline.py`.
- ✅ **Exit gate met:** committed baseline table with sealed hashes + CIs. With the
  offline `mock` model these are the **harness floor (~0)**, explicitly labelled
  "NOT a capability claim" — a real subject model fills in the comparable number.

### Phase 1 — Cache-stable rollout factory (CPU + hosted API) ★A,★B — ✅ **scaffolded**
- ✅ `pipeline/rollout/`: `session.py` (append-only prefix-stable `Session` +
  0.8 compaction), `cost.py` (prefix-cache cost model), `factory.py`
  (`RolloutFactory` with the planner/executor SEPARATE-session split, verifiable
  reward, `AgentTrajectory` emission). Provider via existing `agent/model.py`
  (mock/DeepSeek/OpenRouter/vLLM); offline-deterministic via `ScriptedClient`.
- ✅ Traces emitted in `pretraining/vertical_data/schemas.py` format and validated.
- ✅ **Exit-gate target met (offline):** the deep-session invariant shows **3.3×
  whole-rollout / 3.9× input-only** savings vs. a no-cache baseline (≥3× target),
  proven in `offline_invariants()`; 12 tests in `tests/test_rollout_factory.py`.
  *Caveat (honest):* the win scales with **session depth** — a 2-turn planner/
  executor QA shows ~1.0× because there is no deep prefix yet. The savings are real
  in long "leave-it-running" sessions, which is the Reasonix regime.
- ✅ **Mined from latest Reasonix content (v1.13):** `Session.branch()` → cache-shared
  **best-of-N** (`RolloutFactory.best_of_n`, 4.5× savings sharing one plan prefix);
  AutoResearch **`stale_count`** → `generate_until` (stop on pass or stalled progress);
  smarter **compaction** (keep user turns verbatim, summarize assistant/tool work,
  archive dropped). `tools/gen_rollouts.py` writes **verifier-gated SFT traces** (keep
  only reward==1, R1 cold-start data) — offline via mock, live via any `agent/model.py`
  provider.
- **Next (live):** point `gen_rollouts --model vllm` at the served Spark model to
  bulk-generate ≥10k math + ≥10k code + physics verified traces for Phase 3.

### Phase 2 — Physics verifier substrate (CPU, ~2 wks) ★C — ✅ **landed**
- ✅ `agent/units.py` (pure-Python SI dimensional-analysis engine, no deps),
  `agent/physics_verifier.py` (dimensional → numeric → symbolic tiers),
  `physics_equivalent` + `physics_sound` in `agent/verifiers.py`,
  `provenance_bench/physics_reward.py` + `physics_dataset.py`.
- ✅ `provenance_bench/data/physics_problems.json` (20 problems, 10 families,
  fixed train/eval family-disjoint split); `--task physics` wired into
  `tools/run_rlvr.py`; 23 tests in `tests/test_physics_verifier.py`.
- ✅ **Exit gate met:** offline invariants pass via `run_rlvr.py --task physics
  --dry-run` — including the physics-specific `wrongUnitNegative` trap (right
  number, wrong dimension → −1 reward) and `contaminationFree`.
- **Next:** broaden the pack (more families + traps), add per-problem `rtol`, and
  add a sealed holdout via `holdout_seal.py`.

### Phase 3 — Live RLVR run — **decided: ≤8B, smoke-first, on DGX Spark** ★E
**Plan locked** (author, 2026-06): target **≤8B** (`Qwen/Qwen2.5-7B-Instruct`,
Apache-2.0), **smoke run first** on the author's **DGX Spark** (GB10, 128 GB unified,
aarch64) — not rented cloud. Full step-by-step in
[DGX-Spark-Smoke-Run-Runbook.md](./DGX-Spark-Smoke-Run-Runbook.md).
- ✅ **Repo readied for ≤8B non-GLM:** `tools/run_rlvr.py` now resolves LoRA target
  modules by model family (`resolve_target_modules` — Qwen/Llama split gate/up vs
  GLM fused), and adds `--max-steps` to bound a smoke run. Spark guidance: **bf16
  (skip 4-bit), `--vllm none`, no flash-attn** (aarch64/Blackwell dep hygiene).
- **Tier A (smoke):** `run_rlvr --task math --model Qwen/Qwen2.5-7B-Instruct
  --quant bf16 --vllm none --max-steps 30 …` → mean reward must trend up.
- **Tier B (first delta):** serve base vs. merged adapter, `run_baseline` before/
  after on the **same sealed eval hash** (Phase 0 artifacts).
- **Exit gate:** `passAt1` after > before with separated Wilson CIs on the sealed
  holdout for ≥1 domain → moves the failure-ledger "no live RL run" item toward
  closed (full closure needs the multi-seed, second-judge gated run).

### Phase 4 — Lean formal physics/math (optional, parallel, research) ★D,☆F — 🟡 **on-ramp wired**
- ✅ `agent/physics_verifier.verify(..., use_lean=True, lean_proof=...)` delegates to
  `agent.lean_backend` exactly like the math verifier — abstains fail-closed when
  lean-dojo is absent (CI default), so a physics theorem can be *formally* checked
  the moment a Lean physics library is wired, with zero behavior change until then.
- **Next:** stand up `formal_proofs/eval/` against PhysLib/Lean4Physics; formal reward
  for a problem subset. Optionally synthesize PRM data from formal verification (☆F).
- **Exit gate:** N formally-verified physics theorems passing in CI; reported as
  *coverage*, not a capability claim.

### Phase 5 — External benchmarking + publication (CPU, ~1–2 wks) — 🟡 **physics harness in**
- ✅ Physics external scorer + pack: `agent/external_eval.score_item_physics`
  (dimensional + numeric, judge-free) and `eval/external/physics-style-sample.jsonl`
  (10 items, units-bearing golds, self-consistent). Joins the existing MATH-style /
  MBPP-style sample packs.
- **Next:** run the trained adapter on **external** suites: MATH-500, GSM8K, AIME-style,
  MBPP/HumanEval, SWE-bench-lite (subset), and a real physics set (LeanPhysBench /
  GPQA-physics / SciBench-physics) via the scorer above.
- Score through `provenance_bench/runner.py` + `score.py` (≥2 judge families, κ);
  decontaminate with `eval/contamination.py`; seal with hidden-reviewer packs.
- **Exit gate:** numbers that pass the no-overclaim gate land in
  `agi-proof/benchmark-results/` + `RESULTS.md` with CIs. *These are the
  "benchmarkable, top-notch" numbers you asked for.*

### Phase 6 — Close the self-extension loop (research) ☆G,☆H
- Wire `selfextend` promotions to trigger an RLVR fine-tune; expert-iteration via
  MCTS-filtered traces (☆H). Highest-leverage, most speculative.

---

## 5. Benchmarking plan (how "top-notch" gets *proven*, not claimed)

| Domain | Internal verifier (reward) | External benchmark (proof) | Judge families |
|---|---|---|---|
| Math | sympy `math_sound`, Lean | MATH-500, GSM8K, AIME-subset | symbolic + LLM-judge ×2 |
| Code | sandboxed `code_tests_pass` | MBPP, HumanEval, SWE-bench-lite | exec + LLM-judge |
| Physics | `physics_sound` (dim/num/sym), Lean physics | GPQA-physics, SciBench, LeanPhysBench | numeric + LLM-judge ×2 |

Rules (already enforced by your repo, keep them): sealed holdouts, decontamination
before every run, κ reported, ≥3 seeds, 95% CI, no self-judging, failures logged in
`failure-ledger.md`. A model is "top-notch" only when an external suite's CI
clears the published SOTA band — anything else is "candidate."

---

## 6. Concrete first PRs (smallest valuable steps)

1. `pipeline/rollout/` skeleton (planner/executor + prepend-only sessions) — ★A/B.
2. `agent/physics_verifier.py` + `physics_sound` wiring + trap controls — ★C.
3. `data/physics_problems.json` + `provenance_bench/physics_reward.py` +
   `physics_dataset.py` — feeds Phase 3.
4. `docs/09-Agent/RLVR-Experiment.md` update with the physics task + GPU runbook.
5. Baseline benchmark run + sealed table in `agi-proof/benchmark-results/` — Phase 0.

---

## 7. Open questions for the author

- The "crazy concept in another Claude session" isn't visible here — paste its
  thesis and this roadmap can be tuned to it (esp. §3 ☆ items).
- GPU budget/timeline for Phase 3? RunPod MCP is wired; a single A100/H100 burst is
  enough for a QLoRA GRPO run on a small (≤8B) model.
- Target model size? The repo's validated work is on an 8B local model; "top-notch
  on external SOTA" at ≤8B is realistic for *math/code*, ambitious for *physics*.

---

*Cross-refs:* [Corpus-MathCode-Capability-Roadmap.md](./Corpus-MathCode-Capability-Roadmap.md),
[Formal-Proofs-Eval-Design.md](./Formal-Proofs-Eval-Design.md),
[Reasoning-As-Compute.md](./Reasoning-As-Compute.md),
[../../agi-proof/failure-ledger.md](../../agi-proof/failure-ledger.md).
