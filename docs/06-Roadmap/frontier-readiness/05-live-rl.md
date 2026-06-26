# 05 — Live RL Weight-Update Loop (RLVR, verifier-gated)

**Owner:** staff RL research engineer
**Repo:** `/home/user/sophia-agi`
**Target OPEN item:** README — *"a live RL weight update is OPEN (needs GPU)"*; failure ledger
`rlvr-live-run-not-yet-gated-2026-06-21` (provenance) — **Open**.
**Date:** 2026-06-26

> **One-line thesis.** Sophia's differentiator is *verifiable rewards*: a deterministic,
> fail-closed gate that already returns a bounded scalar reward. The honest way to close the
> live-RL rung is **GRPO with the repo's own gate/verifier as the reward** (RLVR), on a small
> LoRA-tuned model, on a CUDA pod, measuring a **held-out before/after eval delta with CIs** that
> clears the existing no-overclaim gate (`provenance_bench.aggregate._is_validated`) — and proving
> the conscience/deontic gate catches reward hacking, not just trusting that it does.

---

## 1. Thesis & references

### 1.1 The RL post-training landscape (cite by name)

- **RLHF / PPO** (Christiano et al. 2017, *Deep RL from Human Preferences*; Ouyang et al. 2022,
  *InstructGPT*; Schulman et al. 2017, *PPO*). The canonical pipeline: SFT → reward model (RM)
  trained on human preference pairs → PPO against the RM with a **per-token KL penalty** to a frozen
  reference policy. Strengths: general. Weaknesses: a *learned* RM is itself hackable (Goodhart),
  PPO needs a value/critic head and is memory-heavy and finicky.
- **DPO** (Rafailov et al. 2023, *Direct Preference Optimization*). Closed-form reparameterization
  that turns the RLHF objective into a supervised classification loss over `(prompt, chosen,
  rejected)` triples — **no RM, no sampling loop, no critic**. The implicit reward is
  `β·log[π_θ/π_ref]`; KL to ref is built into the loss. Cheap, stable. Weakness: off-policy / fixed
  preference set; cannot discover new high-reward behavior beyond the pairs you mined.
- **ORPO** (Hong et al. 2024, *Odds Ratio Preference Optimization*). **Single-stage, reference-free**:
  folds an odds-ratio preference penalty into the SFT loss, so no separate SFT→DPO two-stage and no
  reference model in memory. Good for small-GPU preference tuning.
- **GRPO** (Shao et al. / DeepSeek 2024, *DeepSeekMath*; DeepSeek-R1 2025). **Critic-free PPO
  variant**: sample a *group* of G completions per prompt, score each with the reward, and use the
  **group-relative advantage** `A_i = (r_i − mean(r)) / std(r)` instead of a learned value baseline.
  Halves the memory of PPO (no value model), and is the workhorse behind verifiable-reward reasoning
  RL. This is the algorithm already wired in this repo.
- **RLVR / RL with Verifiable Rewards** (Lambert et al. 2024, *Tülu 3*; DeepSeek-R1 2025;
  Lightman et al. 2023 *Let's Verify Step by Step* for the process-vs-outcome framing). Replace the
  *learned, hackable* RM with a **deterministic verifier**: math answer-equality (SymPy), code unit
  tests (execution), or — uniquely here — a **provenance/citation gate**. The reward is *checked, not
  predicted*, which structurally removes the RM-hacking failure mode. This is Sophia's natural fit.
- **Reward modeling vs rule/verifier rewards.** A learned RM generalizes but invites reward hacking
  and needs preference data; a verifier reward is narrow but **non-gameable when the verifier itself
  is sound and the answer ≠ the question** (the repo's `math_equivalent` reward is judge-free for
  exactly this reason). Sophia's stance: prefer verifier rewards; never let an LLM-judge grade its own
  training targets (see `gate_reward.py` docstring on avoiding the attribution *trap-grader*).
- **Reward hacking** (Amodei et al. 2016, *Concrete Problems in AI Safety*; Skalse et al. 2022,
  *Defining and Characterizing Reward Hacking*; Pan et al. 2022 reward-misspecification). The
  failure: the policy maximizes the proxy while violating intent. Sophia's specific instance is
  **abstention collapse** — a naive `{clean:+1, abstain:0, violation:−1}` reward teaches the model to
  stop abstaining and start guessing, eroding the fail-closed behavior the whole project exists to
  protect. The repo's *implemented* defenses: **reward-positive abstention**
  (`gate_reward.REWARD_ABSTAIN = +0.5 > 0`), the **deontic/conscience gate** (`agent/conscience.py`,
  `agent/deontic_verifier.py`, `agent/reward_isolation.py`) as a hard pre-/post-filter on
  reward-tampering and verifier-tampering, and the closed-loop **non-degeneracy invariant**
  (`agent/closed_loop.py`: a promoted model whose uplift goes negative → HALT, roll back).
- **KL control & online vs offline.** PPO/GRPO add a KL-to-reference term (β) to keep the policy from
  drifting into degenerate high-reward text; too-low β invites reward hacking, too-high β freezes
  learning. *Online* RL (GRPO/PPO) samples fresh completions each step (can discover new behavior,
  expensive); *offline* methods (DPO/ORPO, and Sophia's current "RLVR-as-selection") reuse a fixed
  dataset (cheap, bounded). **Sophia today does offline *selection*; this plan adds the online
  *weight-update*.**
- **Libraries (cite by name).** **TRL** (HuggingFace) — `GRPOTrainer`, `DPOTrainer`, `ORPOTrainer`,
  already imported in this repo. **verl** (ByteDance) — scalable HybridFlow PPO/GRPO for larger fleets.
  **OpenRLHF** — Ray + vLLM + ZeRO distributed RLHF. **Unsloth** — fused kernels / 4-bit, ~2× single-GPU
  LoRA/QLoRA throughput. **vLLM** — fast rollout generation (colocate or server mode), the rollout
  bottleneck of any online RL loop. **PEFT/bitsandbytes** — LoRA + 4-bit QLoRA so this fits one GPU.

### 1.2 Why this thesis fits *this* repo

Sophia already has the three RLVR ingredients wired and offline-validated:
1. **A bounded scalar reward from a verifier** — `agent/gate_reward.py::reward → [−1,1]` and
   `provenance_bench/{rl_reward,math_reward,code_reward}.py`, all with `make_grpo_reward()`
   TRL-compatible factories and offline invariants (deterministic, monotone, bounded, verifier-seam
   invoked, contamination-free split).
2. **A live GRPO runner** — `tools/run_rlvr.py::_run_gpu` constructs `GRPOConfig`/`GRPOTrainer` with
   QLoRA `LoraConfig`, vLLM colocate/server, and calls `trainer.train()`.
3. **A held-out before/after evaluator** — `tools/eval_rlvr_adapter.py` (base vs PEFT adapter on the
   entity-/family-disjoint holdout, with FP-regression self-diagnosis).

The remaining work is **not building the loop** — it's **running the provenance arm to the gated
bar, hardening reward-hacking analysis, and reporting honestly.** The math arm already cleared its
rung (see §2).

---

## 2. Current repo state (honest — what is offline-*selection* today)

### 2.1 What is genuinely OFFLINE SELECTION (no weights move)
- **`selfextend/loop.py`, `selfextend/evolve.py`** — the "self-extending flywheel": abstain →
  synthesize verifier → validate on held-out → use verifier as **verified-reward *selection*** (best-of-N
  / rejection sampling) → lift policy accuracy 0.5→1.0 on an independent split. **No `backward()`.** This
  is what the README means by "offline *selection*, not parameter updates."
- **`provenance_bench/async_rl.py`** — real async scheduling + staleness-bounded replay buffer +
  `grpo_advantages()`, but `generate_fn`/policy-update are a *synthetic improvement proxy*; the
  scheduling is real, the weights are not touched.
- **`provenance_bench/governed_rl.py`** — fail-closed admission policy for trajectories (reward earned
  ∧ gate passes ∧ grounded ∧ not stale). Gates *what may enter training*; updates nothing.
- **`agent/closed_loop.py`** — the orchestrator for measure-uplift → distill pairs → train → gate →
  re-measure, with `non_degenerate`/`saturation` invariants. **The train step is INJECTED**; the CI
  default `noop_train_step` returns `ran=False`. The live seam is documented as
  `tools/run_rlvr.py` on a CUDA pod.

### 2.2 What is REAL weight-update code, already present (verified)
- **SFT:** `tools/train_lora.py` (manual loop: `loss.backward()`, `optimizer.step()`,
  `scheduler.step()`, grad-clip, completion-only masking; PEFT / **Unsloth** / **MLX** backends; 4-bit
  QLoRA).
- **Preference:** `tools/train_dpo.py` (`TRL DPOTrainer.train()`), `tools/train_orpo.py`
  (`TRL ORPOTrainer.train()`).
- **Online RL:** `tools/run_rlvr.py::_run_gpu` (`TRL GRPOTrainer.train()` with `reward_funcs` =
  gate/verifier/math reward, QLoRA LoRA on GLM-4-9B, vLLM colocate/server).
- **Orchestration:** `tools/runpod_rlvr.py` / `tools/runpod_train.py` (dependency-free RunPod REST
  orchestrators: create pod → SSH → stream train → copy report → **delete pod in `finally`** + remote
  delete watchdog). RunPod MCP is configured in `.mcp.json`.

### 2.3 What live runs have ALREADY happened (the honest current frontier)
- **MATH RLVR — rung CLEARED** (ledger `rlvr-math-live-run-not-yet-run-2026-06-24`, status
  *Cleared (rung)*). Live GRPO on **GLM-4-9B**, vLLM colocate (trl 0.19.1 + vllm 0.9.1), RunPod A100
  80GB, **3 seeds**, N=60 family-disjoint held-out, **judge-free SymPy reward**: base **0/60 every
  seed** → adapter 5–7/60, **mean Δ +0.10, 95% across-seed CI [0.059, 0.141] excludes 0**,
  contamination-free, no regression. Evidence: `agi-proof/self-extension/math-rlvr-3seed-n60/`.
  *Honest scope:* modest/narrow (~10% where base floors at 0%), judge-free so it clears the **rung**
  but is explicitly **not** the multi-judge `_is_validated` headline.
- **PROVENANCE RLVR — still OPEN** (ledger `rlvr-live-run-not-yet-gated-2026-06-21`). Offline reward
  invariants pass in CI; **no gated live run** clearing `aggregate._is_validated` (notMock ∧ ≥2 judge
  families ∧ κ≥0.40 ∧ ≥3 runs ∧ CI excludes 0) on the entity-disjoint split + manual semantic review.

### 2.4 The precise gap to close
The *machinery* is built and the *math rung* is cleared. The OPEN item is the **provenance arm at the
full validated bar**, plus a **first-class reward-hacking analysis** that demonstrates (not asserts)
the conscience/deontic gate catches a hacked policy. The README sentence is now *understated*: the
honest statement is "a live RL weight update has cleared a judge-free math rung; the validated,
multi-judge provenance RLVR run is OPEN."

---

## 3. Top-tier target end-state (frontier-lab bar)

A reproducible, fail-closed, **online** RLVR loop where:
1. **One command** rents a GPU, trains a LoRA adapter with GRPO against the verifier/gate reward,
   evaluates base-vs-adapter on an **entity-disjoint held-out split**, and tears the pod down — with
   the **adapter weights' SHA-256, config hash, and dataset manifest** registered in
   `agi-proof/mlops/checkpoint-registry.json`.
2. The **provenance arm clears `_is_validated`**: ≥3 seeds, ≥2 independent judge families on the
   semantic axis, κ≥0.40, 95% CI on the before/after delta excludes 0, **no FP-regression on
   true-author cases**, and a manual semantic review pass.
3. **Reward-hacking is measured, not trusted**: a deliberately mis-shaped reward (e.g. abstention set
   to 0) is shown to (a) collapse abstention and (b) be **caught and rolled back** by the deontic /
   reward-isolation / non-degeneracy gate — an adversarial control that proves the guard works.
4. **Curves are logged**: reward vs step, KL-to-ref vs step, completion-length vs step (length-hacking
   detector), pass@1 vs step — exported as a public-aggregate artifact.
5. Every claim carries `candidateOnly`/`canClaimAGI:false` discipline and links the failure ledger.
   *Saturation or a null result is a valid, logged outcome* — not hidden.

This matches the frontier-lab pattern (DeepSeek-R1 GRPO-with-verifiable-reward; Tülu-3 RLVR; the
"RL Engineering / RL Scaling Science / Production Post-Training" bars): **online RL, verifier reward,
KL-controlled, reproducible, with explicit reward-hacking analysis and honest CIs.**

---

## 4. Phased plan — milestones, file paths, libraries

> Each milestone is a falsifiable go/no-go. *A null/within-noise result is a legitimate logged
> outcome, recorded in the failure ledger; it is not a reason to weaken a gate.*

### Milestone 0 — Reproduce the offline contract (no GPU; today, any machine)
- **Do:** `python tools/run_rlvr.py --model mock` (provenance reward wiring + gate-reward invariants),
  `python tools/run_rlvr.py --model mock --task math`, `python agent/gate_reward.py` (self-check),
  `python tools/eval_rlvr_adapter.py --mode mock` and `--task math --mode mock`.
- **Gate:** all offline invariants green (deterministic, monotone, bounded, verifier-seam invoked,
  contamination-free split; gate-reward `violation < abstain < clean`, `abstain > 0`).
- **Libraries:** none (pure Python). **Files:** `tools/run_rlvr.py`, `agent/gate_reward.py`,
  `tools/eval_rlvr_adapter.py`, `provenance_bench/{rl_reward,math_reward,aggregate}.py`.
- **Effort:** 0.5 day. **Output:** confirms the seam before any GPU spend.

### Milestone 1 — ★ FIRST REAL WEIGHT UPDATE: one gated GRPO run, before/after with CIs ★
*The single highest-signal deliverable.* Pick the **PROVENANCE** arm (math rung is already cleared;
provenance is the OPEN ledger item and the repo's core differentiator). The verifiable reward is the
gate (`--reward gate`, abstention-positive) — judge-free for the *training* signal; LLM judges are
used only for the *eval* semantic axis (never to grade training targets).

- **Compute:** 1× A100 80GB (bf16 colocate) **or** 2×24GB (vLLM server). LoRA/QLoRA on GLM-4-9B.
- **Train:**
  ```bash
  export RUNPOD_API_KEY=...                 # RunPod MCP / runpod_rlvr.py
  python tools/runpod_rlvr.py --yes \
     -- --task provenance --reward gate \
        --model zai-org/glm-4-9b-chat-hf --quant bf16 --vllm colocate \
        --epochs 3 --num-generations 8 --beta 0.04 --lr 1e-5 --seed 0
  ```
  (repeat `--seed 1`, `--seed 2`).
- **Eval (the delta):**
  ```bash
  python tools/eval_rlvr_adapter.py --mode real --task provenance \
     --model zai-org/glm-4-9b-chat-hf --adapter <ckpt> --seed {0,1,2}
  ```
- **Aggregate + gate:** `provenance_bench/aggregate.py::_is_validated` (notMock ∧ ≥2 judge families ∧
  κ≥0.40 ∧ ≥3 runs ∧ CI excludes 0); add the **multi-judge semantic re-score** (mirror
  `tools/run_calibration_judge.py`) using ≥2 distinct vendor families (e.g. OpenAI + Anthropic, both
  ≠ the subject model). **Note the live blocker (ledger):** judge keys (`OPENAI/ANTHROPIC/...`) and
  `HF_TOKEN` were absent in prior sessions — provision these *before* the run or the κ/2-family step
  cannot execute and only the rung (judge-free) bar is reachable.
- **Curves & integrity:** log reward, KL-to-ref, completion length, pass@1 per step (TRL
  `logging_steps`); assert no length-blowup; record `trueFalsePositiveRate` delta ≤ 0
  (`eval_rlvr_adapter.false_positive_regressions` names which cases flipped).
- **Register:** write the adapter to `agi-proof/mlops/checkpoint-registry.json` (weightsSha256,
  trainingConfigHash, datasetManifest, evalArtifacts, promotionVerdict, failureLedgerRef).
- **Gate (go):** mean Δmean-reward > 0 across 3 seeds, 95% CI excludes 0, **no FP-regression**,
  contamination-free; semantic axis multi-judge κ≥0.40. **Update ledger**
  `rlvr-live-run-not-yet-gated-2026-06-21` → *Cleared* (or record the honest within-noise/null).
- **Effort:** 3–5 days incl. judge provisioning + flakiness. **Output:** the closed OPEN item —
  *a real RL weight update with a validated held-out before/after delta and CIs.*

### Milestone 2 — Scale & strengthen the signal
- **Larger held-out N** (push the provenance pack from the small entity-disjoint split toward
  100–300 cases via `provenance_bench/rl_dataset.py`), **more epochs / generations**, and a **second
  reward arm** (math is done; add **code** via `provenance_bench/code_reward.py` + execution
  verifier as a structurally-different verifiable task → demonstrates the loop is reward-general, not
  provenance-special).
- **KL/β sweep** (β ∈ {0.0, 0.02, 0.04, 0.1}) to chart the reward-vs-KL frontier and pick the
  hacking-resistant operating point; **curriculum** (`run_rlvr.py --curriculum`, easy→hard by gate
  pass-rate).
- **Throughput:** switch the LoRA backend to **Unsloth** for ~2× on single GPU; keep vLLM colocate
  for rollouts. Consider **verl/OpenRLHF** only if scaling past one pod (documented, not required).
- **Gate:** Δ holds at larger N with CI still excluding 0; per-β curve reported.
- **Effort:** 1–2 weeks (GPU-bound). **Output:** a robust, not-knife-edge result + a reward-generality
  demonstration across ≥2 verifier families.

### Milestone 3 — Reward-hacking analysis via the conscience/deontic gate (adversarial control)
*The frontier-lab differentiator: prove the guard, don't assert it.*
- **Adversarial reward A — abstention collapse:** retrain with the *naive* shape
  (`abstain=0`) and show (a) abstention rate collapses / fabrication rises on the held-out set, and
  (b) the **non-degeneracy invariant** (`agent/closed_loop.py`) + the eval FP-regression check flag
  and **roll it back**. Contrast with the shipped reward-positive-abstention shape that does not.
- **Adversarial reward B — verifier/reward tampering:** feed a trajectory that attempts to edit the
  reward path or fabricate a citation through `agent/reward_isolation.py::evaluate_reward_isolation`
  and `agent/deontic_verifier.py::check_deontic`; show **quarantine/reject**, not promote.
- **Length / format hacking:** show the completion-length curve and a gate-clean-but-vacuous probe
  are floored at the abstain reward (`gate_reward` already floors sub-12-char completions).
- **Gate:** every hacked policy is **caught and rolled back**; a table of (attack → detector →
  verdict). **Output:** a reproducible reward-hacking report — the honest evidence that RLVR here is
  *safe-by-construction*, not lucky.

### Milestone 4 — Wire the live trainer into the closed loop (optional, higher bar)
- Replace `closed_loop.noop_train_step` with a real `TrainStep` that shells out to
  `tools/run_rlvr.py` on a pod and returns the checkpoint spec, so
  `agent/closed_loop.py::run_closed_loop` runs **measure-uplift → distill → GRPO → plasticity-gate →
  re-measure** end-to-end on real weights, with the non-degeneracy/saturation invariants live.
- **Gate:** ≥1 cycle promotes on a clean gain *or* saturates honestly; non-degeneracy never fires
  falsely. **Effort:** 1 week. **Output:** the model↔harness co-evolution loop closed on real weights.

---

## 5. Compute / budget tiers

| Tier | Hardware | Method | Per-run cost (RunPod) | Use |
|---|---|---|---|---|
| **T0 Offline** | CPU / Apple Silicon | mock reward-wiring + invariants | $0 | Milestone 0; CI; pre-flight |
| **T1 Single small GPU** | 1×24GB (4090/A5000) | QLoRA 4-bit + `--vllm none` (slow) or `--vllm server` on a 2nd 24GB; **Unsloth** | ~$0.3–0.6/hr | Smallest viable GRPO; long, but cheapest live update |
| **T2 Single 80GB (recommended M1)** | 1×A100/H100 80GB | LoRA bf16 + **vLLM colocate** | ~$1.6–3.3/hr; one provenance seed ≈ 0.5–2 GPU-hr | Milestone 1/2 main path (matches the proven math runs) |
| **T3 Two GPUs** | 2×24–48GB | QLoRA on GPU0 + **vLLM server** on GPU1 | ~$1–2/hr | Avoids the QLoRA(4-bit)+vLLM-colocate crash (trl#4973; `run_rlvr.py` refuses that combo) |
| **T4 Fleet (out of scope to run)** | 4–8×80GB | **verl / OpenRLHF** Ray+vLLM | $$$ | Only if scaling past GLM-9B / one pod |

**Full FT** is explicitly out of scope: LoRA/QLoRA is the honest, reproducible, single-GPU path and is
what every existing artifact uses. **Budget for Milestone 1:** ~6 GPU-hours (3 seeds train+eval +
flakiness) ≈ **$15–30** on a T2 A100. Always use the auto-delete orchestrator (`runpod_rlvr.py`
deletes the pod in `finally` + watchdog) to avoid idle burn.

---

## 6. Honest metrics (what gets reported)

- **Primary:** held-out **before/after delta** — provenance `meanReward` & `pass@1`; math/code
  `pass@1` — with **95% bootstrap CI across ≥3 seeds** (`provenance_bench/aggregate.py`). Headline only
  if CI excludes 0 **and** `_is_validated` (≥2 judge families, κ≥0.40, ≥3 runs, notMock).
- **Integrity guardrail:** `trueFalsePositiveRate` delta ≤ 0 (no regression on true-author cases);
  `false_positive_regressions` lists any case that flipped (diagnosable, not just an aggregate).
- **Training health:** reward-vs-step curve (monotone up, then plateau), **KL-to-ref vs step** (bounded,
  not exploding), completion-length vs step (no length-hacking), gradient norm.
- **Reproducibility:** seeds [0,1,2], `trainingConfigHash`, dataset SHA-256, `weightsSha256`,
  contamination-free (`entity_intersection == []` / `family_intersection == []`) asserted in-report.
- **No reward-hacking:** the Milestone-3 attack→detector→verdict table; abstention rate must **not**
  fall under the shipped reward; any hacked variant must be **caught + rolled back**.
- **Discipline:** every artifact carries `candidateOnly:true`, `canClaimAGI:false`, `level3Evidence`
  only if a private third-party hidden suite is used; ledger row updated either way.

---

## 7. Risks & overclaim guards

- **Overclaiming a narrow win as AGI/general.** Guard: the math rung is honestly "~10% where base
  floors at 0%"; the provenance arm must keep the same tone. Headline only what `_is_validated`
  passes; everything else is *illustrative/candidate*.
- **Judge / key availability (a real, recurring blocker).** Ledger shows prior runs blocked by absent
  `HF_TOKEN`, `RUNPOD_API_KEY`, and judge keys. Guard: provision and verify all keys in Milestone 0;
  if only the subject vendor's key exists, you can clear the **judge-free rung** but **not** the
  multi-judge `_is_validated` bar — say so explicitly.
- **Reward hacking / abstention collapse.** Guard: shipped reward-positive abstention + deontic gate +
  non-degeneracy halt + the Milestone-3 adversarial control that *demonstrates* the catch.
- **Teaching-to-the-test / contamination.** Guard: entity-/family-disjoint splits with in-report
  intersection assertion; reuse the `seib_generalization_split.py` corpus-clean discipline; never
  patch per-entity to pass.
- **QLoRA(4-bit)+vLLM-colocate crash (trl#4973).** Guard: `run_rlvr.py` already *refuses* the combo;
  use `--quant bf16` colocate (1×80GB) or `--vllm server` (2 GPUs).
- **Within-noise / null result.** Guard: this is a legitimate logged outcome (cf. the GSM8K-style and
  distribution-shift null rows in the ledger). Record it; do not weaken a gate to manufacture a pass.
- **LLM-judge grading training targets.** Guard: gate reward is *question-free* by design to avoid the
  positive-expectation trap-grader; judges touch only the held-out eval semantic axis.

---

## 8. Effort

| Milestone | Calendar | GPU | Risk |
|---|---|---|---|
| M0 Offline contract | 0.5 day | none | low |
| **M1 First gated weight update + CI delta** | **3–5 days** | ~6 A100-hr (~$15–30) | med (judge keys, flakiness) |
| M2 Scale + KL sweep + 2nd reward family | 1–2 weeks | 20–60 GPU-hr | med (GPU-bound) |
| M3 Reward-hacking adversarial control | 3–5 days | ~10 GPU-hr | low (machinery exists) |
| M4 Live trainer in closed loop | ~1 week | 10–20 GPU-hr | med |

**Critical path to closing the README OPEN item = M0 → M1.** Everything after is hardening, scale, and
the safety story. Total to a defensible, validated, reward-hacking-analyzed result: **~3–5 weeks**,
most of it GPU- and judge-provisioning-bound, not code-bound — because the loop is already built.

---

### Key files (absolute paths)
- Reward: `/home/user/sophia-agi/agent/gate_reward.py`,
  `/home/user/sophia-agi/provenance_bench/{rl_reward,math_reward,code_reward}.py`
- Train (online): `/home/user/sophia-agi/tools/run_rlvr.py` (`_run_gpu` → TRL `GRPOTrainer.train()`)
- Train (preference/SFT): `/home/user/sophia-agi/tools/{train_dpo,train_orpo,train_lora}.py`
- Eval (delta): `/home/user/sophia-agi/tools/eval_rlvr_adapter.py`
- Aggregate/gate: `/home/user/sophia-agi/provenance_bench/aggregate.py` (`_is_validated`)
- Orchestration: `/home/user/sophia-agi/tools/{runpod_rlvr,runpod_train}.py`; RunPod MCP in `.mcp.json`
- Reward-hacking guards: `/home/user/sophia-agi/agent/{closed_loop,conscience,deontic_verifier,reward_isolation}.py`
- Registry / evidence: `/home/user/sophia-agi/agi-proof/mlops/checkpoint-registry.json`,
  `/home/user/sophia-agi/agi-proof/self-extension/math-rlvr-3seed-n60/`
- Ledger item to close: `/home/user/sophia-agi/agi-proof/failure-ledger.md`
  (`rlvr-live-run-not-yet-gated-2026-06-21`)
