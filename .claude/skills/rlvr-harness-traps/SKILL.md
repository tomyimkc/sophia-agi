---
name: rlvr-harness-traps
description: >
  Run when an RLVR / eval-harness number looks wrong or "too clean" BEFORE you act on it — a 0/N
  pass rate on BOTH base and adapter, per-seed metrics that won't reconcile across a sweep, a
  "broken verifier" (passAt1 0/0 or VSC), a PEFT "Target modules {...} not found", or a real uplift
  recorded as NO-GO. Most of these are MEASUREMENT ARTIFACTS, not capability or idea failures. Use to
  diagnose the artifact, pick the metric that is actually load-bearing (passAt1 / VSC, not
  meanReward), pin a multi-seed sweep for comparability, and uphold the no-overclaim rule — the
  HARNESS is the deliverable; canClaimAGI stays false.
metadata:
  short-description: "Diagnose RLVR/eval-harness artifacts: 0/N-both, chat-template, sweep comparability, step-RLVR metrics, all-linear footgun, no-overclaim"
---

# rlvr-harness-traps

When an RLVR number looks wrong, suspect the **harness** before the model. Today's incidents (all
verified against the code) map to these traps. Companion to **spark-cluster-ops**.

## A. The 0/N-on-BOTH-sides diagnostic  (check this FIRST)
- **Symptom:** a CODE RLVR run scores base 0/48 AND adapter 0/48 → gate quarantine. Looks like a
  dead/weak model; almost always a **no-chat-template artifact**: raw prompts fed to a chat/instruct
  base → it emits prose → the tests-pass reward (`code_reward`→`code_exec`) finds no fenced
  ` ```python ` block → reward 0 for EVERYTHING.
- **RULE:** ALWAYS verify **base passAt1 ≫ 0** before reading any uplift. If base==adapter==0 on a
  chat model's code task, suspect the template, not the model.
- **Fix in tree (PR #263):** `provenance_bench/code_dataset.py::chat_wrap(tokenizer, prompt)` —
  no-op when `chat_template is None` (base/completion model), else `apply_chat_template(
  add_generation_prompt=True)`. Applied on BOTH sides, **CODE task only**: eval
  `tools/eval_rlvr_adapter.py` (run_eval_code), train `tools/run_rlvr.py`. Secondary cause: a workflow
  with no `model` input silently trained the wrong default base — set it explicitly (Qwen2.5-Coder-7B-Instruct).

## B. Which tasks get the template (don't over-fix)
- `chat_wrap` iff the base is a CHAT model AND emission must be STRUCTURED (code). **STEP / MATH /
  PROVENANCE / CONCEPT stay RAW** (`chat_template=False`); the step path prepends `STEP_INSTRUCTION`
  to a raw prompt and a base/completion model needs no template. `chat_wrap` is itself a no-op for a
  completion tokenizer, so "fixing" the step task with it would be wrong.

## C. Pin-for-comparability — a multi-seed sweep is INVALID otherwise
- One mid-sweep `main` change silently mixed the reported metric across pods: `ingest_rlvr_eval.py`
  picks `capabilityMetric` BY TASK — math/code → `passAt1` ([0,1]); else → `meanReward` ([-1,1]).
  The 'code' arm was ADDED mid-sweep; a pod that checked out `main` BEFORE it fell through to
  meanReward, after got passAt1 → the "0.58 vs 0.0 base variance" was a **metric mismatch**, not
  difficulty/non-determinism → a combined across-seed CI is **invalid**.
- **RULES for any sweep:** (1) **pin ONE commit** — `runpod_rlvr.py --source git --branch <sha>` so
  every pod shares code; (2) report the **same metric** (passAt1 for code); (3) pin `--vllm` mode AND
  `--num-generations` (GRPO advantages come from the reward SPREAD over the num_generations group —
  different group size = different estimator = not comparable; QLoRA-4bit + colocate is refused).
  (4) **temp=0** for headline judging (a 70B judge "0.787" was a temp=0.2 draw; true 0.713 at temp=0).
  Never change `main` mid-sweep; **re-run clean rather than clearing a candidate** on mixed metrics.

## D. Step-RLVR: the right metric is passAt1 / VSC, NOT meanReward (and VSC is gameable)
- **passAt1 0/0 is a GENUINE FLOOR, not a broken verifier:** the step verifier DID engage (VSC
  non-zero, e.g. 0.158→0.138). `passAt1` counts ONLY `verdict=='accepted'`; any unverifiable
  transition aggregates to `abstain` → passAt1=0 is real (the model can't produce fully
  machine-verified derivations on unseen families).
- **meanReward is an ABSTENTION ARTIFACT** — the mean of per-check values where abstain=0, dominated
  by abstentions. **Never quote meanReward as the step capability number** (a +0.024 meanReward shift
  with VSC moving *down* is not success).
- **Headline honesty metric = verifiedStepCoverage (VSC)**; gate on the **passAt1 delta**
  (verified-correct rate). The REGISTERED metric is the arbiter, not the artifact.
- **VSC IS GAMEABLE:** an answer-only derivation (one step = the gold, 0 transitions) scores VSC=1.0
  and reward +1 → GRPO can learn to DROP the work. Worse, step rows omit the completion text
  (`eval_rlvr_adapter.py:402` has no `'completion'` key, unlike the code/provenance paths) → you
  can't audit it. **Mitigation:** pair VSC with a minimum-work/step-count requirement, gate on passAt1
  + no-regression, and add `'completion': text` to the step row at `eval_rlvr_adapter.py:402`.

## E. `--target-modules all-linear` footgun (MoE/QAT)
- On OLMoE with offloaded/meta params, PEFT raises `Target modules {...} not found` where the set is
  the unique CHARACTERS of the string (`{'r','a','i','-','n','l','e'}` = letters of "all-linear"):
  `_resolve_target_modules` returns the LITERAL string; all-linear expansion needs MATERIALIZED
  `nn.Linear`, finds none on meta/offload, and the string is iterated as single-char module names.
- **FIX:** pass explicit `--target-modules attn-mlp` → `ATTN_MLP_MODULES` = q/k/v/o_proj,
  gate/up/down_proj (the proven OLMoE-cert recipe). For MoE/QAT, confirm the experts actually carry a
  LoRA adapter (train-time guards WARN when 0 expert modules are adapted).

## F. The no-overclaim rule — the registered metric is not the artifact; the HARNESS is the deliverable
- A real, powered, CI-excludes-0 uplift is STILL a NO-GO if it trips an integrity gate. Code-Integrity
  RLVR (#280): Qwen2.5-Coder-7B GRPO mean Δ +0.135, bootstrap CI [+0.107, +0.168] — yet DISQUALIFIED
  because the hard gate is `reward-hack-rate==0` and 2/3 seeds accepted reward-hacks. Uplift that also
  learned to game the grader is correctly failed.
- NVFP4 low-RAM cert is NO-GO (v5 top1 0.883 < 0.97, mean_kl 0.0506 > 0.05; v3 still best at 0.906).
  Open lever: fused-expert QAT-compose + cert-merge (the 32 fused-expert ParamWrapper LoRA modules
  train BLIND to the NVFP4 grid and the manual-merge fallback skips them on version skew) — needs
  on-Spark dev against the real PEFT ParamWrapper.
- **BOTTOM LINE:** report passAt1/VSC honestly, name the gate that failed, keep `canClaimAGI=false`.
  A clean harness that produces an honest NO-GO is a SUCCESS; a green number from a comparability
  break or a gamed reward is not.
