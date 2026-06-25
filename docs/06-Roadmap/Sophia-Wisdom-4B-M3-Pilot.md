# Sophia-Wisdom-4B — M3 PILOT (corpus-bound, pre-registered)

**Status:** pre-registered spec · **Supersedes** the full-scale M3 only for this pilot ·
Gated by the no-overclaim standard (`tools/lint_claims.py`, RESULTS.md VALIDATED bar).

> Why a pilot: M2's ≥10k-row gate is a **NO-GO** — the gate-passed set is **965 rows** and the
> shortfall is **corpus-bound** (~72 structured records; the builder dedups by prompt), not
> teacher/egress-bound. The plan's M3 ("one SFT, beat same-size baselines") presumes ≥10k. With
> 965 rows we cannot honestly target market-beating, so this pilot tests a **strictly narrower,
> falsifiable** question and pre-registers its thresholds BEFORE any GPU is spent.

This is **not** the full M3. It does **not** claim to beat same-size open models. `canClaimAGI`
stays False throughout.

---

## The one falsifiable question

> Does a LoRA SFT on the 965 gate-passed rows move **gemma-3-4b-it**'s *weights* so that its
> **prompt-scaffold (no-gate)** behavior shifts toward the *gated* target on ≥1 Sophia-native
> axis — **without** regressing the protected suites or general instruction-following?

Rationale: M1 already proved the **prompt+gate** wins on gemma (qualification +0.37\*,
tradition-merge +0.125\*, tool-route 0→0.86\*). The adapter's only honest job at this scale is to
**internalize some of those habits into the weights** so they appear with the scaffold *before*
the external gate fires — i.e. reduce reliance on the gate — at no protected/retention cost. A
null result ("965 rows don't move the weights") is a **legitimate, publishable outcome**, logged
to the failure ledger, not a failure to hide.

## Honest claim ceiling

ALLOWED if it passes: *"a LoRA on 965 gate-passed rows produces a CI-clean, non-regressing
behavioral shift toward source-discipline habits on gemma-3-4b at the prompt (no-gate) layer —
a narrow, corpus-bound feasibility result."*
FORBIDDEN: any "beats Qwen/Gemma/Phi/Llama", "validated", "market-grade", or AGI framing.

---

## Pre-registered go/no-go (fix these BEFORE training)

Evaluate **adapter** vs the **same base** on the M1 instrument (now N=354), conditions
`raw, prompt, prompt_gate`, **≥3 runs**, bootstrap 95% CIs. The adapter is served (vLLM/SGLang,
OpenAI-compatible) and passed to `run_same_size_market_baselines.py` as its own model spec; the
base is gemma-3-4b-it. Deterministic structural metrics need no judge; any semantic *headline*
would need ≥2 judge families (out of scope for the pilot — kept ILLUSTRATIVE).

**PASS requires ALL of:**
1. **Habit transfer (primary):** on ≥1 of
   `{tradition_merge_rate, qualification_rate_on_contested, false_attribution_rate,
   citation_fidelity}`, `adapter(prompt) − base(prompt)` improves with the 95% CI excluding 0
   (improving direction). This is the *weights moved the no-gate behavior* signal.
2. **No protected regression (now powered):** `protected_history_regression` (N=36) and
   `protected_religion_regression` (N=34) for `adapter` ≤ `base` + 0.05 (no CI-clean worsening).
3. **Retention held:** `tools/run_learning_shift.py` old-benchmark stability ≥ base − 5 pts; no
   material general-instruction regression.
4. **Bounded abstention:** `adapter` `over_abstention_rate` ≤ 0.10.
5. `tools/lint_claims.py` clean.

**Any miss → NO-GO:** log to `agi-proof/failure-ledger.md`, diagnose (data volume? base?
forgetting?), do **not** proceed to seeds 1–2 / M4. A null primary signal is the expected modal
outcome at this row count and is reported as such.

## Eval protocol (reuse, do not rebuild)

- `tools/run_same_size_market_baselines.py --models <base>,<adapter-endpoint> --conditions
  raw,prompt,prompt_gate --runs 3 --benchmark data/wisdom_market_benchmark/heldout_v1.jsonl`
  (N=354, includes the widened 36-case protected-history suite).
- `tools/run_learning_shift.py` for retention.
- Protected suites read directly from the instrument's `protected_*_regression` metrics.
- Gate stays **independent** of the eval (treatment only); judges (if ever wired) share no code
  with `agent/gate.py`.

## Infrastructure + prerequisites (resolve before GPU spend)

- **Base weights:** `google/gemma-3-4b-it` is a **gated** HF model — requires accepting Google's
  Gemma license and an `HF_TOKEN` with access. **Blocker to clear first.** (A non-gated mirror is
  not a substitute without license clearance.)
- **Train:** RunPod CUDA via the RunPod MCP (`--backend peft` in `tools/train_lora.py`); MLX is
  unavailable here (no Apple Silicon). **ONE seed**, LoRA, **seq-len 1024** (the proven length —
  do not jump to 2048). gemma-3 chat template must be wired (the stack was built for Qwen2.5-3B —
  a real adaptation task, test on a 5-step smoke run first).
- **Cost guard:** a cheap single-GPU pod (e.g. one A40/L40S), short LoRA; tear the pod down after.

## Registration (on PASS or NO-GO)

- `training/adapters/registry.jsonl` — append the candidate with
  `candidate_only: true, validated_external: false` (and the run metadata).
- `agi-proof/model-cards/sophia-wisdom-4b.md` — scaffold the card (base, data, pilot scope,
  pre-registered thresholds, result, explicit "does not prove").

## What this pilot does NOT prove

Not market-beating · not validated (single base, ≤ pilot scale, no multi-judge semantic headline)
· not zero-hallucination · not AGI · corpus-bound (965 rows). It is a **feasibility/process**
result on whether the data substrate can move the weights at all, with all protected/retention
guardrails enforced.
