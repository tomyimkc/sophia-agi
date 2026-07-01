# Frontier Positioning Plan — the verifier-gated post-training layer

> Scope: a strategy + implementation plan, written to the repo's no-overclaim discipline.
> Every number below is already in `RESULTS.md` / the failure ledger with its CI and label.
> Nothing here is an AGI claim. Candidate work stays labelled candidate until a gate clears.

## 1. The honest position

Sophia is **not** a base-model project and should not be pitched as a GLM / Qwen / Llama
competitor on general benchmarks. It has never pretrained a transformer at scale
(`pretraining/` is closed-form-checked scaling-law research on a toy LM), and its tool-use
corpus is small (`training/tool_use/sft_traces.jsonl` = 80 rows, `dpo_pairs.jsonl` = 200).

What it measurably *is*: an open, **verifier-gated post-training and inference layer** that
makes any base model abstain instead of fabricate, with public confidence intervals and a
public failure ledger. The validated evidence is narrow and real:

- Attribution-hallucination Δ **12.5%** (36.1% → 23.6%), 95% CI [5.6%, 19.4%], 0% false-positive
  cost, 2 judge families (`RESULTS.md`).
- External calibration on public SimpleQA: selective-accuracy lift +15.8% [9.8%, 22.1%]
  (DeepSeek) and +7.8% [2.3%, 13.5%] (Qwen-2.5-72B), κ = 0.97 / 0.99 (`RESULTS.md`).

The differentiating asset is the **machine-verifier farm** (~30 verifiers in `agent/`:
attribution, temporal, legal-citation, math, physics, code, execution, deontic, …) plus the
loop that writes and trust-tests its own verifiers (`agent/verifier_synthesis.py`). No
frontier lab ships a public, measured abstention discipline like this. That is the axis to own.

## 2. What the verifier farm unlocks (the leverage)

The verifiers are deterministic, machine-checked labellers — no learnable judge. That single
property turns them into two compounding assets beyond the runtime gate:

1. **A preference-data engine.** Use the verifier verdict to label `(chosen, rejected)` pairs
   at scale, with verifier provenance instead of an LLM judge — see
   `tools/gen_verifier_dpo.py` (shipped; offline self-test green). This is how the 200-row DPO
   pack grows by orders of magnitude without manual labelling, and the label is unhackable.
2. **An inter-agent trust boundary.** Gate what a sub-agent may tell its siblings, so an
   unverified claim cannot contaminate a multi-agent swarm — see
   `agent/swarm_trust_boundary.py` (shipped; invariants green) and
   `docs/11-Platform/Verifier-Gated-Trust-Boundary.md`.

Both reuse the *same* verdict the runtime gate already produces: verifier-as-guard,
verifier-as-labeller, verifier-as-reward (`provenance_bench/swarm_rl.py`).

## 3. Pre-registered risk (do not assume it away)

The failure ledger already records that **prior trained adapters did not externally
transfer**: `v4-adapter-externally-unvalidated` (GSM8K null −3.0pp [−7.33, +1.00]),
`andreia-courage-gate` *reversed* on raw text (cowardice-error +0.30 [0.27, 0.32]). So:

- Preference pairs minted by the engine are a **training input**, not a result. Whether an
  adapter trained on them improves on a held-out third-party pack is an **OPEN** gate.
- The headline to chase is **one clean, judge-free, verifier-verified win on a third-party
  agentic benchmark** (τ-bench / SWE-bench-Verified subset), reported with the existing
  ≥2-judge-family + CI discipline — not another first-party number.

## 4. Twelve-week plan (mapped to real files + the cost/CI guardrails)

RunPod jobs go through the **GitHub Action** (`.github/workflows/rlvr-runpod.yml`), never local
SSH; the QAT-bypass regression test (`tests/test_qat.py`) must stay green or a cert re-measures
a no-op (root cause: commit `77a1076d`).

| Wk | Work | Touches | Pre-registered gate |
|---|---|---|---|
| 1–2 | **Preference Engine v0** (shipped): score candidates via the gate, mint verifier-labelled DPO pairs. | `tools/gen_verifier_dpo.py`, `tests/test_gen_verifier_dpo.py` | `lint_training_rows`, `assert_decontam` on output. |
| 3–4 | Scale `tool_use/dpo_pairs.jsonl` via the engine; ingest ToolACE / xLAM **as raw fuel re-scored by our verifiers** (do not trust foreign labels). | `training/tool_use/` | Decontamination logged; mixture provenance published. |
| 5–6 | **Multi-turn** RLVR reward half (shipped): `TrajectoryOutcome` + `trajectory_reward` with KL control + length-normalised cost, 7 invariants. Remaining: the GPU GRPO loop that consumes it (3-seed sweep). | `provenance_bench/swarm_rl.py`, `training/swarm_router/train_grpo.py` | invariants green; 3-seed sweep + `eval_stats` CIs via the RunPod GH Action remain OPEN. |
| 7–8 | **Trust boundary** (shipped core + measurement): route sibling reads through `GatedSharedState`; on-vs-off contamination measured. Remaining: the same on-vs-off delta on verified-success / cost over a third-party agentic task. | `agent/swarm_trust_boundary.py`, `tools/measure_trust_boundary.py` | contamination-blocked rate = 1.0 (detectable) measured; third-party verified-success delta OPEN. |
| 9–10 | Train + certify an **abstention-native tool-use adapter** on the verifier-labelled pack. | `training/qat`, `training/swarm_router/adapter` | κ ≥ 0.40 **or** CI excludes zero, ≥2 judge families; candidate until met. |
| 11–12 | **MoE = design only**: add z-loss + shared + fine-grained experts to `moe/router.py` as numpy reference; mark design-only in the ledger. Write up the trust-boundary result. | `moe/router.py`, `agi-proof/failure-ledger.md` | No trained-MoE claim. |

## 5. The pitch (to any lab, including a frontier collaborator)

> "I did not try to out-pretrain you. I built the open, measured, verifier-gated post-training
> layer that makes any base model — including yours — abstain instead of fabricate, with public
> confidence intervals and a public failure ledger. The verifier farm doubles as a preference
> labeller and an inter-agent trust boundary. Here is the held-out result, the seeds, and the κ."

The strongest interview artifacts are not capability badges; they are the QAT-bypass bug-catch
(`77a1076d`), the κ-disciplined M3-SFT panel, and the public failure ledger. Lead with those.

## 6. Status of this plan's artifacts

| Artifact | State |
|---|---|
| `tools/gen_verifier_dpo.py` + test | shipped, offline self-test green |
| `agent/swarm_trust_boundary.py` + test + design doc | shipped, invariants green |
| `provenance_bench/swarm_rl.py` multi-turn `trajectory_reward` (KL + length-norm) + tests | shipped, 7 invariants green |
| `tools/measure_trust_boundary.py` + test + artifact | shipped, contamination-blocked rate = 1.0 (detectable) |
| GPU GRPO sweep, adapter cert, third-party agentic delta, MoE design | OPEN — see the table above and the failure ledger |
