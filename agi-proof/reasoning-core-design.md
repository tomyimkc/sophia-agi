# Reasoning-Core Design — a knowledge-light model that reasons over the fact-checked wiki

**Status: design proposal (not a measured result).** `candidate_only; canClaimAGI:false;
narrow corpus-bound feasibility`. Nothing in this document is a capability claim — it is a
recipe and a pre-registration template. Any number it later produces is subject to the
same no-overclaim gate as everything else in this repo (`tools/claim_gate.py`,
`agi-proof/measurement-thesis.md`). This file describes work that is **not yet run**; the
corresponding failure-ledger entries stay OPEN until a gated run exists.

## Thesis (this is the repo's own bet, scaled up)

`sophia-wisdom-4b-method-note.md` already states it: **truth outside the weights.** Facts
are enforced by the gate + tools + the fact-checked wiki; the weights learn only
*judgment and discipline*. The Wisdom-4B work proved that shape as a LoRA *habit-adapter*.
This design carries the same shape into the **base design of an 8–14B reasoning core**:
a model whose job is to *think, retrieve, and defer to retrieved evidence* — not to
memorize the world.

```
   ground truth: fact-checked wiki (wiki/) + OKF belief graph (okf/) + tools
        │
   external GATE (agent/verifiers, grounded_gate) ── enforces truth ──►  reasoning core 8–14B
        │                                                                 learns ONLY:
   provenance carried forward (okf/graph.py min-over-chain)               retrieve-then-reason,
                                                                          when-to-retrieve,
                                                                          defer-to-source, abstain
```

## Why this is cheap (parameters and compute)

Most of a frontier model's parameters memorize world knowledge (~2 bits/param of factual
storage capacity in the published scaling work). Moving facts into the wiki removes the
need to *store* them; the model needs only language understanding, query formation,
reasoning over retrieved text, tool use, and the discipline to abstain. Those are far
cheaper in parameters than encyclopedic recall — which is why a 4B already learned the
*habit* here.

- **Size:** a dense **8B** core is the recommended sweet spot (Qwen3-8B / Llama-3.1-8B
  class); go to **14B** only if multi-hop agentic reasoning needs the headroom. There is a
  hard floor (~7B) below which "know what to ask the wiki" and reliable multi-step planning
  degrade.
- **Do not pretrain from scratch.** Start from open weights — pretraining buys latent
  world knowledge that this design deliberately offloads. All work below is *post-training*.

## The post-training pipeline (built on what is already shipped)

| Stage | Goal | Builds on (existing) | New work |
|---|---|---|---|
| 0. Base | language + latent reasoning | open 8B weights | — |
| 1. SFT | the retrieve-then-reason *format* + habits | `tools/train_lora.py` (`--guard --scaffold --distill`), `tools/lint_training_rows.py` | data mix (below) |
| 2. RLVR | verifiably good reasoning + retrieval-faithfulness | `provenance_bench/rl_reward.py`, `tools/run_rlvr.py`, `training/swarm_router/train_grpo.py`, `agent/verified_trace_rlvr.py` | **`provenance_bench/retrieval_faithfulness.py`** (shipped with this doc) |
| 3. Calibrate | when-to-retrieve / when-to-hedge | `agent/graded_decision.py`, `data/settled_facts.json`, `agent/calibration.py`, `provenance_bench/calibration_score.py` | conformal abstention threshold |

### Stage 1 — SFT data mix

Hard constraint (`tools/lint_training_rows.py`): every knowledge-bearing discipline row
teaches a **HABIT** (route / qualify / refuse / cite), never a bare ground-truth fact —
otherwise it teaches the model to memorize answers, defeating the whole design. Proposed
mix (ratios *tuned*, not guessed: use `pretraining/data_mixing/run_mixing.py` to find the
interior optimum against held-out reasoning loss, per the data-mixing methodology):

| % | Family | Teaches | Source |
|---|---|---|---|
| 25 | retrieve-then-reason traces | query → `retrieve(q')` → reason over chunks → *cited* answer | teacher → gate → {accept \| correct-abstain} admission |
| 20 | verifiable reasoning (math/code) | raw, knowledge-free reasoning with checkable answers | `training/sophia-math-code-curriculum/` |
| 15 | source-discipline habit | qualify / route / refuse on contested claims | existing `source_discipline` family (lint-enforced) |
| 15 | when-to-retrieve decisions | settled → answer directly; unknown/contested → retrieve | `data/settled_facts.json` (stops over-retrieval / over-hedging) |
| 10 | abstention / fail-closed | unanswerable-from-wiki → abstain with a reason | `agent/graded_decision.py` cases |
| 10 | multi-hop + tool/MCP | fan-out sub-queries, fuse, reason across hops | `agent/ai_search.py` traces |
| 5 | contradiction surfacing | retrieved chunks conflict → surface it, don't silently pick | OKF contradiction ledger |

All pairs decontaminated (`tools/assert_decontam.py`, content-shingle, not just exact
prompt) with a private split held out of every tuning/selection loop (IEC pillar 6).

### Stage 2 — the retrieval-faithfulness RLVR reward

`rl_reward.py` rewards "don't assert a forbidden attribution." A knowledge-from-wiki core
needs more: it must *use* retrieval rather than leak from the weights. The new reward
(`provenance_bench/retrieval_faithfulness.py`) is a deterministic, bounded `[-1, 1]`,
verifier-seam-driven score over a rollout trajectory, composed of five constructs
(weights renormalized over the terms present in a given rollout):

- **correct (0.30)** — task success via `agent/execution_verifiers.py` / gold match (the RLVR anchor).
- **grounding (0.25)** — every *knowledge* claim entailed by a chunk actually in context
  (`agent/source_verifier.py` / `fact_check_text`). A true-but-ungrounded assertion is
  still penalized — the goal is "knowledge from the wiki," not "happened to be right."
- **faithful (0.25) — the centerpiece, a counterfactual citation-drop test.** The rollout
  harness regenerates the answer with a claim's supporting chunk removed; a claim that
  still appears *leaked from the weights* (unfaithful, −1), a claim that flips to
  uncertain/absent *genuinely depended on retrieval* (faithful, +1). This is the term
  `rl_reward` did not have, and it is what distinguishes a reasoning core over a wiki from
  a closet memorizer. It also defeats citation-stuffing: a cited-but-unused chunk earns no
  credit because dropping it changes nothing.
- **decision (0.10)** — retrieved-when-unknown, answered-when-settled (mismatch on a needed
  retrieval is the worst case; over-retrieval is merely wasteful).
- **provenance (0.10)** — asserted confidence ≤ min-over-chain confidence
  (`okf/schema.confidence_rank`, `okf/graph.py` `confidenceLaundered`). Asserting
  consensus-level certainty on a legendary-confidence source is laundering, penalized.

**Hard floors** (mirroring `rl_reward`'s forbidden-assertion `REWARD_MIN`): a claim the
wiki *refutes* (contradicted), or a citation of a chunk that was never retrieved
(fabricated), returns −1. **Anti-hack guards** carried over from `rl_reward`: the
hedge-marker cap (no wrap-everything-in-caveats), a per-retrieval cost, and an
over-refusal penalty (abstaining on an answerable case scores below a correct fail-closed
abstention on an unanswerable one). GRPO wiring reuses `training/swarm_router/train_grpo.py`;
every reward evaluation is stamped via `agent/verified_trace_rlvr.py` for audit.

The counterfactual regeneration needs a model and so runs in the rollout harness; the
*reward computation* consumes the already-collected, machine-checkable trajectory and
stays deterministic and offline (CI-tested in `tests/test_retrieval_faithfulness.py`).

### Stage 3 — calibration

Re-use the Wisdom-4B calibration-fix lesson directly: the model must *not* over-hedge
settled facts. Fit a conformal abstention threshold (`provenance_bench/calibration_score.py`,
`agent/calibration.py`) so the decision term and the hedge cap are tuned to a target
selective-risk, with settled and contested sub-domains disaggregated (the v2→v3 lesson:
an aggregate metric hid a sub-domain regression).

## Pre-registration template (commit BEFORE training)

Per `measurement-thesis.md`, a reward you cannot measure cleanly yields a confident wrong
verdict. One `measurement_spec.json` per experiment, committed before the data:

```jsonc
{
  "experimentId": "reasoning-core-faithfulness",
  "constructs": ["counterfactual-grounding", "llm-judge-faithfulness-2family", "heldout-multihop-transfer"],
  "primaryMetric": "counterfactual_grounding_rate",   // SUPPORTED ∧ flips-when-dropped, / total knowledge claims
  "direction": "core >= base + practicalThreshold",
  "mde": 0.05,
  "requiredN": null,                  // FILL from a power calc at mde, ~80% power (PILLAR 2)
  "uncertainty": "paired-bootstrap-or-confidence-sequence",   // PILLARS 1,4 — RLVR peeks, so anytime-valid
  "stoppingRule": "anytime-valid",
  "decontam": "content-shingle + private-split",              // PILLAR 6
  "judges": ["<family-A != subject>", "<family-B != subject>"],   // PILLAR 7 (!= subject, != gate)
  "practicalThreshold": 0.05,         // PILLAR 8
  "claimCeiling": "candidate_only; canClaimAGI:false"
}
```

`counterfactual_grounding_rate` is a **new construct** (a retrieval-faithfulness verifier)
joining markers + judge panel + behavioral transfer. Promotion only on GO from
`tools/claim_gate.py`, with ≥2 independent judge families for the semantic-faithfulness
construct, and re-test on novel entities for external validity.

## Training cost (on B300, FP8, ~40% MFU)

This is the payoff of post-training over from-scratch:

- **SFT** (8B, ~50–200B high-quality tokens): hours to ~2 days on 8–32 B300.
- **RLVR/GRPO**: dominated by rollout *generation* (verifiers are cheap, offline, CPU);
  days on 8–64 B300. The counterfactual regeneration roughly doubles rollout inference per
  tested claim — budget for it.
- **End-to-end (SFT + RLVR + calibrate) for an 8B core: roughly one 8×B300 node for
  ~1–3 weeks** — versus cluster-months for from-scratch pretraining.

## What this is NOT (the ceiling)

- **Not a measured result.** A design + a shipped reward + tests. No uplift is claimed
  until a pre-registered, powered, multi-family run produces a GO receipt.
- **Not a general LLM and not a hallucination guarantee.** The reward reduces ungrounded
  assertion on the trained surface; it does not eliminate it, and it is corpus-bound.
- **Not validated against first-party frontier** (egress-blocked here), and **not AGI**
  (`canClaimAGI:false`).

## Failure-ledger hooks (to add OPEN entries for)

1. Live rollout-driven GRPO uplift from the faithfulness reward — **unrun (no GPU spent
   yet)**. The full loop is implemented and offline-validated: reward → `group_advantages`
   → `grpo_objective` (k3-KL-regularized policy gradient) → `TorchPolicy` token-level
   update, with the counterfactual regeneration in the sampling path
   (`faithfulness_grpo.train` / `run_live`, wired to `tools/run_rlvr.py --task
   faithfulness`). The objective math is unit-tested (`grpo_objective`: the gradient
   raises faithful answers' log-prob, lowers leaky ones', on an all-correct group where a
   correctness-only reward collapses). The **RunPod launch path** is wired (dry-run
   validated, no pod): `rlvr-runpod.yml` + `runpod_rlvr.py --task faithfulness`, with the
   entailment key forwarded into the pod env, and an **on-pod base-vs-adapter held-out
   eval** on the trained adapter via the local-HF policy seam
   (`faithfulness_eval.make_hf_compare_policies` → `eval_faithfulness.py --compare`).
   **Open:** the actual CUDA dispatch + a powered, multi-family `claim_gate` GO receipt
   (the on-pod compare is `candidate`, not a GO). The torch plumbing is structure-
   validated, not CI-executed.
2. Live seams — retrieve (`faithfulness_seams.make_ai_search_retrieve` on `agent.ai_search`)
   is conformance-checked against the real committed RAG index. Verify
   (`make_entailment_verify`) now runs a **real entailment LLM**
   (`make_llm_entailment` / `entailment_from_provider` over `agent.deepseek_llm` /
   `agent.llmhub_llm`; keys in gitignored `private/secrets/`), live-verified end-to-end,
   with the deterministic `lexical_entailment` as the offline/no-key fallback and a
   fail-closed-to-`irrelevant` guard on network errors. Selectable via
   `run_rlvr --task faithfulness --entailment-provider {deepseek,llmhub}`.
3. `counterfactual_grounding_rate` — the eval instrument is shipped and offline-validated
   (`provenance_bench/faithfulness_eval.py`: `evaluate` / `compare`, fixed-n CI +
   anytime-valid CS via `tools/eval_stats.py`), and the experiment is **pre-registered**
   (`agi-proof/benchmark-results/faithfulness/measurement_spec.json`: mde 0.10, requiredN
   377 @ p0 0.6). **Open:** the actual base-vs-adapter run (needs a trained adapter) and a
   reserved private held-out split.
4. ≥2-family judge validation of the faithfulness construct — DeepSeek + an LLMHub family
   are wired and live; the panel run + inter-judge agreement (κ/AC1) is not yet executed.
