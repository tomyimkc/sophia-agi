# Sophia Long-Context Architecture Contract

**Status:** scaffold contract, not a trained model  
**Scope:** future large-context, small-parameter Sophia candidate systems  
**Claim boundary:** This document defines artifacts and verifier gates for future
measurement. It does not declare a capability result, a promoted checkpoint, or AGI.

Sophia's near-term architecture bet is that a modest local model can behave better
when long context is structured, routed, verified, and measured. The model is not
trusted as the proof system. Machine-checkable artifacts, deterministic verifier
outcomes, ablations, and failure ledgers decide what can be promoted.

## Architecture Contract

The runtime candidate has four separable duties:

1. Pack context into explicit channels.
2. Route only the needed memory, tools, and councils into the prompt.
3. Produce an answer with source boundaries intact.
4. Pass deterministic verifier gates before a result is counted as evidence.

Every component below must be independently suppressible in ablation runs. A
candidate can be measured only when the same task pack can run as `raw`,
`long_context_only`, `long_context_plus_memory`, `long_context_plus_tools`,
`verifier_gated`, and `sophia_full`.

## Context Channels

| Channel | Purpose | Allowed inputs | Required metadata |
|---|---|---|---|
| `task` | User request, benchmark task, or sealed evaluator prompt | Current task only | task id, visibility policy |
| `system_contract` | Non-negotiable behavior and claim boundaries | Sophia system prompts and verifier policy | prompt hash, version |
| `source_evidence` | Cited evidence used for factual claims | OKF records, local corpus, approved web evidence, benchmark-provided sources | source id, license, retrieval score, checksum when available |
| `working_memory` | Short-lived scratch context | Intermediate notes and tool summaries from the same run | run id, expiry, writer |
| `episodic_memory` | Append-only previous observations | Reviewed memory rows only | memory id, timestamp, reviewer/verifier state |
| `semantic_memory` | Reusable distilled facts or patterns | Promoted summaries with provenance links | source ids, distillation method, verifier state |
| `tool_trace` | Tool calls and tool outputs | Scrubbed MCP/tool traces | tool name, arguments hash, result hash, verdict |
| `council_notes` | Role-specific deliberation | Domain council outputs | council role, dissent notes, chair verdict |
| `failure_context` | Known blockers and negative cases | Failure ledger excerpts, rejected examples, held cases | ledger ref, reason, next action |

No channel may silently override another channel. When channels conflict,
`source_evidence` and verifier state dominate model memory. If evidence is missing
or contested, the model should hold, abstain, or ask for retrieval rather than fill
the gap.

## Memory Layers

| Layer | Retention | Write rule | Read rule |
|---|---|---|---|
| `scratch` | one run | free-form, not persisted | always allowed inside run |
| `episode` | append-only | write only after source or reviewer acceptance | routed when task similarity and domain match |
| `semantic` | durable | distilled from accepted evidence cards only | routed only with source refs and confidence |
| `procedural` | durable | tool/council habits from reviewed traces | routed as policy hints, never as factual evidence |
| `negative` | durable | rejected, held, or abstained cases | routed to prevent repeated failures |

Writes must carry a verifier state: `accepted`, `rejected`, `held`, or `abstain`.
Only `accepted` evidence can support factual promotion. `held` and `abstain` rows
are useful training signals for uncertainty, not proof of model capability.

## Router Decisions

The router is a deterministic policy surface first and a learned policy later. A
router decision record should include:

- `decisionId`: stable id for the task/run.
- `selectedChannels`: context channels included in the prompt.
- `omittedChannels`: channels deliberately excluded, with reason.
- `toolPolicy`: `allow`, `retrieve`, `clarify`, `abstain`, or `block`.
- `councilPolicy`: which council seats are active and why.
- `memoryPolicy`: memory layer reads, writes, and expiry.
- `riskFlags`: source conflict, hidden benchmark, citation required, tool risk,
  privacy risk, or overclaim risk.
- `verdict`: `accepted`, `rejected`, `held`, or `abstain`.

The router must be measurable without training. The MVP is a dry-run manifest and
validator; future learned routers must beat the deterministic policy without
increasing false positives, over-calling, or unsupported claims.

## Verifier Gates

Deterministic gates run before semantic judges:

1. **Source discipline gate:** forbid known false attribution and lineage merges.
2. **Evidence-card gate:** every factual claim in a promoted run points to an
   accepted evidence card or is marked illustrative.
3. **Context-packing gate:** packed context preserves source ids, channel labels,
   token budgets, and verifier states.
4. **Router gate:** risky tasks without evidence are held or abstained, not
   answered as settled.
5. **Promotion gate:** aggregate results meet thresholds, preserve failures, and
   keep `canClaimAGI=false`.

When deterministic checks are insufficient, semantic review may be added only as a
separate judge family and must be labeled as such. No model-judged result alone can
promote a public claim.

## Council Roles

| Role | Responsibility | Failure mode it watches |
|---|---|---|
| `source_chair` | Enforce citations, lineage boundaries, and abstention | unsupported attribution |
| `memory_curator` | Decide whether memory is relevant and safe to read/write | memory contamination |
| `tool_steward` | Choose tools conservatively and inspect tool errors | over-calling or wrong tool |
| `domain_specialist` | Review domain-specific evidence boundaries | false expertise |
| `compression_auditor` | Check whether compressed context preserves key facts | lost evidence or polarity |
| `risk_chair` | Stop overclaims and require failure-ledger updates | inflated capability claim |

The chair synthesizes, but dissent remains part of the artifact when it affects the
verdict. A council output is advisory until a verifier accepts it.

## Architecture Bets

The initial bets are recorded in
`agi-proof/architecture-bets/manifest.json` and remain scaffold-only:

- verifier-gated long context;
- hybrid memory;
- selective tool-use router;
- council orchestration;
- verifier-as-reward;
- long-context compression and recall;
- architecture-aware eval harness.

Each bet must name its hypothesis, ablations, metrics, failure ledger refs, and
promotion criteria before a result can be presented.

## Promotion Criteria

A long-context candidate can be promoted only as an internal candidate when all of
the following are true:

- candidate config, context-packing manifest, architecture bet manifest, and MLOps
  run template validate offline;
- every promoted factual claim has accepted evidence or is explicitly marked as an
  abstention/held result;
- at least three seeds are available for headline metrics;
- false-positive cost, over-abstention, wrong-tool rate, and memory contamination
  do not regress beyond pre-registered thresholds;
- ablation results show which architecture component caused the uplift;
- failures are preserved in the failure ledger;
- `candidateOnly=true` and `canClaimAGI=false` remain true.

## Claim Boundaries

Allowed language:

- "long-context architecture scaffold";
- "candidate-only config";
- "offline validator";
- "illustrative sample";
- "not yet measured";
- "promoted internal candidate" only after the promotion gate clears.

Disallowed language for this scaffold:

- "proves AGI";
- "is AGI";
- "guarantees no hallucination";
- "validated long-context capability" without run artifacts.

## MVP Artifacts

This scaffold adds:

- architecture bet schema and manifest;
- context-packing evidence-card schema and illustrative manifest;
- candidate config sample;
- MLOps architecture run template;
- offline validators for context packing and architecture ablation dry runs.

These artifacts make future experiments easier to run and harder to overclaim.
They do not train a model or report a benchmark score.

## Summary / 摘要

**EN:** Sophia's long-context plan is a verifier-first architecture scaffold. It
defines context channels, memory layers, router records, council roles, ablations,
and promotion gates so future model runs can be measured without claiming more
than the artifacts prove.

**中文:** Sophia 的长上下文方案目前只是可验证的架构脚手架。它定义上下文通道、记忆层、
路由记录、委员会角色、消融实验和晋级门槛，用来支持未来测量；它不声称已经训练出模型能力，
也不声称 AGI。
