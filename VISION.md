# Sophia — Vision & Charter

**A provenance-aware, verifiable, fail-closed reasoning platform.**

**Wisdom before intelligence.**

Sophia's mission is *not* to claim AGI or machine consciousness. 

It exists to build the most trustworthy, verifiable, provenance-aware reasoning system possible — one that traces every claim to its sources, knows when it must abstain, and can prove *why* it believes what it outputs.

> **Guiding principle.** A reasoning system earns trust by being *checkable*, not
> by being *confident*. Every advance should move Sophia from "probabilistic
> guessing" toward "verifiable, traceable inference."

This is the canonical statement of intent. The README describes what exists today;
this file says what Sophia is *for*, and holds new work accountable to it. (Scope
disclaimers in [README.md](README.md), [SECURITY.md](SECURITY.md), and
[RESULTS.md](RESULTS.md) remain authoritative — **this is not a claim of AGI**.)

## Core design commitments

1. **Provenance over assertion.** Every belief, claim, or conclusion carries a
   traceable lineage — its source, the reasoning that derived it, and a confidence
   estimate. Sophia maintains a belief graph where nodes are claims and edges are
   justifications, and it supports counterfactual queries
   (*"what would I conclude if this source were removed?"*).
2. **Verification over generation.** Raw model output is a *hypothesis*, not an
   answer. Hypotheses are checked — code execution, logic/constraint solvers,
   knowledge-graph consistency, self-consistency — before being promoted to
   confident claims.
3. **Fail-closed, not fail-open.** When confidence is low, sources conflict, or
   verification fails, Sophia says "I don't know" or escalates. It never fabricates
   to fill a gap. Confidentiality and integrity boundaries (Bell-LaPadula-style
   classification, airgap profiles) are enforced by default.
4. **Functional self-modeling, not consciousness.** Sophia represents and reasons
   about its own knowledge, uncertainty, and limits (calibration, metacognitive
   monitoring, knowing when to defer). It makes **no** claims to subjective
   experience or sentience — this is the scientifically grounded, ethically safe
   sense of "self-aware."
5. **Local-first and sovereign.** Sophia runs on the user's own infrastructure with
   multi-backend LLM support, preserving privacy, control, and airgapped operation.

## Architectural pillars — and where they live today

| Pillar | Status | Implementation |
|--------|--------|----------------|
| **Reasoning core** — multi-backend LLMs, deliberation (CoT, best-of-N, self-verification, process-reward), councils | ✅ | `agent/model.py` (10+ presets), `agent/best_of.py`, `agent/council_deliberate.py`, `agent/sector_council.py`, `provenance_bench/rl_reward.py` |
| **Belief graph & provenance engine** — queryable claims/justifications graph; confidence propagation; retraction; **counterfactual analysis**; audit trails | ✅ | `okf/graph.py` (min-over-chain propagation, contradiction ledger), `okf/counterfactual.py` (counterfactual removal + retraction), `agent/wiki_store.py` |
| **Memory & retrieval** — long-term (vector + structured KG), rolling consolidation, provenance-tagged | ⚠️ partial | `agent/memory_consolidation.py` (85 LOC), `agent/vector_store.py`, `agent/rag_pipeline.py`, `okf/` wiki tiers. **Honest status:** runtime decision memory is a 42-line append-only JSONL log (`agent/memory.py`); retrieval now has a **deterministic offline vector tier** (`agent/lexical_embed.py` — numpy-free hashed n-gram cosine, the live default between learned embeddings and keyword overlap), but **learned** semantic embeddings still require a model/API and no `rag/index/embeddings.npz` is committed, so semantic vector recall is not yet in the live path. |
| **Verification layer** — neurosymbolic checks, executable verification, citation/source validation gating claim promotion | ✅ | `agent/verifiers.py`, `agent/verifier_synthesis.py`, `agent/gate.py`, `agent/grounded_gate.py` |
| **Self-model & calibration** — uncertainty estimation, "I don't know", routing below threshold | ⚠️ partial | `agent/calibration.py` (ECE, risk-coverage), `agent/graded_decision.py`, `agent/corroboration.py`. **Honest status:** metrics run on a synthetic suite; the graded answer/hedge/abstain **router is not yet wired into any runtime** (`graded_decision.decide()` has no live caller). |
| **Security & confidentiality** — Bell-LaPadula classification, confidentiality verifiers, airgap, sandboxed MCP | ✅ | `agent/security/labels.py`, `agent/dataflow/firewall.py`, `agent/policies.py`, `sophia_mcp/audit.py` |
| **Agentic orchestration** — stateful, checkpointed agent graphs, human-in-the-loop for high-stakes actions | ⚠️ partial | `agent/harness.py`, `agent/guarded.py`, `sophia_mcp/gateway_wiring.py`, gate + audit on writes. **Honest status:** the fail-closed gateway is now **wired into the live MCP server** for the 4 side-effecting/external tools behind `SOPHIA_MCP_GATEWAY=1` (pre-dispatch authz → firewall → kill-switch → BLP → taint-label; red-team tested in `tests/test_server_gateway_live.py`). Still deferred: output re-verification on the served surface, MCP federation, and a concurrent queue (the queue is single-process JSONL). |

## Development philosophy

- **Assemble and orchestrate; innovate at the trust layer.** Don't try to out-train
  frontier labs. Sophia's contribution is provenance, verification, calibration, and
  fail-closed reasoning — the layers labs under-invest in.
- **Every feature must make reasoning more checkable.** Reject additions that
  increase fluency at the cost of traceability.
- **Honesty as a feature.** Distinguish what Sophia *knows*, what it *infers*, and
  what it's *guessing* — visibly, in every output. Every public number must clear
  the no-overclaim measurement gate (multi-judge consensus + confidence intervals;
  see [RESULTS.md](RESULTS.md)).

## Explicit non-goals

- No claims of AGI, sentience, consciousness, or subjective experience.
- No fabrication to appear more capable; no fluency-over-truth tradeoffs.
- No autonomous high-stakes action without oversight; risk-proportional
  human-in-the-loop.

## Success criteria

Sophia succeeds when a skeptical user can:

1. **Trace** any claim to its sources and reasoning (`okf.belief`, `okf.counterfactual`).
2. **Trust** that "I don't know" means the system genuinely lacked grounded support
   (the fail-closed gate and calibrated abstention).
3. **Run** it fully locally / airgapped (`SOPHIA_PROFILE=airgap`, local backends).
4. **Verify** that confidence scores are well-calibrated against real accuracy
   (`agent/calibration.py` — ECE, risk-coverage).

The benchmark is **trustworthiness and verifiability**, not breadth of knowledge or
human-likeness.
