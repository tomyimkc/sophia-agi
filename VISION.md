# Sophia — Vision & Charter

**A provenance-aware, verifiable, fail-closed reasoning platform.**

Sophia's mission is *not* to claim artificial general intelligence or machine
consciousness. It is to be the most trustworthy, verifiable, and provenance-aware
local reasoning system we can build — one that reasons transparently, knows the
boundaries of its own knowledge, refuses to fabricate, and can prove *why* it
believes what it outputs.

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
| **Memory & retrieval** — long-term (vector + structured KG), rolling consolidation, provenance-tagged | ✅ | `agent/memory_consolidation.py`, `agent/vector_store.py`, `agent/rag_pipeline.py`, `okf/` wiki tiers |
| **Verification layer** — neurosymbolic checks, executable verification, citation/source validation gating claim promotion | ✅ | `agent/verifiers.py`, `agent/verifier_synthesis.py`, `agent/gate.py`, `agent/grounded_gate.py` |
| **Self-model & calibration** — uncertainty estimation, "I don't know", routing below threshold | ✅ | `agent/calibration.py` (ECE, risk-coverage), `agent/graded_decision.py`, `agent/corroboration.py` |
| **Security & confidentiality** — Bell-LaPadula classification, confidentiality verifiers, airgap, sandboxed MCP | ✅ | `agent/security/labels.py`, `agent/dataflow/firewall.py`, `agent/policies.py`, `sophia_mcp/audit.py` |
| **Agentic orchestration** — stateful, checkpointed agent graphs, human-in-the-loop for high-stakes actions | ✅ | `agent/harness.py`, `agent/guarded.py`, gate + audit on writes |

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
