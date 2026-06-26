# Sophia VISION

Sophia is a **verifier-gated, provenance-first cognitive system** for building and
measuring small, auditable, high-trust models and agents.

## Core principles

1. **Everything is a claim with sources.** No fluent paragraph leaves the system
   without an attached, machine-readable attribution trail.
2. **Verification is first-class.** Detectors, gates, and human-reviewable ledgers
   decide what may be emitted. The model is not trusted to self-police.
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
| **Memory & retrieval** — long-term (vector + structured KG), rolling consolidation, provenance-tagged | ⚠️ partial | `agent/memory_consolidation.py` (85 LOC), `agent/vector_store.py`, `agent/rag_local_embed.py`, `agent/rag_pipeline.py`, `okf/` wiki tiers. **Honest status:** vector recall is now **live in the retrieval path** — a committed, reproducible `rag/index/embeddings.npz` (built offline by `agent/rag_local_embed.py`, a deterministic CPU/airgap hashing embedder; `tools/build_rag_index.py --verify` checks it in CI) drives cosine search in `agent/retrieval.retrieve`, with the index self-describing its backend so queries embed in the same space. Honest bounds: the committed backend is a **lexical-semantic hash** embedding (generalizes surface form, not deep meaning — the Gemini backend remains the higher-quality option when a key is present), and runtime *decision* memory is still a 44-line append-only JSONL log (`agent/memory.py`); structured-KG consolidation is not yet in the live path. A deterministic **numpy-free lexical vector tier** (`agent/lexical_embed.py`) is also available as an offline middle tier between learned embeddings and keyword when desired via `SOPHIA_RETRIEVAL=lexical` or `vector`. |
| **Verification layer** — neurosymbolic checks, executable verification, citation/source validation gating claim promotion | ✅ | `agent/verifiers.py`, `agent/verifier_synthesis.py`, `agent/gate.py`, `agent/grounded_gate.py` |
| **Self-model & calibration** — uncertainty estimation, "I don't know", routing below threshold | ⚠️ partial | `agent/calibration.py` (ECE, risk-coverage), `agent/graded_decision.py`, `agent/corroboration.py`. **Honest status:** metrics run on a synthetic suite. The graded answer/hedge/abstain router is now **wired into the live grounded path** (`agent/grounded_agent.grounded_answer(graded=True)` → `apply_graded_decision` → `graded_decision.decide`): a gate-passing answer is **downgraded** to hedge/abstain when its confidence is low (downgrade-only, fail-closed, opt-in, no-op without a confidence signal; tested in `tests/test_grounded_agent_graded.py`). The confidence *source* is now **live**, not caller-supplied: `agent/grounded_confidence.py` pools the routed page's `authorConfidence` with neighbor corroboration into a provenance-grounded confidence (`grounded_answer(confidence_from_sources=True)`). Measured discrimination over the OKF wiki: weak sources (disputed/legendary/anachronism) downgraded **100%**, strong sources (consensus/attributed) kept **67%** (`tools/eval_graded_confidence.py`, candidate report). A **real stochastic-model calibration run** (deepseek over the 35 in-domain attribution traps, deterministic trap scorer — `tools/run_graded_calibration_live.py`) found the provenance confidence is a **weak, non-monotonic predictor** of answer correctness (balanced accuracy 0.52 at the hi=0.7 default, only 0.58 when fitted to hi=0.74/lo=0.35): it measures *source quality*, not answer correctness, so the hand-picked default `hi=0.7` is already near-optimal and the fitted thresholds (small-N candidate) are **not adopted**. Honest takeaway: this confidence is a sound provenance prior but should not be over-trusted as a correctness signal. |
| **Security & confidentiality** — Bell-LaPadula classification, confidentiality verifiers, airgap, sandboxed MCP | ✅ | `agent/security/labels.py`, `agent/dataflow/firewall.py`, `agent/policies.py`, `sophia_mcp/audit.py` |
| **Agentic orchestration** — stateful, checkpointed agent graphs, human-in-the-loop for high-stakes actions | ⚠️ partial | `agent/harness.py`, `agent/guarded.py`, `sophia_mcp/gateway_wiring.py`, gate + audit on writes. **Honest status:** the fail-closed gateway is now **wired into the live MCP server** for the 4 side-effecting/external tools behind `SOPHIA_MCP_GATEWAY=1` (pre-dispatch authz → firewall → kill-switch → BLP → taint-label; red-team tested in `tests/test_server_gateway_live.py`). **Served-output re-verification** is now wired too (opt-in `SOPHIA_MCP_OUTPUT_VERIFY=1`): a governed tool's returned text is re-checked through the epistemic gate and a fabricated-attribution payload is **withheld** fail-closed before it reaches the caller (`gateway_wiring.verify_output`, red-team tested in `tests/test_gateway_output_reverify.py`) — defense-in-depth on the inference/search reads, though it cannot roll back a completed write (writes self-gate pre-mutation). Still deferred: MCP federation, and a concurrent queue (the queue is single-process JSONL). |

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
- **Governed scaling.** When a scaling/efficiency primitive *is* useful (memory,
  routing, continual RL, kernel optimization), adopt it only with a trust governor
  bolted on — promote-only-what-verifies, bound off-trust drift, measure
  over-reliance, and admit optimizations only with an equivalence/error-bound proof.
  The path forward is *scale that carries its own proof*, never scale for its own
  sake. See [docs/11-Platform/Governed-Scaling.md](docs/11-Platform/Governed-Scaling.md).

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
