# Runtime Trust-Layer — next development lanes

**Context.** The training / GPU / external-eval track (PRs #97, #98; RunPod SFT;
math-code curriculum) is owned by a parallel effort. These three lanes are the
**runtime trust-layer** counterpart: pure-Python, offline, CPU/CI-friendly, no GPU,
no model training. Each one closes a gap that **VISION.md already flags as ⚠️ partial**,
so the work is honesty-repair, not new scope. Nothing here is an AGI claim.

Priority: **L1 → L2 → L3** (smallest/highest-honesty first).

---

## L1 — Wire the graded answer/hedge/abstain router into the live runtime *(do first)*

**The gap (verified).** `agent/graded_decision.decide()` maps `(gate_passed, confidence)`
onto `answer | hedge | abstain` and is explicitly written as a "drop-in replacement for
the static `on_fail` branch" — but **nothing calls it**. The only references in the tree
are a docstring in `agent/guarded.py`, the proof-package manifest, and a *different*
`policy.decide` in `agent/conformal_gate.py`. VISION.md's Self-model & calibration row
says exactly this: *"the graded answer/hedge/abstain router is not yet wired into any
runtime (`graded_decision.decide()` has no live caller)."* The calibration pillar is
measured on a synthetic suite but never **decides** anything at runtime.

**Where to wire it.** `agent/grounded_agent.py::grounded_answer()`. Today it returns a
`policy` outcome (`grounded_strict`, `grounded_fallback`, `fallback_gated_abstain`, …)
straight from `answer_with_policy()` with **no confidence signal applied**. The router is
the missing post-step.

**Build.**
1. After `answer_with_policy(...)`, compute a confidence in `[0,1]` for the produced
   answer. Reuse the two already-built-but-unwired modules the router's docstring names:
   - `agent.corroboration.corroborated_confidence` (Bayesian log-odds pool over the
     routed source + any neighborhood corroboration), and/or
   - `agent.calibration.self_consistency` (label-free agreement across a small number of
     sampled `complete()` generations — gate behind a `samples` arg so CI stays at 1).
2. Treat `gate_passed = policy in {"grounded_strict", "grounded_fallback"}` (a clean
   grounded/verified-fallback answer) vs the abstain policies.
3. Call `graded_decision.decide(gate_passed=…, confidence=…, thresholds=…)` and let it
   **downgrade**: a `grounded_strict` answer whose confidence `< lo` becomes `abstain`
   (suspicious low-confidence pass); a `fallback_gated_abstain` near-miss with confidence
   `>= hi` may `hedge` instead of hard-abstaining. Surface `{action, confidence, reason}`
   in the returned dict; never *upgrade* an abstain into a confident answer (fail-closed).
4. Add an opt-in flag (`graded=False` default) so existing callers/tests are unchanged
   until they opt in — mirrors the `--grounded` rollout pattern.

**Acceptance.**
- New `tests/test_grounded_agent_graded.py` (offline, mocked `complete`): a low-confidence
  strict pass is downgraded to abstain; a high-confidence gate-fail near-miss is hedged;
  a clean high-confidence answer is unchanged; thresholds are honored; fail-closed on
  `confidence=None`. Wired into `ci.yml`.
- VISION.md Self-model row updated from ⚠️ partial → ✅ with the live caller named.
- No public metric changes (this is plumbing); if any benchmark uses `grounded_answer`,
  re-run it with `graded=False` to confirm zero drift.

**Effort:** ~1 focused session. **Resource:** none (offline). **Risk:** low — additive,
flag-gated, fail-closed by construction.

---

## L2 — Make vector recall real (Memory pillar: scaffolding → live)

**The gap (verified).** `rag/index/` ships `chunks.jsonl` (484 KB) but **no
`embeddings.npz`**. Retrieval is keyword-only; VISION.md: *"Vector recall is scaffolding,
not in the live path."* Runtime decision memory is still the 44-line append-only JSONL
(`agent/memory.py`).

**Build.**
1. A `tools/build_rag_index.py` that embeds `chunks.jsonl` with a small CPU-feasible model
   (e.g. a MiniLM-class sentence-transformer; pin in `requirements-rag.txt`) and writes
   `rag/index/embeddings.npz` (vectors + row→chunk id map). Deterministic, reproducible,
   committed alongside a hash manifest so the index is auditable.
2. Put cosine-similarity recall in the live `agent/rag_pipeline.py` path **as a
   provenance-tagged augmentation of**, not a replacement for, keyword recall — every
   recalled chunk keeps its source id so the gate can still trace it.
3. Honest fallback: if `embeddings.npz`/the model is absent, fall back to keyword-only and
   say so (no silent capability claim).

**Acceptance.** Offline test that vector recall returns the expected chunk for a held-out
query the keyword path misses; provenance id preserved end-to-end; index build is
reproducible (same input → same hash). VISION.md Memory row updated honestly (note it's
retrieval-augmentation, not the full structured-KG consolidation).

**Effort:** ~1–2 sessions. **Resource:** none → CPU. **Risk:** medium (new dependency;
keep it optional and fail-closed).

---

## L3 — Output re-verification on the served MCP surface

**The gap (verified).** The fail-closed gateway is wired into the live MCP server for the
4 side-effecting/external tools (pre-dispatch authz → firewall → kill-switch → BLP →
taint). VISION.md Agentic-orchestration row lists what's **still deferred**: *"output
re-verification on the served surface."* Inputs are gated; what we hand back is not
re-checked.

**Build.** A post-dispatch hook in `sophia_mcp/gateway_wiring.py` that runs served tool
**outputs** back through the existing attribution/grounded gate (`agent/grounded_gate.py`)
before return — fail-closed: a response that asserts an unsourced/contradicted claim is
downgraded to an abstention or flagged, not emitted. Keep it behind the existing
`SOPHIA_MCP_GATEWAY=1` flag and the single-process queue.

**Acceptance.** Red-team test alongside `tests/test_server_gateway_live.py`: a tool whose
output smuggles a fabricated attribution is caught on the way out; a clean output passes
untouched; gate-off path is unchanged. VISION.md row updated to remove this from the
deferred list.

**Effort:** ~1–2 sessions. **Resource:** none (offline). **Risk:** medium — touches the
live MCP path; gate behind the existing flag and test hard.

---

### Why these three, and why now
All three are gaps **the project's own VISION.md admits in writing**. Closing them
converts three ⚠️ partial pillars (calibration, memory, orchestration) toward ✅ with
honest, machine-checkable evidence — and none of them collide with the training/GPU track.
L1 is the highest leverage per line of code: it turns a measured-but-inert calibration
module into runtime behavior, which is the single clearest "metric that doesn't yet act"
in the codebase.
