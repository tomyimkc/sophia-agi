# AGI-Shaped Capability Interfaces — Stage 1 (toy reference implementations)

**Status:** fail-closed **interfaces with toy reference implementations**, deterministic/offline.
**Boundary:** This is **not** proof that Sophia is AGI, and these are **not** the capabilities their
names describe. Each module is a *plumbing seam + a deterministic toy* so the real capability can be
tested later without weakening the no-overclaim gate. Artifacts are marked `candidateOnly: true`,
`level3Evidence: false`, and `depth: "toy-reference"`.

"Implemented" below means **the interface and a toy reference exist** — not that program induction,
planning, world-modelling, etc. are solved.

> **What each module literally is** (owning the toy-ness so it cannot be "exposed"):
> - `program_induction.py` — template-matching over ~3 hand-coded fitters (affine/quadratic/string/grid), **not** generative synthesis.
> - `active_inference.py` — a priority-sorted **TODO list** from report metadata, **not** an adaptive belief-update loop.
> - `planner_mcts.py` — MCTS against a **scripted simulator** with hardcoded outcomes, **not** planning under real uncertainty.
> - `predictive_world_model.py` — a `{(state,action): Counter}` **lookup table** over a handful of traces, **not** a learned world model.
> - `continual_plasticity.py` — a promotion **scorecard that updates no weights**, **not** online learning.
> - `layered_memory.py` — a permission-gated **dict** with token-overlap retrieval, **not** hierarchical cognitive memory.

## What was added (interface + toy reference)

| Pillar | Module | What it implements | Fail-closed rule |
|---|---|---|---|
| Fluid abstraction + program induction | `agent/program_induction.py` | Learns small executable transformations from examples across numeric/string/grid tasks. Optional proposed programs pass an AST sandbox. | Promote only if validation and test splits clear floors; otherwise abstain. |
| Autonomous active verification | `agent/active_inference.py` | Converts held claims, low-confidence accepts, stale/provisional candidates, and insufficient-source reasons into prioritized verification plans. | Emits agenda/quarantine candidates only; no canonical write. |
| Deliberate System-2 planning | `agent/planner_mcts.py` | MCTS-style search over verification tool plans with source-count, contradiction, cost, and risk. | Unsupported judge/vote cannot publish; final execution still goes through `fact_check_gate`. |
| Predictive world model | `agent/predictive_world_model.py` | Learns discrete `P(next_state, reward | state, action)` from traces, reports uncertainty/OOD, and chooses actions only when supported. | OOD or high-uncertainty state/action pairs hold. |
| Safe parametric plasticity | `agent/continual_plasticity.py` | Promotion gate for LoRA/RLVR/skill updates using target improvement, protected-suite regression, contamination, and verifier artifacts. | Contaminated/regressing updates reject; weak evidence quarantines. |
| Layered memory | `agent/layered_memory.py` | Working/episodic/semantic/procedural memory with trust-separated writes and retrieval. | Semantic/procedural memory requires `accepted` verdict plus evidence. |

## One-command artifact

```bash
python tools/run_agi_missing_pillars.py
```

Default output:

```text
agi-proof/agi-kernel/missing-pillars.public-report.json
```

The output contains component reports and invariants. It is a candidate proof artifact, not Level-3 evidence.

## Test commands

```bash
python3 tests/test_program_induction.py
python3 tests/test_active_inference.py
python3 tests/test_planner_mcts.py
python3 tests/test_predictive_world_model.py
python3 tests/test_continual_plasticity.py
python3 tests/test_layered_memory.py
python3 tests/test_agi_missing_pillars_bundle.py
python3 tools/run_agi_missing_pillars.py
```

## Acceptance criteria for this stage

1. Program induction promotes learnable toy transformations and abstains on an unlearnable/OOD mapping.
2. Active inference produces concrete actions for every detected gap and extra-source requirements for high-risk claims.
3. MCTS planning gathers sufficient independent source actions for high-risk claims and rejects when contradiction search finds a contradiction.
4. The world model chooses supported high-reward actions and holds on OOD actions.
5. The plasticity gate promotes only clean improvements and rejects contamination/protected regression.
6. Layered memory blocks unverified semantic/procedural writes.
7. The bundle report preserves no-overclaim fields: `candidateOnly: true`, `level3Evidence: false`.

## What this still does not solve

- It does not train a neural model or update weights locally.
- It does not prove broad external benchmark superiority.
- It does not replace third-party hidden evaluation.
- It does not remove the need for live source adapters, long-horizon runtime logs, and independent replication.

## Next development path

1. Connect `active_inference` to the live fact-check backends and quarantine learning loop.
2. Feed `planner_mcts` actions into real MCP/tool execution traces.
3. Replace toy program-induction tasks with ARC-like hidden packs and table/code transformation packs.
4. Use `predictive_world_model` over real agent traces to reduce repeated failed tool calls.
5. Require `continual_plasticity.evaluate_update` before any LoRA/RLVR adapter becomes default.
6. Back `layered_memory` with OKF/PROV storage and source-freshness TTL.

## 中文摘要

本階段把 Sophia 缺少的 AGI 形狀能力做成可測、可拒絕、可審計的基礎設施：程式歸納、主動驗證、MCTS 工具規劃、預測世界模型、安全持續學習閘門、分層記憶。這些都是候選機制，不是 AGI 證明；所有輸出都必須保留 `candidateOnly: true` 與 `level3Evidence: false`，並且只有經過 verifier gate 的內容才能進入可信記憶或公開結果。
