# Handover — Analyze Sophia's Workflow Flowcharts & Recommend AGI/ASI Improvements

**You are** a research engineer with full read access to the `tomyimkc/sophia-agi` repository. Your
job in this session: **read the workflow flowcharts, reconstruct how the system actually works, then
recommend the most effective and efficient path to raise these workflows toward AGI/ASI standards.**
Be concrete, code-grounded, and honest about feasibility. Name the strongest objection to each of your
own recommendations.

---

## 1 · What to read (in this order)

All under `docs/09-Agent/flowcharts/`:

1. `Sophia-Workflow.md` — **the combined walk-through**. One document, the whole system: master chart
   + all 8 subsystems + connective narrative + a "where the leverage is" section. Start here.
2. `00-Master-Flowchart.md` — the master chart alone, with the subsystem index table.
3. `01`–`08` — the eight subsystem charts, each with a **Thesis note** flagging the citable design
   point or the open gap for that subsystem:
   - `01-Intake-Routing` · `02-Grounded-Context` · `03-Council-Answer`
   - `04-Epistemic-Gate` (largest subsystem, 49 modules) · `05-Calibration-Abstention`
   - `06-Self-Evolution-RSI` · `07-Proof-Harnesses` · `08-Training-Path` (the only weight-changing path)

Rendered `.png` (screen) and `.svg` (print/LaTeX) of every chart are in `png/` and `svg/`.

**Provenance caveat:** these charts were built from a *working clone* (branch
`feat/oscillatory-crosspollination`, uncommitted local mods including `Sophia-Architecture.md`).
Before you treat any node label as authoritative, verify it against the code on the branch you're
actually reviewing — `git ls-files agent/ | wc -l`, then spot-check the modules each chart names.

## 2 · Verify the charts against the code (do this before recommending anything)

Do **not** trust the diagrams blindly. Confirm the load-bearing wiring the charts assert:

- `agent/gate.py` imports/calls `verifiers`, `claim_router`, `sector_council`, `benchmark_checks`.
- `agent/self_evolving_agent.py` drives `continual_plasticity` + `continual_retention`.
- `agent/gate_reward.py` → `agent/gate.py`; `agent/multiaxis_reward.py` → `gate_reward` + `prosoche` +
  `verifiers`.
- `agent/continual_plasticity.py:evaluate_update(...)` gate thresholds: `min_target_delta≥0.03`,
  `max_protected_regression≤0.01`, `require_artifacts≥2`, retention check.
- The per-case pipeline `tools/run_hidden_eval_sophia.py:run_case()` exposes each stage as a
  suppressible ablation flag (`use_intake`, `use_kb`, `use_evidence`, `use_council`, `use_gate`,
  `use_memory`, `use_tools`, `allow_repair`, `use_claim_router`).

Report any place the chart and the code disagree — that is itself a finding.

## 3 · The core diagnosis to pressure-test

The charts encode a thesis about where Sophia is strong and where it is weak:

> **Sophia is an extraordinarily complete epistemic *measurement* engine whose signals are almost
> never turned into a *learning* signal.** It measures gate verdicts, calibration/ECE, selective
> risk, provenance rank, fixed-point residuals — thoroughly — but `agent/calibration.py`,
> `agent/abstention_scoring.py`, and `agent/selective_risk.py` contain **no differentiable loss**.
> They score; they do not train. The self-improvement loop (chart 6) improves *assets*
> (memory / skills / verifiers), not weights, unless an explicit `training/` run is invoked (chart 8).

Your task is to decide whether this diagnosis is correct and, if so, **how to close the
measurement→learning gap most efficiently** without breaking the repo's honesty discipline.

## 4 · Known open seams (already scaffolded, not yet live)

These are documented seams in the current code — sanity-check that they still exist, then treat them
as the cheapest available levers:

- **Process-reward from gate verdicts** — `tools/distill_process_reward_model.py` (turns three-way
  gate verdicts into a PRM). Seam: the reward is not yet wired into `tools/run_rlvr.py`.
- **Calibration proper-scoring as a loss** — `tools/train_calibration_objective.py`. Seam: produces a
  target, not yet a DPO/MLX training step.
- **Provenance rank as a per-example loss weight** — `tools/provenance_weighted_training.py`
  (`agent/source_ranking.py` gives deterministic trust tiers).
- **Hidden-state featurizer** — `agent/activation_probes.py:build_hidden_state_featurizer` is a
  documented seam (MLX residual-stream hook). Several probe/energy directions
  (`tools/energy_verifier_head.py`, `tools/probe_representation_training.py`) are blocked on it.
- **Prior prospectuses** you can build on, not restate:
  `agi-proof/untapped-training-2026-07-01/` (W1–W5), `agi-proof/oscillatory-crosspollination-2026-07-01/`
  (O1–O5), and `out/Sophia-SkillOpt-CrossPollination.md` (S1–S5, skill-doc optimization).

## 5 · What to deliver

1. **Verification report** — chart-vs-code discrepancies (or confirmation), 5–10 bullets.
2. **Ranked recommendation list** — 3–7 improvements to move these workflows toward AGI/ASI standards,
   each with: the exact modules/files touched, why it raises capability *and* preserves epistemic
   honesty, an **effort estimate**, and the **single strongest objection** to it. Rank by leverage
   (capability gain per unit effort), not by novelty.
2b. For each recommendation, state whether it changes **weights** (chart 8 path — then the
   post-training calibration re-audit is mandatory) or only **assets** (charts 2/6 — cheaper, safer).
3. **One "efficient frontier" pick** — if the team could do exactly one thing this quarter, which, and
   what pre-registered metric + gated harness (chart 7 discipline) would prove it worked.

## 6 · Hard constraints (do not violate)

- **Honesty boundary.** Nothing is a "result" until an independent, decontaminated, gated harness
  (chart 7) clears a pre-registered threshold with a CI excluding zero. Keep `candidateOnly:true` /
  `canClaimAGI:false` until then. Do not claim a capability the code cannot yet demonstrate.
- **Fail-closed.** Any weight-training recommendation must include re-auditing calibration/abstention
  afterward — distilling gated behaviors into weights can un-learn the inference-time gate that made
  them safe (chart 8's `EVALGATE` node).
- **Verify before citing.** If you reference an external method or paper, confirm it resolves; flag any
  identifier you could not verify rather than asserting it.
- **Additive.** Prefer changes that extend the existing `run_case()` pipeline and its ablation flags
  over rewrites — the suppressibility of every stage is what makes the evidence story work.

**First message back to the requester should be:** your verification report (§5.1) plus your single
efficient-frontier pick (§5.3), so they can steer before you expand the full ranked list.