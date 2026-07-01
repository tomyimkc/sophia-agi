# Continuation prompt — sophia-agi AGI-proof evidence tools

You are continuing work on the `tomyimkc/sophia-agi` repo. A package of five drop-in
evidence tools was built, adversarially reviewed, and had its offline-fixable defects
corrected. Your job is to (1) confirm the fixes on the current tree, (2) do the
maintainer-level changes the drop-in could not, and (3) run the first live experiment.
You have the repo checked out and (per the prior session) live compute + model backends.

## Read these first (in the repo, ~30 KB total)

- `agi-proof/candidate-tools-2026-07-01/RESPONSE-TO-REVIEW.md` — **the source of truth.**
  A defect-disposition table: what was fixed in code, what was redesigned in docs, what
  still needs a maintainer. Start here.
- `agi-proof/candidate-tools-2026-07-01/README.md` — package overview + binding audit +
  leverage-ordered run guide.
- `agi-proof/candidate-tools-2026-07-01/failure-ledger-additions.md` — the five Open
  ledger rows (paste-ready for `agi-proof/failure-ledger.md`).
- `tools/ablation_no_executor.patch.md` — the WS-D ablation redesign (single-flag plan
  withdrawn; two options).
- `agi-proof/candidate-tools-2026-07-01/HANDOVER-REVIEW-PROMPT.md` — the original review
  mandate, for the deeper design questions.

## State of the tree

- Branch `feat/realtime-grounding-loop` (verify `git rev-parse HEAD`; last known `537279f9`).
- All package files are **untracked additions** — nothing tracked was modified. Files:
  `tools/{make_independent_hidden_pack,run_t1_gated_self_training,run_arc_agi_sophia,run_long_horizon_timed}.py`,
  `tools/ablation_no_executor.patch.md`, `tests/test_*.py` (4), and the
  `agi-proof/candidate-tools-2026-07-01/` docs. They are NOT git-committed yet.
- Tools import the repo (`agent.*`, `tools.hidden_eval_protocol`), so run everything with
  the repo root on the path: `PYTHONPATH=. python3 …`.

## What is already fixed (confirm, don't redo)

- **D1** model binding: `default_client(spec).generate(system, user)` + mock-provider
  fail-closed guard (was the non-existent `Model(spec).complete(prompt)`).
- **D3** WS-A now validates via the runner's real `tools.hidden_eval_protocol.validate_pack`,
  not `schema.json`; fail-closed if the repo isn't importable.
- **D8** WS-B binds real `REWARD_CLEAN`/`REWARD_ABSTAIN`; a substantive pass requires
  clean-level reward + a gold-reference match; no-reference items never credited.
- **D5** WS-B `run_round(..., protected={suite:[items]})` marks `EvalMetric(protected=True)`
  so a protected-suite regression triggers reject.
- **Q-C(1)** ARC parser strips fences / ignores prose / extracts the grid block.
- 24 offline tests pass — name the four package files (`tests/` holds ~529 repo tests, so a
  bare `pytest tests/` runs the whole repo suite, not these): `PYTHONPATH=. python3 -m pytest
  tests/test_make_independent_hidden_pack.py tests/test_run_t1_gated_self_training.py
  tests/test_run_arc_agi_sophia.py tests/test_ws_d_free_wins.py -q` (4+6+10+4=24).

## Your tasks, in priority order

1. **Verify the fixes on the current HEAD.** Re-run the 24 package tests (the four named files
   above, not a bare `pytest tests/`). Re-confirm the six binding
   targets still match (the tree moves): `agent.model.default_client/generate`,
   `agent.gate_reward.{reward,REWARD_CLEAN,REWARD_ABSTAIN}`,
   `agent.continual_plasticity.{evaluate_update,EvalMetric,UpdateCandidate}`,
   `agent.long_horizon.{build_ledger,run_long_horizon}`, the `Ablation` fields, and
   `tools.hidden_eval_protocol.validate_pack`. Report any drift.

2. **RUN WS-C first — ARC-AGI-1, gate-off vs gate-on (highest evidence-per-effort).**
   - Scope to ARC-AGI-1/2 only (NOT ARC-AGI-3 — exact-grid-match is a category error for the
     interactive -3; keep them in separate rows).
   - Place the official ARC-AGI-1 eval tasks under a `--tasks` dir (the tool does not vendor
     the dataset; it fail-closes if the dir is absent).
   - Run two arms on the same tasks: gate-on and a gate-off baseline. Report **accuracy AND
     coverage** (accuracy-at-matched-coverage / selective risk), not a single headline number.
   - Command: `PYTHONPATH=. python3 tools/run_arc_agi_sophia.py --tasks <dir> --adapter <spec>
     --out agi-proof/benchmark-results/arc-agi1.public-report.json`
   - Acceptance gate: ≥N tasks scored end-to-end, 0 backend failures, exact-match accuracy +
     abstention rate reported per arm, artifact under `agi-proof/benchmark-results/`.
   - **Sanity-check the parser on real model output** before trusting any score (a silent
     grid-drop reads as a reasoning failure).

3. **Maintainer module change — D4 (make WS-D `--minutes` binding).** In
   `agent/long_horizon.py`: add `deadline_monotonic: float | None = None` to
   `run_long_horizon`, `import time`, and at the top of the node loop (after the
   `while steps < max_nodes:` guard) `if deadline_monotonic and time.monotonic() >=
   deadline_monotonic: break`. Thread it into `run_subagent` for sub-node interruption.
   Then wire `tools/run_long_horizon_timed.py` to pass it. Add a test that a tiny budget
   actually stops a multi-node run.

4. **Decide the D2 ablation question.** Either (Option 1) run the EXISTING `sophia-no-tools`
   mode and report the delta on cases WITHOUT `requiresToolLog`/`requiresMemoryDiff` (no code
   change), or (Option 2) implement a real multi-site `use_executor` gate + case exclusion.
   Option 1 is the honest cheap path unless you can justify Option 2. Construct any `Ablation`
   with keyword args only (it has `use_context_packing`/`context_packing_policy` fields).

5. **WS-B — wire a real trainer (only if you have GPU + a training harness).** The promote
   path is unreachable until an SFT/DPO step produces `gen_after` with after>before, and
   `heldout_shifted` is a GENUINELY disjoint distribution (not a paraphrase of
   `heldout_scored`). Route judge-free where possible (verifier-checkable code/math held-out);
   report passAt1, not meanReward; pin one commit across seeds.

## Discipline constraints (non-negotiable — this repo's whole point)

- **Fail-closed:** no backend / missing data / missing dep → an "environment artifact, not a
  score" report or a clean abstain; never fabricate a number, never crash. The mock provider
  (`cfg.kind == "mock"`, auto-selected when no API key) counts as NO backend.
- **No overclaim:** every artifact carries `candidateOnly:true`, `level3Evidence:false`,
  `canClaimAGI:false`. A public number needs ≥2 judge families or a CI excluding zero. These
  close rungs/cells, NOT the AGI claim.
- **Decontam floor** on any eval data; an unreadable corpus fails closed.
- Leave the five ledger rows **Open** until their stated acceptance gate is actually met by a
  real run — then, and only then, flip the specific row and cite the artifact + checksum.

## Report back

For each task attempted: what you ran (exact command), the artifact path + checksum it wrote,
the measured numbers with their honesty fields, and which ledger row (if any) it moves and to
what status. If a binding drifted or a run fail-closed, say so plainly with the cause. Do not
commit or flip any ledger row on the basis of a mock-backend or fail-closed run.