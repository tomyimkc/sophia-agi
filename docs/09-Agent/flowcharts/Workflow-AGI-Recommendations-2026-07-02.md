# Workflow Analysis — Chart-vs-Code Verification & AGI/ASI-Direction Recommendations

> **Date:** 2026-07-02 · **Branch reviewed:** `claude/sophia-workflow-agi-recommendations-2z8pbt` @ `29a7712`
> **Scope:** verify `docs/09-Agent/flowcharts/` against the code on this branch, then rank the
> most leverage-per-effort moves toward AGI/ASI-grade workflows.
> **Honesty boundary:** every recommendation below is a *candidate direction*. Nothing here is a
> result; `candidateOnly:true` / `canClaimAGI:false` holds until an independent, decontaminated,
> gated harness (chart 7 discipline) clears a pre-registered threshold with a CI excluding zero.

---

## 1 · Verification report (chart vs. code)

Verified by direct read/grep on this branch — not from the charts' own claims.

1. **CONFIRMED — gate wiring.** `agent/gate.py` uses `agent.benchmark_checks` (top-level import,
   line 9) and lazily imports `agent.verifiers` (lines 45, 74), `agent.sector_council` (line 89),
   and `agent.claim_router` (line 158) inside functions. Chart 4's fan-out is real; the nuance is
   that most of it is lazy/conditional, so a static import graph under-reports the gate's reach.
2. **CONFIRMED — self-evolution drive.** `agent/self_evolving_agent.py` imports
   `agent.continual_plasticity` (line 47) and `agent.continual_retention` (line 53) and documents
   the evolve → no-hack → promote → retain cycle in its module docstring. Chart 6 matches.
3. **CONFIRMED — reward chain.** `agent/gate_reward.py` invokes `agent.gate.check_response`
   (lazy, line 127). `agent/multiaxis_reward.py` imports `agent.gate_reward` and
   `agent.verifiers` top-level and `agent.prosoche` lazily (line 181, explicitly to avoid an
   import cycle). Chart 8's signal box is wired as drawn.
4. **CONFIRMED — update-gate thresholds.** `agent/continual_plasticity.py:216`
   `evaluate_update(...)` defaults are exactly as charted: `min_target_delta=0.03`,
   `max_protected_regression=0.01`, `require_artifacts=2`, optional retention evidence. There is
   also an uncharted `evaluate_update_multigoal(...)` variant (line 137) — worth adding to chart 6.
5. **CONFIRMED — suppressible spine.** `tools/run_hidden_eval_sophia.py:run_case()` (line 1156)
   exposes all nine ablation flags (`use_intake`, `use_kb`, `use_evidence`, `use_council`,
   `use_gate`, `use_memory`, `use_tools`, `allow_repair`, `use_claim_router`), with named ablation
   arms (`sophia-no-*`, `sophia-claim-router`). Note `use_claim_router` defaults **False** — the
   claim router is charted as part of the gate but is off in the default pipeline.
6. **CONFIRMED — the measurement→learning diagnosis.** `agent/calibration.py`,
   `agent/abstention_scoring.py`, and `agent/selective_risk.py` contain no loss, gradient, torch,
   or mlx reference. They score; they do not train. The core thesis of the charts holds.
7. **STALE SEAM CLAIM (finding).** The handover/chart framing "gate reward not yet wired into
   `tools/run_rlvr.py`" is out of date: `run_rlvr.py` already supports `--reward gate` and
   `--reward multiaxis` (`gate_reward.make_grpo_reward()` at lines 384–397), runs gate-reward
   invariants in its offline mode (line 663), and uses `gate_violations` for curriculum ordering
   (line 214). What is genuinely unwired is the **distilled PRM**
   (`tools/distill_process_reward_model.py` — zero references in `run_rlvr.py`). Chart 8's
   SIGNALS→CORPORA edge is therefore *partially live*, not aspirational.
8. **CONFIRMED — the featurizer stub.** `agent/activation_probes.py:99`
   `build_hidden_state_featurizer(...)` is a pure `raise RuntimeError` seam. It blocks
   `tools/energy_verifier_head.py`, `tools/probe_representation_training.py` (W5), and the
   real-feature upgrade of the W1 PRM, exactly as documented.
9. **MISSING ARTIFACTS (finding).** The `out/` directory does not exist on this branch:
   `out/Sophia-SkillOpt-CrossPollination.md` (S1–S5) and `out/Sophia-Untapped-Training-Theses.md`
   (referenced by `agi-proof/untapped-training-2026-07-01/README.md` line 5) are both absent. The
   S-series prospectus is unavailable here; the W-series README survives and is self-contained.
   The provenance caveat on the charts (built from a working clone with 388 uncommitted mods) is
   real — this branch has 291 `agent/*.py` files.
10. **CONFIRMED — harness layer.** `tools/run_ablation_sophia.py`, `run_learning_shift.py`,
    `run_long_context_sophia.py`, `run_long_horizon.py`, `run_replication_check.py`, and
    `agi-proof/preregistered-thresholds.md` all exist. The only RLVR reports are mock-offline
    (`rlvr.public-report.json` `mode:"mock-offline"`); the live run is an open ledger row
    (`rlvr-live-run-not-yet-gated-2026-06-21`). Also uncharted but relevant:
    `tools/run_t1_gated_self_training.py` already closes generate → `gate_reward.reward()` →
    `evaluate_update()` as an asset-level loop — an existing bridge between charts 6 and 8.

**Verdict on the core diagnosis:** correct, with one amendment. Sophia measures epistemics far
more than it learns from them — but the first learning bridge (gate-as-reward GRPO) is already
*coded and offline-validated*; it has simply never been *run live and gated*. The gap is now less
"write the training signal" and more "execute and gate the runs that exist."

---

## 2 · Ranked recommendations (leverage = capability gain ÷ effort)

Each entry: files touched · why it raises capability while preserving honesty · effort ·
weights-vs-assets · strongest objection.

### R1 — Execute the pre-registered live gate/multiaxis RLVR run (rank 1)
- **Touches:** nothing new in `agent/`; ops only — `tools/run_rlvr.py --reward gate|multiaxis`
  on a rented CUDA GPU (via GitHub Actions per repo guardrail), then the post-train re-audit
  (`agent/calibration.py`, `agent/abstention_scoring.py`, eval ladder).
- **Why:** it is the only measurement→learning bridge that is fully coded, CI-validated offline,
  and already pre-registered (ledger row `rlvr-live-run-not-yet-gated-2026-06-21`). It converts
  the repo's central thesis from scaffold to evidence either way — a null result is also
  information, and it exercises the entire chart-7 claim discipline end to end.
- **Effort:** ≈0 new code; 1–2 days ops + ≥3 seeds of GPU time.
- **Weights?** YES (chart 8) → post-training calibration/abstention re-audit is mandatory.
- **Strongest objection:** the GRPO stack targets GLM-4-9B on CUDA, not the Qwen2.5-3B MLX
  backend Sophia actually serves — a positive result validates the *method*, not the *product*.
  And `gate_reward` is question-free by design: it cannot distinguish abstain-on-answerable from
  abstain-on-trap, so the policy can drift toward over-abstention while reward climbs. The
  re-audit must therefore report answerable-coverage, and per `rlvr-harness-traps`, the
  load-bearing metric is pass@1/VSC on the held-out split — never meanReward.

### R2 — Wire the W2 calibration proper-scoring loss into the MLX path (rank 2)
- **Touches:** `tools/train_calibration_objective.py` (loss exists and is offline-proven to lower
  ECE 0.33→0.22 by the repo's own instrument), a new corpus-extraction step turning harness
  reports' (confidence, correct, action) rows into DPO pairs, `training/` + `mlx_lm lora`.
- **Why:** closes the "scores but doesn't train" gap on the honesty axis, on the model Sophia
  actually serves. Measured before/after by `agent/calibration.py`, so the improvement claim
  uses Sophia's own instrument.
- **Effort:** medium — 1–2 weeks.
- **Weights?** YES → mandatory re-audit; regression on protected suites rejects the adapter.
- **Strongest objection:** a post-hoc temperature/Platt calibrator on the frozen model captures
  much of the ECE gain for ~zero risk — the weight-level version must be pre-registered to *beat
  that baseline*, not merely beat uncalibrated output, or it isn't worth the un-learning risk.

### R3 — Implement `build_hidden_state_featurizer` (rank 3, infrastructure key)
- **Touches:** `agent/activation_probes.py:99` (MLX residual-stream hook, keeping the lazy
  fail-closed contract); unblocks `tools/probe_representation_training.py` (W5),
  `tools/energy_verifier_head.py`, and the real-feature W1 PRM.
- **Why:** one week of plumbing unblocks three documented directions at once; it changes no
  behavior by itself (text-feature path stays the offline default).
- **Effort:** ~1 week incl. tests.
- **Weights?** NO — assets/infra only (cheapest risk class).
- **Strongest objection:** at 3B scale the residual stream may simply not carry a strong
  truthfulness signal; pre-register a probe-AUC floor *before* building so a weak signal is a
  recorded negative result, not a sunk cost rationalized upward.

### R4 — Promote the claim router to default-on via ablation evidence (rank 4)
- **Touches:** `tools/run_ablation_sophia.py` (`sophia-claim-router` arm already exists),
  `agent/claim_router.py`; flip `use_claim_router` default in `run_hidden_eval_sophia.py` only
  after the evidence clears `evaluate_update()` discipline.
- **Why:** cheapest capability lever in the repo — per-claim verification is charted as central
  to the gate (chart 4) yet is off by default; the harness can price it in days, additively.
- **Effort:** days.
- **Weights?** NO — pipeline asset (charts 1/4).
- **Strongest objection:** noisy claim segmentation could raise false rejections on multi-claim
  answers and *reduce* answerable coverage; the promotion gate must include a coverage delta as a
  protected metric, not just fabrication catch-rate.

### R5 — Wire the distilled PRM into `run_rlvr.py` as a dense reward arm (rank 5)
- **Touches:** `tools/distill_process_reward_model.py` → new `--reward prm` arm in
  `tools/run_rlvr.py`; real features after R3.
- **Why:** dense per-step reward attacks GRPO's sparse-reward variance problem; the label
  pipeline and held-out-domain generalization measurement already exist.
- **Effort:** 1–2 weeks, sensibly sequenced after R1's live harness exists.
- **Weights?** YES → mandatory re-audit.
- **Strongest objection:** the PRM is a learned proxy for a symbolic verifier that is *still
  available* as the true reward — optimizing against the proxy invites Goodhart precisely in the
  region (steps the symbolic checker can't reach) where the proxy's labels are least trustworthy.

### R6 — Provenance-weighted SFT (W3) (rank 6)
- **Touches:** `tools/provenance_weighted_training.py`, `agent/source_ranking.py` (deterministic
  trust tiers), per-example weights into the `training/*.jsonl` → `mlx_lm lora` path.
- **Why:** near-free to implement; teaches source discipline at the weight level using a signal
  that is already deterministic and auditable.
- **Effort:** days–1 week.
- **Weights?** YES → mandatory re-audit.
- **Strongest objection:** likely the smallest effect size on any pre-registered metric, and
  trust-weighting risks distribution collapse toward one register of source; rank it last and
  let R1/R2 results decide whether it is worth a run.

---

## 3 · Efficient-frontier pick (one thing this quarter)

**R1 — run the live, gated gate/multiaxis RLVR experiment.**

- **Pre-registered metric (already on the ledger):** held-out, entity-disjoint **pass@1** rise vs
  the untrained base adapter, at ~0 false-positive regression, under
  `provenance_bench.aggregate._is_validated` — notMock, ≥2 judge families, Cohen's κ ≥ 0.40,
  ≥3 runs, 95% bootstrap CI excluding 0. pass@1/VSC is the load-bearing metric; meanReward is
  explicitly not.
- **Gated harness (chart 7 discipline):** `tools/run_rlvr.py` live arms (`--reward gate` and
  `--reward multiaxis`, plus the `--graded-craving` variant), ≥3 seeds, entity-disjoint split
  already enforced by the offline invariants; report lands in
  `agi-proof/benchmark-results/*.public-report.json` with `candidateOnly:true` until
  `_is_validated` passes.
- **Mandatory fail-closed step (chart 8 EVALGATE):** post-training calibration/abstention
  re-audit — ECE, selective risk, **and answerable-coverage** (the abstention-collapse inversion
  is the failure mode this reward was built to fix; prove it didn't overshoot).
- **Why this over R2/R3:** it requires no new code, it discharges the oldest open
  measurement→learning ledger row, and every other weight-path recommendation (R2, R5, R6)
  inherits its harness, its seeds protocol, and its re-audit template. If it fails, the failure
  is recorded against a pre-registered threshold — which is itself the AGI-standard behavior the
  charts are arguing for.

---

*This document makes no capability claims. All statuses above are code-verification findings or
candidate proposals; the claim ladder (`agi-proof/preregistered-thresholds.md`) remains the only
authority on what is proven.*
