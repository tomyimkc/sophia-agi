# RLVR next paid arm — pre-registered 2026-07-02 (before any dispatch)

> Registers the decision rule for the NEXT paid RLVR arm, written AFTER the
> 2026-07-02 six-run provenance sweep (runs #65–#70, commit `6a12a05`) and its
> contested promote verdicts (failure-ledger row
> `rlvr-sweep-identical-metrics-artifact-2026-07-02`), and BEFORE any pod is
> rented. candidateOnly:true / canClaimAGI:false regardless of outcome.
> Thresholds are FROZEN at dispatch time; any change after the first run
> constitutes a NEW pre-registration. Ledger row
> `rlvr-live-run-not-yet-gated-2026-06-21` stays Open until the decision rule
> below passes in full.

## Arm definition

- **Task:** `step` (process-verified derivations; `tools/run_rlvr.py --task step`).
- **Reward:** process reward. Baseline sub-arm uses the default symbolic
  per-step verifier; the PRM option is `--step-reward prm`
  (`tools/run_rlvr.py:626-642`) — NOTE: it fail-closes without
  `--prm-derivations <jsonl>` (`tools/run_rlvr.py:377`), and a derivations
  JSONL does not exist yet, so the PRM sub-arm is dispatched ONLY if that
  artifact is built and sealed first; otherwise the symbolic arm runs alone.
- **A4 levers (both noted, at most one flipped per comparison so effects stay
  attributable):**
  - mixed-outcome filtering — the concept lives in
    `provenance_bench/rl_data_curation.py:36-47` (`is_mixed_outcome` /
    `mixed_outcome_keep`); there is NO dedicated `--mixed-outcome-filter` CLI
    flag on `tools/run_rlvr.py` as of this commit, so wiring it (or invoking
    the curation path explicitly) is a precondition for that sub-arm.
  - `--advantage-shaping papo --lambda-neg 0.5`
    (`tools/run_rlvr.py:643-650`; PAPO asymmetric shaping, successes never
    double-counted).
- **Model / config:** Qwen/Qwen2.5-Coder-7B-Instruct, epochs 1.0, adapter
  `sophia-rlvr-v1`, seeds {0,1,2} minimum, ONE pinned commit for every pod.

## Measurement-integrity preconditions (from the sweep forensics; all mandatory)

1. Pinned commit that includes the forensics fixes: task/reward/seed-stamped
   adapter-eval paths with stale-path clearing
   (`tools/runpod_rlvr.py:500,517,879-880`), fail-closed passAt1 ingest
   (`tools/ingest_rlvr_eval.py:67-84`), and the per-report split/seed audit
   (`tools/eval_rlvr_adapter.py:86,151`).
2. Fresh or wiped network volume (no report or checkout reuse across arms).
3. Every report's `audit.effectiveSeed` must equal the dispatched seed and
   `audit.splitHash` must be DISTINCT across the three seeds; any collision
   voids the run (it reproduces the #66/#70 artifact signature).
4. The seed must reach GRPOConfig (training RNG), not only the data split.

## Decision rule (all required; frozen before dispatch)

1. **PRIMARY:** capability gated on `passAt1` ONLY (single greedy completion,
   verifier-accepted; emitted by `_score_step`,
   `tools/eval_rlvr_adapter.py:445`), per `tools/ingest_rlvr_eval.py`
   (`capabilityMetric="passAt1"`, fail-closed if absent). Adapter−base
   passAt1 delta positive with a 95% CI across ≥3 seeds excluding 0.
   `meanReward` is advisory only and can never gate.
2. **SECONDARY (reported, not gated):** `verifiedStepCoverage` (VSC,
   `tools/eval_rlvr_adapter.py:446`) before/after, per seed.
3. **PROTECTED SUITES:** regression ceiling 0.01 — any protected-suite drop
   > 0.01 is a hard reject regardless of the primary delta.
4. **POST-TRAINING RE-AUDIT (mandatory, before any verdict):** calibration,
   abstention, and answerable-coverage re-audit — a passAt1 win bought by
   abstention collapse or calibration loss is a FAIL.
5. **CONTAMINATION:** entity/family-disjoint held-out split sealed per seed
   (`audit.splitHash` recorded); seal or contamination break aborts the arm.

## Pre-declared outcomes

- **GO:** the result is still candidateOnly — it feeds the SSIL gate and the
  ledger; it does NOT by itself close `rlvr-live-run-not-yet-gated-2026-06-21`
  (that row additionally requires the judge-family/kappa criteria written on
  the row itself). canClaimAGI stays false.
- **NO-GO / artifact:** ledger row with the numbers; if any integrity
  precondition fails, the run is classified a measurement artifact (not a
  capability reading) and no capability statement is made in either direction.
