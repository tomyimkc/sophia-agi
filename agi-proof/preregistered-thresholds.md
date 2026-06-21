# Pre-Registered Proof Thresholds

These thresholds prevent moving the goalposts after seeing results.

## Claim Levels

| Level | Claim | Required evidence |
|---:|---|---|
| 0 | Corpus project | Valid schema, source records, training examples |
| 1 | Source-discipline benchmark | Visible benchmark pass rates and leaderboards |
| 2 | AGI-candidate architecture | Agent/RAG/local-model paths plus gate and memory |
| 3 | Strong AGI-candidate evidence | Hidden reviewer tasks, ablations, long-horizon logs |
| 4 | Expert-reviewable AGI evidence | External benchmark results and clean-clone replication |
| 5 | Proven AGI | Not claimed by this repository |

## Minimum Data To Escalate Beyond Level 2

- At least 100 hidden reviewer tasks across four or more domains.
- Baseline/ablation deltas showing Sophia-full beats raw model and component
  removals by a meaningful margin.
- At least three long-horizon runs: 30 minutes, 2 hours, and 1 day.
- Distribution-shift experiment with pre-test, append-only learning phase, and
  fresh post-test.
- Failure ledger published beside success results.
- Independent reviewer reproduction from a clean clone.

## Falsification Rules

Sophia must not be marketed as AGI if any of these occur:

- hidden benchmark performance collapses relative to visible benchmarks;
- raw model baselines match or beat Sophia-full;
- old knowledge is silently overwritten during learning;
- reviewers cannot reproduce reported local results;
- long-horizon tasks require frequent human steering;
- external benchmark results are absent but described as achieved.

## RLVR experiment (verifier-as-reward GRPO)

A separate, narrower pre-registration for `tools/run_rlvr.py` (RLVR: the
deterministic provenance verifier IS the GRPO reward). See
[docs/09-Agent/RLVR-Experiment.md](../docs/09-Agent/RLVR-Experiment.md).

- **Pre-registered claim (LIVE, gated):** on the held-out entity-disjoint split,
  mean reward / pass@1 rises vs the untrained base adapter at ~0 false-positive
  regression. Validated only under the no-overclaim gate
  (`provenance_bench.aggregate._is_validated`: notMock + ≥2 judge families +
  Cohen's kappa ≥ 0.40 + ≥3 runs + 95% bootstrap CI excludes 0).
- **Offline invariants (asserted in CI today):** reward is deterministic,
  monotone in the correct direction, a forbidden-assertion completion scores
  negative, the `agent.verifiers` seam is actually invoked, the reward is bounded
  in [-1, 1], and the train/eval split is contamination-free (entity-disjoint).
  `python tools/run_rlvr.py --model mock` exits non-zero if any fail.
- **Falsification:** the RLVR claim must not be reported as met if (a) a single
  run or a `mock` run is the only evidence; (b) train-split numbers are reported
  as headline; (c) the held-out split is not entity-disjoint; or (d) the gate
  (`_is_validated`) is not cleared. It is explicitly **not** an AGI claim — RLVR
  improves pass@1 within the verifier's reach, not the base model's capacity.
