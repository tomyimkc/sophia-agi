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
