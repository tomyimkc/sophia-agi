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

## sophia-math-code-curriculum (Qwen2.5-7B MATH + CODE)

**Status:** OPEN — registered 2026-06-25 on branch `claude/sophia-math-code-curriculum`
**before** any GPU curriculum training. Structured manifest:
`agi-proof/sophia-math-code-curriculum/preregistration.json`. Oracle split:
`agi-proof/sophia-math-code-curriculum/oracle-split.md`.

### Base model and recipe

- Base: `Qwen/Qwen2.5-7B-Instruct`
- QLoRA 4-bit, ~2 epochs, `--mask-prompt`, seeds `{0, 1, 2}` (≥3 for cited numbers)
- Training data: sympy/exec-verified **synthetic** curriculum (NOT sealed benchmarks)
- Decontamination: `python tools/build_local_sophia_dataset.py --check` → **CLEAN**
- Holdout seal: `agi-proof/sophia-math-code-curriculum/heldout-seal.manifest.json`
  (`python tools/seal_math_code_heldout.py --check`)

### Training oracle vs evidence oracle (THE ONE RULE)

| Family | Purpose | May cite as benchmark evidence? |
|---|---|---|
| **Training oracle** | `agent/math_verifier.py` (sympy), `agent/code_verifier.py` (sandboxed exec), synthetic packs (`tools/gen_math_pack.py`, verifier-synthesis) | **No** — curriculum gate only |
| **Evidence oracle** | Sealed MATH/GSM8K/HumanEval/MBPP style samples, `benchmark/code_tasks.json` eval, hidden reviewer pack | **Yes** — when ≥3 seeds, 95% CI excludes 0 |

Training-oracle passes must **never** be cited as MATH/GSM8K/HumanEval/MBPP proof.

### Pre-registered evidence thresholds (≥3 seeds, CI excludes 0)

| Metric | Threshold | Notes |
|---|---|---|
| MATH-style accuracy Δ vs base | ≥ **+5.0%** | Style sample until official MATH licensed |
| GSM8K-style accuracy Δ vs base | ≥ **+3.0%** | Numeric exact-match |
| HumanEval-style pass@1 Δ vs base | ≥ **+5.0%** | Hidden-test execution |
| MBPP-style pass@1 Δ vs base | ≥ **+3.0%** | Hidden-test execution |
| religion/history protected suites | **no regression** | `promote_adapter` protected floor |

### Success / failure

- **Success (headline):** all evidence-oracle thresholds met; contamination CLEAN; holdout seal OK;
  `lint_claims.py` OK; `canClaimAGI` stays **False**.
- **Failure:** protected regression → reject; seal/contamination break → abort; evidence thresholds
  not met → honest negative in failure-ledger; training-oracle cited as benchmark → violation.

## sophia-7b-train-verify (Qwen2.5-7B SFT + DPO + disjoint evidence)

**Status:** OPEN — registered 2026-06-25 on branch `claude/sophia-7b-train-verify` **before**
any GPU training run. Structured manifest:
`agi-proof/sophia-7b-train-verify/preregistration.json`. Oracle split:
`agi-proof/sophia-7b-train-verify/oracle-split.md`.

### Base model and recipe

- Base: `Qwen/Qwen2.5-7B-Instruct`
- QLoRA 4-bit, ~2 epochs, `--mask-prompt`, seeds `{0, 1, 2}` (≥3 for cited numbers)
- Data: `training/local_sophia_7b/` after `build_local_sophia_dataset.py --check` → **CLEAN**
- Holdout seal: `agi-proof/sophia-7b-train-verify/heldout-seal.manifest.json`
  (`python tools/seal_sophia_7b_holdout.py --check`)

### Training oracle vs evidence oracle (THE ONE RULE)

| Family | Purpose | May cite as third-party evidence? |
|---|---|---|
| **Training oracle** | Moral Gate, eval ladder CONTENT, `promote_adapter`, invariant suite (`solverChecked`), positive control, DPO hard-negatives | **No** — release gate only |
| **Evidence oracle** | Vectara HHEM (lead), hidden reviewer pack, HF Open LLM + Artificial Analysis (context) | **Yes** — when run with keys and ≥3 seeds |

Training-oracle passes must **never** be cited as Vectara/hidden-pack/leaderboard proof.

### Pre-registered evidence thresholds

| Metric | Threshold | Notes |
|---|---|---|
| Vectara HHEM hallucination rate | ≤ **12.0%** mean across ≥3 seeds | Baseline recorded before training |
| Hidden-pack calibration Δ | ≥ **+10.0%** with 95% CI excluding 0 | Reviewer-controlled / third-party pack |
| MMLU-Pro vs base | regression ≤ **2.0** points | Honest abstention tradeoff — report even if gate passes |
| Internal release gate | `promote_adapter` promote + `solverChecked: true` | **Not** counted toward evidence thresholds |

### Success / failure

- **Success (headline):** all evidence-oracle thresholds met; contamination CLEAN; holdout seal OK;
  `lint_claims.py` OK; `canClaimAGI` stays **False**.
- **Failure:** protected-suite regression → reject; seal/contamination break → abort; evidence
  blocked → ledger blocker (no training-oracle substitute); MMLU-Pro regression >2.0 → withdraw headline.

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

## Conformal abstention gate (C1) + abstention-aware scoring (C3)

**Status:** OPEN — registered 2026-06-26 on branch `claude/agi-asi-research-ideas-g5ss38`.
Implements the first two candidates of
[Frontier-Research-Implementation-Plan](../docs/11-Platform/Frontier-Research-Implementation-Plan.md).

### C1 — conformal abstention

- **Pre-registered claim (LIVE, gated):** a split-conformal threshold fitted on a
  third-party labeled outcome pack carries its distribution-free coverage guarantee
  onto a held-out split — i.e. on held-out data a new correct answer is accepted with
  probability ≥ 1−α (within finite-sample slack) — and the conformal answer/abstain
  boundary achieves a strictly better risk-coverage frontier than the hand-picked
  `DEFAULT_THRESHOLDS` boundary on the same rows. Pre-registered α grid: {0.05, 0.10, 0.20}.
- **Offline machinery check (asserted in CI today):** `tools/fit_conformal_policy.py
  --synthetic N` fits, runs the held-out validity check, and reports VALID for all α on a
  deterministic synthetic set where nonconformity is a noisy predictor of correctness.
  `decide_conformal` is downgrade-only (gate-failed never yields `answer`) and fails safe
  to the default boundary with no artifact. `python3 tests/test_conformal_gate.py` and
  `tests/test_guarded_conformal.py` exit non-zero if any invariant fails.
- **Falsification:** the C1 claim must not be reported as met if (a) the only evidence is
  synthetic or a single run; (b) held-out correct-coverage falls below 1−α beyond slack;
  (c) the conformal boundary shows no risk-coverage gain over the hand-picked boundary at
  matched risk; or (d) the labeled pack is self-authored (third-party pack required). It is
  explicitly **not** an AGI claim — it certifies an abstention boundary, not capability.

### C3 — abstention-aware scoring (Kalai reform)

- **Pre-registered claim (methodology):** under a confident-wrong penalty λ ≥ λ*, a
  fail-closed policy that abstains scores at least as well as always-guessing on a real
  labeled run; report the λ-curve and the break-even λ* alongside (never replacing) the
  legacy binary score.
- **Offline invariants (CI):** an abstention scores 0 (never penalised as wrong); raising
  λ never raises the score; an unknown action fails closed as answered (never a free
  abstention). `python3 tests/test_abstention_scoring.py`.
- **Falsification:** not a capability claim; must not be reported as a Sophia advantage
  unless computed on a real model run with {correct, action} labels from a third-party pack.
