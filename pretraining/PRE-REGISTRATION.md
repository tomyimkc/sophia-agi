# Pre-Registration — `pretraining/` studies

Registered **2026-06-25** on branch `claude/deepseek-pretraining-alignment-o281ju`.

These gates are committed **before** the headline numbers are trusted, in the same spirit
as [`../agi-proof/preregistered-thresholds.md`](../agi-proof/preregistered-thresholds.md):
state the falsification condition first so results can't be back-fit.

## Scope & honesty bound

Every study runs a **toy, pure-Python** model. The claim is about **methodology and
research taste**, never about frontier capability. Any reading of these artifacts as a
capability claim is a misreading; the reports carry an explicit `honesty_note`.

## Gates

### G1 — The nano model genuinely learns
Held-out NLL after training must drop **≥ 0.05 nats** below the untrained model on a fresh
sample. *Falsified if* loss does not decrease (the "model" is not learning and no scaling
claim is valid). — `tests/test_pretraining.py::test_nano_model_reduces_loss`

### G2 — Power-law fit is exact on clean synthetic data
Given losses generated from a known `L = E + A·D^-p`, `fit_with_floor` must recover `p` and
`A` to **< 1 %** with log-space **r² > 0.999**. *Falsified if* the fitter cannot recover a
planted law. — `tests/test_pretraining.py::test_fit_recovers_known_powerlaw`

### G3 — Scaling extrapolation (the planning claim)
Fitting `L(D)` on the smaller sizes and extrapolating to the largest held-out size must land
within **10 % relative error** of the measured loss. *Falsified if* the gate fails — then the
law does not support planning in this regime. Reported as `prediction.passes_10pct_gate` in
`scaling/scaling-curve-latest.json`.

### G4 — Floor identifiability is reported honestly
The free-floor fit's recovered `E` is checked against the analytic entropy. We do **not**
claim recovery; we **report** `recovered_floor_within_15pct` and, when false, label the data
range UNDER-IDENTIFIED. *Falsified if* the report ever claims floor recovery without the
check passing.

### G5 — MoE routing does not silently collapse
The top-1 router's busiest expert must take **< 85 %** of tokens on the test source (perfect
balance = 1/`k`). *Falsified if* routing collapses to one expert without the report flagging
`balanced: false`. — `tests/test_pretraining.py::test_moe_routes_without_collapse`

### G6 — Synthetic-data collapse is real, not asserted
The low-fidelity synthetic run must show held-out loss **rising** after its optimum
(`collapsed: true`), and the high-fidelity run must not, on the same budget. *Falsified if*
the collapse story can't be reproduced from the script.

### G7 — Data passport is fail-closed
Exact duplicates must share a `dedup_cluster`; rows without a license must be flagged
`unlicensed`; rows below the quality floor flagged `low_quality`. *Falsified if* any flagged
condition passes silently. — `tests/test_pretraining.py::test_passport_dedup_and_flags`

### G8 — Eval matrix surfaces gaps, not just coverage
The matrix must report **uncovered cells** (including `multimodal`), never round coverage up.
*Falsified if* gaps are hidden. — `tests/test_pretraining.py::test_eval_matrix_builds`

### G9 — Vertical-data validators fail closed
Missing provenance (`source`/`license`), malformed steps, out-of-range reward, and unknown
record types must all be rejected with explicit errors. *Falsified if* an invalid record
validates. — `tests/test_pretraining.py::test_vertical_validators_fail_closed`

### G10 — Reviewer agent is honest & fail-closed
The role-conditioned reviewer agent must (a) carry `canClaimAGI: false` and never name itself
an AGI agent, (b) return `cannot_assess` (never `pass`) for any missing report, and (c)
surface the known critiques (high dup rate, low eval coverage) rather than hide them.
*Falsified if* the agent claims AGI, passes an unassessable study, or suppresses a real
concern. — `tests/test_pretraining_agent.py`

### G11 — Autonomous runner is a real, fail-closed loop
The autopilot must (a) improve on its starting config by reading measured results (real
closed loop, not a fixed grid), (b) score a diverged run `inf` and exclude it from `best`
(never fabricate a finite number), (c) carry `canClaimAGI: false`, and (d) **never launch a
paid GPU run autonomously** — the RunPod escalation is dry-run by default and `launched` is
always `False`, gated behind explicit launch + cost ceiling + API key. *Falsified if* the
loop fabricates a score, the escalation launches without the guard, or it claims AGI.
— `tests/test_pretraining_autopilot.py`

## Status

| Gate | Status (2026-06-25) |
|---|---|
| G1 nano learns | ✅ PASS |
| G2 fit exact | ✅ PASS |
| G3 extrapolation ≤10 % | ✅ PASS (≈3 % on default run) |
| G4 floor identifiability | ⚠️ HONEST-FAIL — floor UNDER-IDENTIFIED in tested range, reported as such |
| G5 MoE no-collapse | ✅ PASS |
| G6 synthetic collapse | ✅ PASS |
| G7 passport fail-closed | ✅ PASS |
| G8 eval gaps surfaced | ✅ PASS |
| G9 validators fail-closed | ✅ PASS |
| G10 reviewer agent honest | ✅ PASS |
| G11 autonomous runner fail-closed | ✅ PASS |

### G12 — Cost governor is fail-closed against the ceiling
The RunPod calibration machinery must (a) refuse to project/launch trials whose projected
spend exceeds the hard USD ceiling (`can_afford`/`guard` fail-closed), (b) parse the eval
ladder into an objective and return `+inf` for a malformed/missing ladder (never a fabricated
score), (c) build DRY-RUN commands by default and never launch a paid pod from library code,
and (d) keep the paid launch behind a typed `confirm=SPEND` + ceiling gate in
`calibrate-runpod.yml`. *Falsified if* any code path can spend past the ceiling, fabricate an
objective, or launch without the gate. — `tests/test_pretraining_calibrate.py`

| Gate | Status (2026-06-26) |
|---|---|
| G12 cost governor fail-closed | ✅ PASS |

G4 is intentionally listed as an honest-fail: it is the study's most useful lesson, not a
defect — you cannot read off the irreducible loss without runs near saturation.
