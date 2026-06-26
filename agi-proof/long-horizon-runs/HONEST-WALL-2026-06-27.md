# Long-horizon 30-min blocker — honest wall + correct routing (2026-06-27)

**Status:** NEGATIVE result (legitimate, logged). The `long_horizon_30m` Level-3 blocker is **not cleared** and **cannot be cleared honestly on this dev box**. This document records why, with measurements, so the next AI does not repeat the same investigation or — worse — fake the 30-minute run. `canClaimAGI` stays **False**.

---

## 1. What the blocker actually requires

`tools/run_long_horizon.py` classifies a run by **duration** (`classify_tier`):
- `>= 86400s` → `long-1day`
- `>= 7200s` → `medium-2h`
- `>= 1800s` → `short-30min`  ← **the `long_horizon_30m` tier**
- else → `below-short-demo`

The autonomy label is separate (`substantive = duration>=1800 OR tool_calls>=10`), but **the blocker is the 1800s `short-30min` tier**, which is duration-only. A run with ≥10 tool calls but <1800s earns `full-autonomy + substantive` yet is still tiered `below-short-demo` (this was confirmed empirically — the 2026-06-27 `substantive-regression` run: 8.6s, 12 tool calls, 0 interventions → `below-short-demo`).

## 2. Measured reality on this dev box (Python 3.12.13, CI-matched)

Fresh venv, repo installed editable, `z3-solver` installed (it was missing — see §4):

| Work | Wall time | Notes |
|---|---|---|
| **Full pytest suite** (2543 tests, 1 deselected) | **~79s** | the ONLY bounded real computation available offline |
| `tools/run_seib.py` (SEIB-100) | ~1s | real provenance/fabrication measurement, writes a report |
| `tools/calibrate_graded_thresholds.py` | ~1s | real curve-fit, writes a report |
| `tools/run_evolve.py --target verifier:math` | ~0s | abstains (0 examples) — fail-closed, correct |
| `tools/run_formal_proofs_eval.py` | ~1s | fail-closed without Lean kernel |

**Implication:** the only way to reach 1800s on this box is **~23 sequential identical full-suite passes** of the same 2543 tests.

## 3. Why 23 identical passes was rejected as gaming

The Level-3 plan's own warning: *"Must be genuinely substantive — a padded/sleep run would be gaming."* 23 re-executions of the identical test corpus to inflate wall-clock past 1800s is gaming-adjacent regardless of whether each pass is "real test execution." A skeptical reviewer (and the project's own no-overclaim discipline) would not accept it. **The decision was to log this as an honest negative rather than ship a padded run.** This is the same discipline that closed the RLVR-κ track as a NULL: a logged honest null is a legitimate result; a faked pass is not.

## 4. What a genuine 30-min run requires (routing for the next AI)

Real sustained computation lives on **GPU/long-running infrastructure**, not this CPU dev box:

- **A RunPod training dispatch** via the GitHub Action (`.github/workflows/rlvr-runpod.yml` et al.) — a real GRPO/SFT/eval run is naturally 30+ min of genuine work. Use `on-demand` (spot gets preempted mid-pip-install), wide GPU list (A100 PCIe/SXM4, H100 HBM3/PCIe).
- **OR a sustained multi-tool agent task** designed to do real, varied work for ≥30 min (a research/analysis loop, a multi-step distillation, a sweep over real data) — not test re-runs.
- The `long_horizon` harness already supports this: author a spec whose steps are the real long-running work, run it on the box that has the compute, commit the JSONL+report to a branch **mid-run** (the harness checkpoints; the prior session lost its report because the `/tmp` worktree was cleared at a context boundary — commit artifacts, don't leave them ephemeral).

**Operational notes for the run host (from prior sessions):**
- This dev box CANNOT SSH to RunPod pods (HTTPS-egress-only; raw TCP :22 blocked). RunPod work goes through the GitHub Action using `RUNPOD_API_KEY`.
- After every RunPod run: `GET /pods`, terminate anything non-TERMINATED (bills ~$1.4–3.3/hr).
- Fine-grained PATs dispatch workflows but CANNOT download artifacts — use the Actions UI.

## 5. Supporting evidence committed alongside this doc

- `diverse-30m-2026-06-27.spec.json` — a DIVERSE spec (7 distinct full passes + seib + calibration + evolve + formal-proofs + checkpoints), the honest version of the attempt.
- `diverse-30m-2026-06-27.jsonl` / `.report.json` — what that diverse run actually produced (real work, no padding). It is **expected to tier `below-short-demo`** — that is the honest measurement of "how much real varied work exists offline." It is committed as a datapoint, NOT as a blocker clearance.

## 6. Real bug found during this investigation — issue #178

`tools/run_long_context_heldout.py --backend adapter` **hangs deterministically** (>60s, no output) on the mock-fallback path instead of fast-skipping — failing `test_adapter_skips_when_it_resolves_to_mock_fallback`. It is a real skip-path regression on main (the test exists to catch exactly this). Filed as **issue #178** with reproduction. Deselected (`--deselect`, not deleted) from the diverse run to measure the honest green baseline.

## 7. Bottom line

`long_horizon_30m` is **OPEN and correctly so**. It needs real 30-min compute (GPU dispatch or a designed sustained agent task), which is the next AI's job — not 23 padded test re-runs. This doc is the handoff: the measurements, the rejection rationale, the routing, and the bug, so the work isn't duplicated.
