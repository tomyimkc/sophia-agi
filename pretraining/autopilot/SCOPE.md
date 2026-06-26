# SCOPE — searching the real Qwen LoRA pipeline on RunPod

Plan to extend the autonomous runner (`pretraining/autopilot/`) from toy nano experiments to
searching the **real** Qwen2.5 LoRA pipeline on rented RunPod GPUs, so configs the loop likes
transfer directly. Built to the repo's honesty discipline: measured before claimed, cost
capped, spending always a deliberate human action.

## Grounding facts (measured, not guessed)

| Fact | Value | Source |
|---|---|---|
| Real pod price | **$0.69/hr** (24–48 GB tier) | `agi-proof/benchmark-results/runpod-train/sft-3seed-*.log` |
| Default model | Qwen2.5-**3B**-Instruct (7B optional) | `tools/runpod_train.py` |
| Method | 4-bit QLoRA (`--rslora --neftune`, holdout early-stop) | pipeline |
| Training data | small — ~765 SFT rows, ~624 DPO, 144 math-code | `training/local_sophia_7b/*` |
| Per-trial pipeline | prepare → train_lora → eval_ladder (4 rungs) → promote → tar back | `tools/runpod_train.py` |
| Objective | `score(adapter+gate) − score(base)`, combined channel | `eval_ladder_objective.py` (verified +15.6 on the committed ladder) |
| Built-in safety | watchdog auto-deletes pod; 3 hr hard timeout | startup script |
| **Known risk** | SSH egress from agent shell times out → launch from GitHub Actions | log + failure-ledger |

## Components

| ID | Component | Status |
|---|---|---|
| **C1** | Real RunPod backend: config→`runpod_train.py` argv + eval-ladder→objective parser | ✅ Step 1 (`runpod_backend.py`, `eval_ladder_objective.py`) |
| **C4** | Cost governor: hard USD ceiling, projected/actual spend, fail-closed guard | ✅ Step 1 (`cost_governor.py`) |
| **—** | Calibration harness + gated CI launch (1 real trial → measured cost) | ✅ Step 1 (`calibrate.py`, `.github/workflows/calibrate-runpod.yml`) |
| **C2** | Search space + objective: LoRA rank/alpha/lr/epochs/NEFTune knobs + sampler | 🟡 Step 2 — space + sampler built (`search_space.py`); `runpod_train.py` passthrough args still TODO |
| **C3** | Trial-efficient strategy: ASHA / successive-halving over expensive trials | ✅ Step 2 (`asha.py`, demo `run_asha_demo.py`) — cost-governed, fail-closed |
| **C5** | Orchestration: parallel pods, spot-eviction retries, idempotent resume, provenance | ⬜ Step 2 |

**C2/C3 status (built, offline-verified):** the ASHA scheduler prunes bad configs on real
measured results (nano demo: ~39 % fewer runs than naive, converges to the best learning
rate) and is **fail-closed on the cost ceiling** — it refuses to start a rung it cannot fully
afford, spending $0 past the ceiling. The LoRA search space + deterministic sampler exist and
tag each knob as *transfers-today* (epochs/seed/model) vs *needs-passthrough* (rank/alpha/lr/
NEFTune). The one remaining Step-2 code change before a full-space GPU sweep is threading the
passthrough knobs through `runpod_train.py`'s remote command builder.

## Cost estimate (anchored at $0.69/hr)

Per-trial ≈ 0.5–0.85 GPU-hr (3B) → **~$0.35–0.60** central (7B ≈ 1.5–2×). Add ~25–30 % buffer
for image pulls / evicted spot pods / SSH retries. The calibration run replaces these with a
measured number.

| Tier | What | 3B est. | 7B est. |
|---|---|---|---|
| Calibrate | 1 real trial — measure true time/cost | $0.50–1.50 | $1–2 |
| Small | ~10 configs, ASHA, 1 seed | $3–8 | $6–15 |
| Medium | ~24 configs, ASHA + 3-seed final on top-2 | $8–20 | $15–40 |
| Generous | ~40 configs, 3 seeds, full eval | $20–45 | $40–90 |

Recommended ceilings: **$25** (3B small→medium) or **$75** (7B medium). The governor enforces
whatever number you set.

## How to run Step 1 (the $1 calibration)

1. Set the repo secret `RUNPOD_API_KEY` (Settings → Secrets → Actions).
2. Dispatch **calibrate-runpod** (Actions tab): `confirm = SPEND`, `cost_ceiling_usd = 1.00`.
3. The workflow pre-flights the cost projection (aborts if it exceeds the ceiling), runs ONE
   timed trial, then writes `calibration-measured.json` — actual cost, the objective uplift,
   and sweep-tier costs **re-projected from your measured per-trial cost**.
4. Use those measured tiers to choose a ceiling for the Step-2 sweep.

Locally, everything is dry-run-testable with **no cost**:

```bash
python -m pretraining.autopilot.calibrate --ceiling 1.0          # projection only, no pod
```

## Honest caveats

- The calibration trial submits the **default** pipeline. Tuning rank/lr/mixture (C2) needs
  `runpod_train.py` to expose those passthrough args — a small, flagged Step-2 change.
- Nano-loop search ≠ Qwen hyperparameters; this scope is exactly about closing that gap by
  searching the real pipeline. Step 1 proves the cost/latency model and the launch path first.
