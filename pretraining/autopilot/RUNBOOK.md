# RUNBOOK — first real RunPod calibration trial

Operational steps to run ONE real RunPod LoRA trial and turn it into measured cost +
objective numbers (Step 1 of [`SCOPE.md`](SCOPE.md)). Written so the run is reproducible and
cost-safe. **Spending is always a deliberate human action** — nothing here auto-spends.

## 0. Security first

- **Never paste an API key into chat or commit it.** Keys belong in GitHub Actions secrets.
- If a key was ever exposed (e.g. pasted into a chat), **revoke it in the RunPod console and
  generate a fresh one** before using it.
- The workflows read `secrets.RUNPOD_API_KEY` and never print it.

## 1. Prerequisite — add the secret (one-time, user action)

GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**

- Name: `RUNPOD_API_KEY`
- Value: *(a freshly-rotated RunPod key)*

There is intentionally no API/agent path to write this secret — it must be set by a human in
the GitHub UI.

## 2. Why GitHub Actions, not a local/agent shell

`tools/runpod_train.py` rents a pod over the REST API (HTTPS) and then drives training **over
SSH**. Outbound SSH from the agent/CI-web shell has timed out (see
`agi-proof/failure-ledger.md`), which would rent a pod, fail to connect, and waste the spend.
GitHub Actions runners do not have that restriction — so all real runs go through a workflow.

## 3. Launch options

| Option | Workflow | On `main`? | Cost guardrail |
|---|---|---|---|
| **A (chosen)** | `train-runpod` | ✅ yes (dispatchable today) | 1-epoch scope + small data + pod watchdog/3 hr timeout |
| B | `calibrate-runpod` | ❌ feature branch only — must merge to `main` first | in-workflow `$` ceiling preflight + measured-cost report |

`workflow_dispatch` requires the workflow to exist on the **default branch** (`main`).
`calibrate-runpod.yml` currently lives only on `claude/deepseek-pretraining-alignment-o281ju`,
so it is not dispatchable until merged. `train-runpod` is already on `main`.

## 4. The chosen run — train-runpod, 1 epoch (≈ the $1 calibration)

A single 1-epoch run on the default recipe is one real trial: expected **~$0.40–0.60**
(anchored to the observed `$0.69/hr`), comfortably under a $1 ceiling.

**Dispatch (GitHub UI):** Actions → `train-runpod` → Run workflow →
- `confirm = RUN`
- `model = Qwen/Qwen2.5-3B-Instruct`
- `epochs = 1`
- `seed = 0`
- `interruptible = false` (on-demand — a clean cost measurement; spot can be evicted mid-run)
- Branch/ref: `claude/deepseek-pretraining-alignment-o281ju`

**Dispatch (REST API equivalent):**

```
POST /repos/tomyimkc/sophia-agi/actions/workflows/train-runpod.yml/dispatches
{ "ref": "claude/deepseek-pretraining-alignment-o281ju",
  "inputs": { "confirm": "RUN", "model": "Qwen/Qwen2.5-3B-Instruct",
              "epochs": "1", "seed": "0", "interruptible": "false" } }
```

## 5. After the run — turn it into measured numbers

1. Download the `train-runpod-artifacts` artifact (and read the run log).
2. From the log capture: wall-clock of the launch step (hours) and the quoted price
   (`costPerHr=<P>`).
3. Run the calibrator on the returned ladder:

   ```bash
   python -m pretraining.autopilot.calibrate \
     --from-result agi-proof/benchmark-results/runpod-train/<...>eval_ladder_adapter.json \
     --wall-clock-hours <H> --price-per-hr <P> --ceiling 1.0
   ```

   It writes `calibration-latest.json`: **actual trial cost, the combined-channel uplift
   `score(adapter+gate) − score(base)`, and sweep-tier costs re-projected from the measured
   per-trial cost.** Those measured tiers replace every estimate in `SCOPE.md`.

## 6. After calibration — the sweep (Step 2)

With a measured per-trial cost, pick a ceiling ($25 funds a 3B ASHA sweep over ~10–16 configs)
and use the cost-governed scheduler (`asha.py`) + the LoRA passthrough (now wired through
`$SOPHIA_HPARAMS`). Remaining build item: **C5** orchestration (parallel pods, eviction
retries, resume) — optional; the sweep works sequentially without it.

## Cost-safety summary

- 1-epoch scope + small curated data → minutes of GPU.
- Pod **auto-delete watchdog** + 3 hr hard timeout (built into `runpod_train.py`).
- On-demand (not spot) → no mid-run eviction skewing the measurement.
- The in-workflow `$` ceiling preflight exists in `calibrate-runpod.yml` (Option B) for when a
  hard cap must be enforced by the workflow itself.
