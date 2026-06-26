# `pretraining/` — Pretraining research artifacts

> Honest, pure-Python, falsifiable pretraining-direction research, built to the same
> no-overclaim discipline as the rest of Sophia.

This package exists to answer one question concretely: *what would Sophia look like if it
spoke the language of a pretraining (data / algorithm) researcher?* It is **deliberately
small** — every study runs in pure Python with **no numpy/torch** — because the
contribution here is **verifiable methodology**, not scale. Where a frontier lab fits a
scaling law on thousands of GPU-hours, we fit the *same functional form* on a toy model
whose **irreducible loss is known in closed form**, so the fit can be checked, not trusted.

This maps directly onto the two tracks of the DeepSeek 预训练研究员 role:

| Track | Artifacts here |
|---|---|
| **预训练算法研究** (algorithm) | scaling laws (`scaling/`), optimizer dynamics & stability (`optimizer_probe/`), novel architecture / MoE routing (`architecture/`) |
| **预训练数据研究** (data) | data provenance/governance (`data_passport/`), mixture-ratio 配比 (`data_mixing/`), synthetic-data scaling & collapse (`synthetic_scaling/`), multi-dimensional eval coverage (`eval_matrix/`), vertical data schemas — agent / feedback / multimodal (`vertical_data/`) |

The shared substrate is `nano/` — a real, tiny, hand-backpropped language model trained on
an order-*k* Markov source. Because the source's average conditional entropy is computable
(`nano.source_entropy`), it serves as the **known irreducible-loss floor `E`** that every
study checks its fits against.

## The studies

Each study has a `--quick` mode (seconds, used by tests/CI) and a default mode (minutes)
that writes a dated-style `*-latest.json` report next to the script.

### Algorithm direction

- **`scaling/` — data-scaling law `L(D) = E + A·D^-p`.** Measures held-out loss across
  training-set sizes, fits the law, and runs a **pre-registered prediction**: fit on the
  smaller sizes, extrapolate to a held-out larger size, check predicted-vs-measured.
  *Headline finding:* the known-floor fit hits **r² ≈ 0.93** and the extrapolation passes a
  **10 % gate (≈3 % error)** — but the *free-floor* fit **cannot recover `E`** from data
  that hasn't approached saturation (it collapses `E→0`). That identifiability limit is a
  real property of scaling-law fitting, reported honestly rather than hidden.
  → `python -m pretraining.scaling.run_scaling`

- **`optimizer_probe/` — optimizer dynamics & stability.** SGD vs momentum vs Adam across a
  learning-rate grid, reporting final loss, divergence, max grad-norm, and a grad-spike
  count — i.e. a **stability/performance frontier**, not just a winner.
  → `python -m pretraining.optimizer_probe.run_probe`

- **`architecture/` — top-1 MoE vs dense at matched active compute.** A faithful toy of the
  sparse-scaling idea, with a load-balancing penalty so routing **collapse** is surfaced.
  `ARCHITECTURE.md` documents the real DeepSeek **MLA + fine-grained MoE** design this
  gestures at. *Honest result:* at nano scale a single dense block already captures the
  order-2 source, so MoE doesn't win — and the report says so.
  → `python -m pretraining.architecture.run_arch`

### Data direction

- **`data_passport/` — per-row training-data provenance.** Stamps every row of a JSONL pack
  with a passport (content hash, source, license, quality score, MinHash near-dup
  signature, dedup cluster) and emits a **datasheet**. Fail-closed: unlicensed / low-quality
  rows are flagged. Hooks into the existing `provenance_bench` contamination guard.
  *Real finding on this repo:* the committed `sophia-math-code-curriculum` pack is
  **~60 % near-duplicates** (144 rows → 57 clusters) and fully unlicensed — exactly the kind
  of issue this surfaces. → `python -m pretraining.data_passport.build_passport <pack.jsonl>`

- **`data_mixing/` — mixture-ratio (配比) sweep at fixed token budget.** Sweeps the mix of
  two sources and finds the ratio that minimizes loss on a target distribution. *Finding:*
  the optimum tracks the target, and a **blended target has an interior optimum** — pure
  single-source data is suboptimal. → `python -m pretraining.data_mixing.run_mixing`

- **`synthetic_scaling/` — synthetic-data scaling & collapse.** Adds increasing synthetic
  data (from a drifted generator) to a fixed real budget. *Finding:* high-fidelity synthetic
  scales and saturates near the floor; **low-fidelity synthetic collapses the model** once
  it dominates — quantity can't substitute for fidelity.
  → `python -m pretraining.synthetic_scaling.run_synthetic`

- **`eval_matrix/` — multi-dimensional eval coverage.** Buckets the repo's eval packs into a
  capability×domain matrix (auto vs human scoring) and **surfaces the gaps**. *Finding:*
  ~22 packs / ~451 cases cover only **9/90 cells (10 %)**; multimodal is entirely uncovered.
  → `python -m pretraining.eval_matrix.run_matrix`

- **`vertical_data/` — typed schemas for 垂类 data.** Provenance-aware, fail-closed validators
  for **agent trajectories, user-feedback signals, and multimodal items**, with example
  records. Gives the multimodal/agent/feedback streams a first-class, checkable format.

### Reviewer agent (`agent/`)

- **`agent/` — the pretraining-researcher role, as a *critic*.** Deliberately **not an "AGI
  agent"** (Sophia's charter forbids AGI claims — every output carries `canClaimAGI: false`).
  It embodies the data/algorithm researcher role as an auditable rubric and **reviews the
  studies above**: each is checked against its pre-registered gate, overclaims are flagged,
  and a next experiment is proposed. Fail-closed — a missing report is `cannot_assess`, never
  `pass`. Two uses: a **regression harness** (re-run after any change to confirm the gates
  still hold) and a **persona benchmark** (the role scoring the repo). Deterministic and
  offline; an optional `--llm` critique is purely additive and degrades gracefully.
  → `python -m pretraining.agent.run_review` (exit 0 iff every study produced its artifact)

- **`autopilot/` — autonomous experiment runner (propose → run → read back → iterate).** A
  real closed loop: a search **strategy** proposes a config, the **local backend RUNS it**
  (real nano training, free, CPU), the loop reads the measured loss, and the strategy decides
  the next config from that result — until it converges or the trial budget is spent. Three
  tasks: **learning-rate** tuning (adaptive hill-climb — walks 0.05→0.0125 reading losses),
  **data-mixture** 配比 (ternary search → interior optimum), and **compute allocation** (N vs
  D at fixed compute → finds an interior compute-optimal point, Chinchilla-flavoured). Honest
  by construction: every score is measured, **diverged runs are recorded as failures
  (score=inf), never fabricated**, and `canClaimAGI: false`. A `--escalate` flag emits a
  **gated** RunPod plan for the winning config — **dry-run by default, cost-guarded, never
  auto-spends GPU money** (launching needs explicit `--launch` + `--cost-ceiling` +
  `RUNPOD_API_KEY`, and even then it only prints the real `tools/runpod_train.py` command for
  you to run). → `python -m pretraining.autopilot.run_autopilot --task lr`

  **Real-pipeline escalation (Step 1, built):** to search the *actual* Qwen LoRA pipeline so
  configs transfer directly, `autopilot/` ships a **cost governor** (hard USD ceiling,
  fail-closed), a **RunPod backend** (config→`runpod_train.py` argv + eval-ladder→objective
  parser, verified +15.6 uplift on the committed ladder), and a **calibration harness** that
  runs ONE real trial to replace every cost estimate with a measured number. The paid launch
  is a gated GitHub Actions workflow (`calibrate-runpod.yml`, `confirm=SPEND` + ceiling).
  Everything is dry-run-testable with no cost: `python -m pretraining.autopilot.calibrate
  --ceiling 1.0`. Full plan + grounded cost table: [`autopilot/SCOPE.md`](autopilot/SCOPE.md).

## Honesty posture

- **No fabricated numbers.** Every `*-latest.json` is produced by running the script.
- **Known floor.** Fits are checked against the analytic source entropy, not eyeballed.
- **Pre-registration.** Falsification gates live in [`PRE-REGISTRATION.md`](PRE-REGISTRATION.md).
- **Toy, and labelled toy.** These demonstrate *method and taste*, not frontier results.
  This is consistent with Sophia's charter: *don't out-train frontier labs — be honest and
  checkable* (see [`../VISION.md`](../VISION.md)).

## Run everything

```bash
python tests/test_pretraining.py                       # fast property tests
python -m pretraining.scaling.run_scaling              # + each study's CLI
```
