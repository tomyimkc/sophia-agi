# Spark Theory-Test Forecast — pre-registered, fastest-first GPU queue

**Status:** design-intent forecast registry (no capability claim; `canClaimAGI` stays `false`).
**Purpose:** before spending Spark GPU time, *pre-register* each theory test with an explicit
**forecast** (predicted outcome + confidence), ordered fastest-first. After each run we paste the
**actual** result and **analyse the divergence**. The point is not just to measure — it is to
measure *how well we predicted*, and to update our design priors when we are wrong. This file is
the living scoreboard for that loop.

> Why forecast at all? A pre-registered prediction turns every run into two results: the
> measurement, and the calibration of our own model of the system. A test that confirms the
> forecast costs little surprise; a test that *violates* it is where the repo learns something.
> (Same discipline as `docs/06-Roadmap/*Preregistration*.md` and the failure ledger — extended to
> forecasting, not just gating.)

## The loop (the workflow this file drives)

```
pick fastest unrun test  ─▶  pre-register hypothesis + FORECAST (here)
        ▲                              │
        │                              ▼
  update priors  ◀── analyse ◀──  run on Spark ──▶ paste ACTUAL result (here)
   (next forecast)   divergence
```

Each test below has four blocks: **Hypothesis**, **Gate & bar**, **FORECAST** (filled now),
**RESULT + DIVERGENCE** (filled after the run). Fastest GPU job first, so we get the most
forecast-vs-actual signal per hour.

---

## The verification gates these tests answer to

So the forecasts are grounded, here is what "pass" means for each gate the queue touches:

| Gate | Where | Pass bar | Notes |
|---|---|---|---|
| **VALIDATED judge gate** | `tools/run_lora_uplift_validation.py` | notMock · ≥2 judge families · judge≠subject · mean pairwise **κ≥0.40** · ≥3 seeds · 95% CI excludes 0 | the bench-A gate; κ is the recurring fail (bench-a-03: κ=0.394) |
| **LowRamGate (NVFP4 cert)** | `tools/certify_lowram.py` | **mean_kl ≤ 0.05 AND top1 ≥ 0.97**; protected slice KL ≤ 0.10 AND agree ≥ 0.95 | deterministic bars, **no judge / no κ** — the cleanest path to a real GO |
| **Virtue 2-family gate** | `tools/run_{sophrosyne,dikaiosyne}_eval.py` + labelers | 2-family consensus labels (κ≥0.40) → paired Δ vs a real no-gate baseline, bootstrap CI | same κ exposure as bench-A |
| **Faithfulness measurement** | `tools/run_faithfulness_battery.py` | *measurement, not a GO* — reports intrinsic flip-rate + cued/uncued split with bootstrap CIs | a number to characterise, never a claim |
| **No-overclaim meta-gate** | `make claim-check`, `tools/lint_claims.py`, `tools/claim_gate.py` | every number carries CI + N/seeds + ≥2 families OR CI-excludes-0, else labelled candidate | enforced in CI on every push |

**Design read:** judge-based gates (bench-A, virtues) keep landing CANDIDATE because κ sits on the
0.40 line. The **NVFP4 cert is the only queued test whose bar is deterministic** — so it is both the
fastest *and* the likeliest to yield a genuine GO. That is why it is first.

---

## The queue (fastest GPU job first)

| # | Test | Gate | Est. GPU time | One-line forecast | Confidence |
|---|---|---|---|---|---|
| **T1** | NVFP4 mixed-precision cert on an **existing** adapter (`KEEP_SUFFIXES=down_proj`) | LowRamGate | **~10–20 min** (eval-only, no train) | down_proj-bf16 lifts top1 but **still lands ~0.94–0.96 < 0.97** on the v3/v4 adapter → NO-GO, closer | 60% |
| **T2** | CoT **faithfulness** battery, real local model (`FAITH_MODEL=ollama:qwen2.5:7b`) | measurement | **~20–45 min** | local instruct model shows **unfaithfulCueUse ≈ 0.25–0.45**; intrinsic flip-rate moderate | 55% |
| **T3** | **Sophrosyne** temperance gate-improves-decisions (`--bench-virtues`, sophrosyne only) | virtue 2-family | **~45–70 min** (judge farm) | positive decision Δ on explicit signals, but **κ < 0.40** again → CANDIDATE | 65% |
| **T4** | **Council-vs-generalist** on one real trained discipline adapter | council eval + VALIDATED | **hours** (LoRA train + eval) | per-discipline adapter beats monolith **on-discipline** (+small Δ), risks **off-discipline regression** | 50% |

Rationale for the order: T1 is eval-only and deterministic (no κ, no judge farm) → fastest + a real
GO is achievable. T2 is a measurement (no flaky gate) and high research value (CoT faithfulness is a
frontier question) but needs many generations. T3 reuses the bench-A judge farm (slow) and inherits
the κ risk. T4 needs training, so it is last despite being the highest-thesis-value test.

---

## T1 — NVFP4 mixed-precision cert (down_proj held bf16)

**Hypothesis.** The v5 lever — holding the most KL-sensitive served projection (`down_proj`) in
bf16 while NVFP4-quantizing the rest — lifts top-1 agreement over the 0.97 floor without breaking
mean_kl, *on an already-trained adapter* (no retrain needed). Tests whether the top-1 gap is a
**quantization-granularity** problem (fixable at cert time) vs a **training** problem (needs v5).

**Gate & bar.** LowRamGate: mean_kl ≤ 0.05 **AND** top1 ≥ 0.97; protected KL ≤ 0.10, agree ≥ 0.95.

**Command (Spark; eval-only, points at an EXISTING adapter — no `--run-train`):**
```bash
# pick an adapter that already exists on the Spark (v4 completed per trainwatch; v3 also)
KEEP_SUFFIXES=down_proj \
QAT_ADAPTER=training/lora/checkpoints/olmoe-qat-spark-v4 \
CERT_NEVAL=256 \
bash scripts/run_local_benchmarks.sh --bench-b --execute     # B1 train SKIPPED, B2 certs the adapter
```

**FORECAST (2026-06-30).**
- mean_kl: **0.03–0.05** (✓ likely passes — v3 already hit 0.045 served-only).
- top1: **0.94–0.96** — improves over v3's 0.906 (holding down_proj removes its quant error) but
  **likely still short of 0.97**. Closing 0.906→0.97 by freezing *one* projection is a big ask.
- Verdict prediction: **NO-GO on top1, by a small margin.** Protected slice passes.
- Confidence: **60%** it stays NO-GO; ~30% it squeaks ≥0.97; ~10% mean_kl regresses if v4 (the
  over-fit lambda=0.01 adapter, protected_max_kl 0.71) is the one certified — **prefer v3**.
- If NO-GO: next lever = hold `down_proj,gate_proj` bf16, or cert a properly v5-trained adapter.

**RESULT (fill after run):** _pending_
**FORECAST vs ACTUAL divergence:** _pending — if top1 ≥ 0.97, the gap was granularity not training
(update prior toward "cert-time mixed precision suffices"); if mean_kl regressed, wrong adapter._

---

## T2 — CoT faithfulness battery (real local model)

**Hypothesis.** A local instruct model's chain-of-thought is **partly unfaithful**: when an answer
is swayed by an injected cue, the written reasoning often hides the cue (rationalises a post-hoc
justification). The battery's cued/uncued split exposes the `unfaithfulCueUseRate`.

**Gate & bar.** None — this is a **measurement** (reports rates + bootstrap CIs over `FAITH_SEEDS`).
The honest output is a characterised number, never a GO.

**Command (Spark; real local model via ollama):**
```bash
FAITH_MODEL=ollama:qwen2.5:7b-instruct@http://127.0.0.1:11434/v1 \
FAITH_SEEDS=3 \
FAITH_BATTERY=benchmark/faithfulness_cot_battery_v2.json \
SOPHIA_CAPTURE_THINKING=1 \
bash scripts/run_local_benchmarks.sh --bench-faithfulness --execute
```

**FORECAST (2026-06-30).**
- `unfaithfulCueUseRate`: **0.25–0.45** (cue-influenced answers whose reasoning hid the cue).
- cue-follow rate (answer changes with the cue): **0.30–0.55** (7B models are quite cue-suggestible).
- intrinsic flip-rate (answer flips when a load-bearing reasoning step is perturbed): **0.15–0.35**.
- Confidence **55%** the unfaithful rate lands in [0.25, 0.45]; wider tails are plausible because
  small instruct models can be *either* very cue-suggestible *or* near-random on the harder items.
- Design read if confirmed: motivates a **faithfulness verifier** in the gate stack (the repo
  currently gates *provenance*, not *reasoning faithfulness* — a candidate new seat).

**RESULT (fill after run):** _pending_
**FORECAST vs ACTUAL divergence:** _pending — a much-lower unfaithful rate than forecast would mean
the v2 battery isn't discriminating on this model (revisit battery design); much higher would
strengthen the case for a faithfulness gate._

---

## T3 — Sophrosyne temperance gate improves decisions

**Hypothesis.** The temperance gate (`agent/sophrosyne.py`, MQ = ε − δ over expenditure-vs-demand)
catches both **excess** (verbosity, over-hedging, over-retrieval, runaway loops) and **deficiency**
(premature stop, under-answer). With the gate on, decisions score better than a real no-gate
baseline, judged by 2 independent families.

**Gate & bar.** Virtue 2-family gate: consensus labels (κ≥0.40) → paired Δ (gate vs baseline) with a
bootstrap CI excluding zero.

**Command (Spark; reuses the bench-A judge farm — apply the parallel-families patch first):**
```bash
bash scripts/run_local_benchmarks.sh --bench-virtues --execute   # sophrosyne + dikaiosyne; ~judge-farm cost
```

**FORECAST (2026-06-30).**
- Decision-quality Δ (gate − baseline): **+0.05 to +0.15**, CI likely excludes zero on the explicit-
  signal arm.
- Inter-judge **κ ≈ 0.30–0.42** — **on the 0.40 line again** (the repo's recurring reliability wall);
  forecast leans **CANDIDATE, not VALIDATED** (65%).
- The "derived-signal-weak-on-raw-text" ledger pattern predicts the *derived* temperance signal
  underperforms an *explicit* anchor — so the gate helps most when handed an explicit budget signal.

**RESULT (fill after run):** _pending_
**FORECAST vs ACTUAL divergence:** _pending — a κ≥0.40 here would be the first judge-based VALIDATED;
analyse whether forced-choice + a stronger 2nd family is what tipped it._

---

## T4 — Council vs generalist on a real trained adapter

**Hypothesis.** A discipline-routed council (per-seat 3B LoRA + per-seat verifier) catches more
errors *on its discipline* than one monolithic gate (Branch-Train-MiX / S-LoRA premise). The risk is
**off-discipline regression** and that the tiny seed corpora make each adapter weak.

**Gate & bar.** `tools/eval_council_vs_monolith.py` (error-catch Δ) then the VALIDATED judge gate on
the on-discipline uplift.

**Command (Spark; LoRA-train ONE cheap discipline first, then eval — hours):**
```bash
# train one seat from its seed pack, then eval council vs monolith with the real adapter as answer source
python tools/train_lora.py --model Qwen/Qwen2.5-3B-Instruct \
  --train training/council_seeds/mathematics.jsonl --output training/lora/checkpoints/council-math
python tools/eval_council_vs_monolith.py --emit agi-proof/benchmark-results/council-vs-monolith.json
```

**FORECAST (2026-06-30).**
- On-discipline error-catch Δ (council − monolith): **+0.05 to +0.20** (positive but modest — small
  corpus caps the gain).
- Off-discipline: **flat to −0.05** (mild regression risk from over-specialisation).
- Confidence **50%** — genuinely uncertain; the seed packs are LIMA-scale (~3–4 traces/seat), which
  could be too thin to move a 3B meaningfully. This is the test most likely to **surprise**.

**RESULT (fill after run):** _pending_
**FORECAST vs ACTUAL divergence:** _pending — if the tiny corpus still moves on-discipline catch,
update strongly toward "verifier-routing > corpus size"; if flat, the council needs bigger seats._

---

## Filling in a result (protocol)

When a run lands:
1. Paste the receipt numbers into that test's **RESULT** block (point estimate + CI + N/seeds).
2. Write the **DIVERGENCE** block: was the forecast inside/outside the CI? Which direction? *Why* —
   model wrong, gate quirk, underpowered, adapter mismatch?
3. If the forecast was wrong, state the **prior update** in one line (what we now believe).
4. Log the GO/NO-GO (or measurement) in `agi-proof/failure-ledger.md` as usual — this file is the
   forecast scoreboard; the ledger stays the authoritative outcome record.
5. Pick the next fastest unrun test; pre-register its forecast before running.

**Calibration tally (update as results land):**

| Test | Forecast | Actual | Inside forecast? | Prior updated |
|---|---|---|---|---|
| T1 | top1 0.94–0.96, NO-GO | _pending_ | _—_ | _—_ |
| T2 | unfaithful 0.25–0.45 | _pending_ | _—_ | _—_ |
| T3 | Δ>0 but κ<0.40, CANDIDATE | _pending_ | _—_ | _—_ |
| T4 | on-disc +0.05–0.20 | _pending_ | _—_ | _—_ |

`canClaimAGI` stays **false** throughout; forecasts are predictions, not results, and every landed
number goes through `make claim-check` before it is called anything but candidate.
