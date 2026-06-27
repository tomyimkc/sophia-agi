# RunPod real-GPU training evidence

Artifacts from `.github/workflows/train-runpod.yml` — real-GPU, gate-disciplined QLoRA
training + eval-ladder runs. **Candidate-only / illustrative evidence. Not a capability
claim. `canClaimAGI` = false.**

## Eval-ladder runs

### `qwen2.5-3b-lora-1ep-seed0.eval-ladder.public-report.json`

First successful real-GPU P6 run — QLoRA (4-bit) fine-tune of `Qwen/Qwen2.5-3B-Instruct`
(1 epoch, seed 0), evaluated base-vs-adapter × with/without the Sophia provenance gate on
a 32-item held-out provenance set (philosophy / psychology / history / religion).

**Headline (honest null):** the adapter is **capability-neutral on held-out content** —
base and adapter both score **23/32 = 71.9%** content. It redistributed within domains
(philosophy ↑ 6→8, psychology ↓ 6→5, religion content ↓ 5→4) for a net-zero content
change. The +6.2pt on the *combined* (format∧content) channel is format-driven, not a
content gain. The provenance gate left content unchanged at every rung.

**Scope:** single-judge, 32-item, 1-epoch — well below the VALIDATED bar (κ≥0.40 /
2 judge families / ≥3 runs / 95% CIs excluding zero). It validates the GitHub-Actions →
RunPod → eval-ladder path end-to-end and records a legitimate directional null to build on
(more epochs / larger curated data / the `moe/adapt` allocation), not a result to overclaim.

Provenance (run URL, head SHA, GPU pool, artifact id + sha256) is in the report's
`provenance` block. The eval-ladder numbers are transcribed from the workflow's own step-7
report printed in the run log; the W2 `promotion.public-report.json` verdict lives inside
the run artifact.

### `qwen2.5-3b-lora-3ep-seed0.eval-ladder.public-report.json`

Same setup, **3 epochs** (seed 0). This is the "does more training move it" follow-up to the
1-epoch null.

**Headline (directional positive):** 3 epochs converts the 1-epoch null into a **+12.5pt
CONTENT uplift** — adapter **27/32 = 84.4%** content vs the FP16 base's **23/32 = 71.9%**
(content per suite: philosophy 6→7/9, psychology 6→7/9, history 6→8/8, religion held at 5/6 —
the *protected* suite did not regress). So **more training is the lever that moves content
here**, unlike epoch-1.

**W2 promotion verdict = `reject`, but NOT on quality.** Every content/quality gate passed
(`protected_floor_content`, `no_total_regression` total_after=0.719, `contamination_zero`
overlap=0, `provenance_complete`); the scorecard and continual-plasticity decision both say
*promote* (targetDelta +0.219). The sole breach is `solver_attestation` → **"held:
z3-solver not installed"** — the formal oracle abstains for a *missing-tooling* reason, not a
capability regression. The gate correctly refuses to rubber-stamp without its solver.

**Scope:** still single-judge / 32-item / 1-seed → **candidate-only / directional**, not
VALIDATED. The +12.5pt is not separable from noise at this power, and the training traces were
religion-repair while religion (protected) held flat and *other* domains rose (transfer or
noise — undetermined). A VALIDATED claim needs the κ≥0.40 / 2-judge-family / ≥3-run / 95%-CI
protocol. `canClaimAGI` unchanged (false).

### 1-epoch vs 3-epoch (content channel)

| run | epochs | config | base content | adapter content | uplift | W2 verdict |
|---|---|---|---|---|---|---|
| #5 | 1 | uniform | 23/32 (71.9%) | 23/32 (71.9%) | **0.0pt** (null) | — |
| #7 | 3 | uniform | 23/32 (71.9%) | **27/32 (84.4%)** | **+12.5pt** | reject (z3 unavailable; all quality gates passed) |
| #9 | 3 | `--lora-rank-alloc` + z3 | 23/32 (71.9%) | **29/32 (90.6%)** | **+18.75pt** | **promote** (solver-checked, z3) ✓ |

> Run #9 (`qwen2.5-3b-lora-3ep-rankalloc-seed0.eval-ladder.public-report.json`) exercises BOTH
> new mechanisms on hardware, **verified from the run's artifact + logs**:
> - **moe/adapt allocator ran** — literal `rank_pattern={q/k/v/o:17, gate:16, up:15, down:16}`
>   (uniform r=16). The redistribution is **modest** (15–17), so the allocated and uniform-3ep
>   adapters are near-identical in capacity — which is exactly why the +6.2pt over uniform-3ep is
>   **eval noise, not the allocation**.
> - **z3 W2 oracle PROMOTED** — `solver_attestation: accepted (z3)`, `oraclePromote: true`,
>   `verdict: promote`, `breachingInvariants: []` — flipping run #7's reject-on-missing-z3 to a
>   real solver-checked promote, exactly as proven offline.
>
> The 90.6% content and +18.75pt over base remain **single-judge / n=32 / 1-seed → candidate-only**.
> A validated comparison needs the P6 preregistration protocol (≥3 seeds, 2 non-qwen judge
> families, κ≥0.40, 95% CI).

The `*.log` files in this directory are earlier SFT / SSH-smoke run logs.
