# Model card — Sophia-Wisdom-4B (M3 PILOT adapter, candidate-only)

**Status:** `candidate_only: true`, `validated_external: false`. A narrow, corpus-bound
**feasibility** result. **Not** market-beating, **not** validated, **not** AGI, **not** a
hallucination guarantee. `canClaimAGI: false`.

## What this is
A LoRA adapter on the **language tower** of the multimodal `google/gemma-3-4b-it`
(`Gemma3ForConditionalGeneration`), trained on the **deterministic gate-passed** Sophia-Wisdom
dataset to test one pre-registered question (`docs/06-Roadmap/Sophia-Wisdom-4B-M3-Pilot.md`):

> Does SFT on the gate-passed rows move the base's **prompt-scaffold (no-gate)** behavior toward
> the gated source-discipline target on ≥1 Sophia-native axis, without protected/retention regression?

## Training
- Base: `google/gemma-3-4b-it` (HF-gated; language tower only — vision tower frozen).
- Data: deterministic gate-passed rows rebuilt on the pod (`build_sophia_wisdom_dataset.py`, no
  live teacher → reproducible ~730 rows incl. 28% OASST/Apache general-instruction retention).
- LoRA r=16, α=32, dropout 0.05, target = language-tower attn+MLP projections; seq-len **1024**,
  **1 epoch, seed 0**, lr 1e-4, prompt-masked loss. ONE seed (pilot).
- Infra: RunPod CUDA (H100 80GB), SSH-free self-report workflow `wisdom-pilot-runpod`.
- **Adapter weights are NOT persisted** (the pilot measured behavior, not a shipped artifact);
  reproducible from the committed data + code + seed 0.

## Evaluation (full M1 instrument, N=354 × 3 runs, bootstrap 95% CI)
Artifact: `agi-proof/benchmark-results/wisdom-market/M3-pilot-eval.json`. Deterministic structural
scorers (judge-independent); **no LLM judge** → semantic quality is ILLUSTRATIVE, not headline.

**Pre-registered PRIMARY — adapter(prompt) − base(prompt), `*` = 95% CI excludes 0:**

| metric | Δ (improvement) | 95% CI | CI-clean |
|---|---|---|---|
| qualification_rate_on_contested | **+0.475** | [0.459, 0.508] | **\*** |
| tradition_merge_rate (儒/道)     | **+0.143** | [0.125, 0.161] | **\*** |
| false_attribution_rate          | **+0.014** | [0.012, 0.018] | **\*** |
| citation_fidelity               | +0.028 | [0.000, 0.083] | no (touches 0) |

→ **3 of 4 primary metrics improve CI-clean** (pre-registered bar: ≥1). Secondary:
`moral_route_accuracy` +0.569*, `provenance_accuracy` +0.013*; `tool_route_accuracy` Δ0,
`contested_fabrication_rate` +0.006 (ns).

**Guardrails (all hold):**
- Protected-history regression: base 0.083 → **adapter 0.000** (prompt_gate); ≤ base at every
  condition. Protected-religion: ≤ base everywhere (raw 0.098 → 0.010). **No protected regression.**
- Over-abstention (adapter): raw 0.019 / prompt 0.003 / prompt_gate 0.018 — all ≤ 0.10.
- Usefulness (retention proxy): useful_correctness **rises** at prompt (0.653→0.688) and prompt_gate
  (0.611→0.677); dips slightly at raw (0.502→0.460). No refusal collapse.

## Verdict
**PASS of all pre-registered criteria** (corrected 2026-06-26). Primary: a LoRA on the gate-passed
data produces a **CI-clean, non-regressing behavioral shift toward source-discipline habits at the
prompt (no-gate) layer** of gemma-3-4b — 3-seed robust, 3-judge-family corroborated. Stability
(criterion #3): on a **POWERED N=970 probe** the adapter **retains** general capability (Δ −0.001, CI
[−0.020, +0.018], gate GO; forgetting beyond ~2pts ruled out) — the ~12pt "forgetting" first reported
was N=34 small-sample noise (see caveat 2). NOTE this is still a **narrow, corpus-bound feasibility
PASS**, NOT a market-beating, validated, or AGI claim: deterministic marker metrics, ~730 training
rows, single base, single seed on retention. Stays `candidate_only`; `canClaimAGI: false`.

## Honest caveats (why this is NOT a headline / validated claim)
1. **Marker/structural metrics only.** Scores are deterministic marker- and forbidden-assertion-based.
   The large `qualification`/`moral_route` deltas partly reflect the adapter learning source-discipline
   **format/markers** (the very habit trained) — genuine behavior shift, but **semantic quality needs
   a ≥2-family LLM-judge pass** before any headline. The forbidden-assertion reductions
   (tradition-merge, false-attribution) are the more substantive signal.
2. **GENERAL-CAPABILITY RETENTION — criterion #3 PASSES, now on a POWERED probe (final 2026-06-27).**
   The pre-registered stability check is gemma-3-native: base vs adapter on the **held-out generality
   probe** (`data/generality_tasks.json`, abstraction/arithmetic/logic/analogy/out-of-domain,
   deterministic scoring, no LLM judge — `M3-pilot-retention-eval.json`). The measurement was matured
   in three steps, each more resolved than the last — a case study in *measure properly before you
   conclude*: **N=34 → Δ −0.118** (looked like forgetting, no CI); **N=70 → Δ −0.014, CI [−0.071,+0.043]**
   (overturned it, but the CI lower tail still dipped past −0.05); **N=970 (powered: 70 curated + 900
   programmatically-generated, gold correct by construction) → base 0.489 → adapter 0.488, Δ −0.001,
   95% CI [−0.020, +0.018], `retains:true`.** At N=970 the probe's MDE (~0.06 worst-case, lower under
   the paired structure) finally resolves the 5pt criterion, and the tight CI **rules out forgetting
   beyond ~2pts**. No category meaningfully drops (abstraction 0.73→0.70, logic 0.98→0.98, analogy and
   out-of-domain up). The −0.118 was confirmed pure small-sample noise. Gate (`tools/retention_gate.py`)
   → **GO**. This is the contract's headline lesson: the fix was a better instrument, not a model change.
3. **Single base, single seed, corpus-bound** (~730 deterministic rows; M2 volume is a NO-GO). Needs
   seeds 1–2 for stability and the multi-judge semantic pass before promotion.
4. Train/eval share structural families (decontaminated by exact prompt, not by format) — format-overlap
   caution applies, as flagged previously for the religion channel.

## Corroboration added after the pilot (seed 1 + LLM judges)

**Stability (seeds 1 AND 2, full N=354 × 3 runs each — `M3-pilot-eval-seed{1,2}.json`).** Across THREE seeds the primary signal is CI-clean improving every time: qualification +0.475/+0.371/+0.383, tradition_merge +0.143/+0.113/+0.125, false_attribution +0.014/+0.010/+0.010, moral_route +0.569/+0.588/+0.686 (seed0/1/2); no protected regression on any seed; over-abstention ≤0.023. Three-seed robust. (Original seed-1 detail:) The primary signal
reproduces: adapter(prompt)−base(prompt) is CI-clean improving on the SAME 3 metrics — qualification
+0.372, tradition_merge +0.113, false_attribution +0.010 (citation_fidelity again not CI-clean);
moral_route +0.588. No protected regression (history 0.074→0.028), over-abstention 0.023. Two seeds
agree.

**Independent 3-family semantic judges (`M3-pilot-judge.json`).** To test whether the marker gains
hold up SEMANTICALLY (vs learned format), three judge families distinct from the gemma subject AND from
`agent/gate.py` re-scored 268 source-family cases blind (randomized A/B) — `tools/judge_pilot_answers.py`:
- deepseek-chat: adapter preferred **94.4%** · mistral-small-3.2-24b: **82.1%** · llama-3.3-70b: **67.9%**.
- Majority-vote adapter win-rate **0.832**; **UNANIMOUS (all 3 agree): adapter 169 vs base 5**.
- **All three independent families prefer the adapter** → the source-discipline gains are semantically
  real, not just keyword emission. This substantially addresses caveat (1).

**Agreement statistics (the honest nuance).** Pairwise Cohen's **κ = 0.12–0.38 (< 0.40)** — but κ is
deflated by *prevalence skew* (all judges agree the adapter is better, inflating chance agreement: the
well-known κ paradox). The prevalence-robust **Gwet's AC1 = 0.68–0.79 across all pairs (substantial,
well above 0.40)**, and raw agreement is 73–81%. So inter-judge agreement is genuinely substantial by
the appropriate statistic; the literal **pre-registered κ≥0.40 bar is still NOT met**, so this remains
**strong corroboration, NOT a formally validated semantic claim** (we do not move the goalposts — κ is
reported failing; AC1 is the documented robustness read given the skew).

## Next steps before any promotion
ALL pre-registered follow-ups are now DONE: seeds 1–2 stability ✓ (3-seed robust) · 3-judge-family
semantic re-score ✓ (adapter preferred, AC1 substantial, κ<0.40 reported failing) · retention check ✓
(**criterion #3 FAILS — ~12pt general-capability forgetting**) · M4 ORPO ✓ (NO-GO: from-base coin-flip,
on-SFT does not beat SFT). **Net: candidate stays `candidate_only` and is NOT promotable** — the
forgetting makes it unfit as a general model, and no external/market/AGI claim is made (`canClaimAGI:
false`). A real promotion would need a forgetting-mitigated recipe (e.g. replay/rehearsal of the
general slice, lower-rank or fewer-epoch SFT, or a KL/anchor penalty) that keeps the source-discipline
gains WITHOUT the general-reasoning drop.
