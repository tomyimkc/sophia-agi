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
**PASS** of the pre-registered M3-pilot primary criterion: a LoRA on the gate-passed data produces a
**CI-clean, non-regressing behavioral shift toward source-discipline habits at the prompt (no-gate)
layer** of gemma-3-4b — the "weights internalize the habit" feasibility signal — at bounded
over-abstention and rising scaffold-conditioned usefulness.

## Honest caveats (why this is NOT a headline / validated claim)
1. **Marker/structural metrics only.** Scores are deterministic marker- and forbidden-assertion-based.
   The large `qualification`/`moral_route` deltas partly reflect the adapter learning source-discipline
   **format/markers** (the very habit trained) — genuine behavior shift, but **semantic quality needs
   a ≥2-family LLM-judge pass** before any headline. The forbidden-assertion reductions
   (tradition-merge, false-attribution) are the more substantive signal.
2. **Retention proxy, not `run_learning_shift.py`.** Pre-registration named that tool; here retention
   is inferred from useful_correctness + over_abstention. Running `run_learning_shift.py` on the
   adapter is the open follow-up.
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
Seeds 1–2 (stability) · ≥2-judge-family semantic re-score of qualification/route quality ·
`run_learning_shift.py` retention check · then M4 (ORPO on the preference pairs). Until then this stays
`candidate_only` and no external/market claim is made.
