# Research Note — "Small model carries a frontier teacher's reasoning" (Gemma-class distillation thesis)

> **Status: research / brainstorm. candidate_only; canClaimAGI:false.** Nothing here is a
> measured result. This note triages an externally-supplied thesis (a Gemma-class 12B
> fine-tune "matching Fable 5 reasoning on 8 GB VRAM") into (a) mechanisms that are real and
> transferable to this repo's training stack, and (b) marketing claims this repo's
> measurement contract forbids us from repeating. Then it maps the transferable parts onto
> concrete files we already have, as proposals to be gated like everything else.

---

## 0. Why this note exists

The operator supplied a thesis describing how a ~12B open model can be made to "carry a
frontier teacher's reasoning" via: (1) an architecture optimised for intelligence-per-byte
(Per-Layer Embeddings, mid-size MoE, QAT), (2) **execution-verified chain-of-thought
distillation from two complementary teachers** (a high-volume teacher plus an auxiliary
teacher that *patches the hard, failed subset*), and (3) aggressive but QAT-friendly
quantisation for low-RAM serving.

The useful observation is that **we already operate most of this machinery** — verifiers,
gated distillation, QAT, low-RAM certification, a MoE-vs-dense probe. The thesis is less a
new idea than a *recombination* of pieces we have, plus one piece we do **not** yet do
cleanly (auxiliary-teacher failure-patching) and one discipline the thesis ignores entirely
(decontaminating teacher traces against the held-out eval).

---

## 1. Thesis triage: real vs. unverifiable

### 1a. Mechanisms that are real and transferable

| Thesis mechanism | Reality | Where we already touch it |
|---|---|---|
| **Execution-verified CoT distillation** (keep only traces whose code passes deterministic tests) | Real, well-established (rejection-sampling / RFT, STaR-style). | `tools/distill_export.py` (epistemic-gate + keyword filter), `agent/code_verifier.py`, `agent/execution_verifiers.py`, `provenance_bench/code_reward.py`. |
| **Outcome verification before inclusion** (reasoning that *leads to correct outcomes*, not fluent nonsense) | Real; this is exactly the RLVR/verifier philosophy. | `agent/verifiers.py` (the pluggable oracle), `provenance_bench/*_reward.py`. |
| **Auxiliary teacher patches the failed subset** ("debug & recover" behaviour) | Real and underused by us. This is the one genuinely new lever. | *Not yet*: `distill_export.py` is single-teacher, no failure-routing tier. |
| **Sample-efficiency from small, high-signal data** | Real; small clean SFT beats large noisy SFT for habit transfer. | `tools/lint_training_rows.py` (habit-not-fact), `pretraining/synthetic_scaling/` (fidelity > volume). |
| **QAT so 4-bit serving degrades gracefully** | Real (Gemma 3 ships QAT checkpoints; the technique is sound). | `training/qat.py`, `tools/certify_lowram.py`, `serving/lowram_eval.LowRamGate` (KL ≤ 0.05, top-1 ≥ 0.97). |
| **MoE: many experts, few active per token → big-model quality at small active compute** | Real (DeepSeek-V3, Mixtral, Qwen-MoE). | `pretraining/architecture/` (top-1 MoE vs dense at matched active compute). |
| **Per-Layer Embeddings (PLE) for cheap representational capacity** | Real — but it is a **Gemma 3n** feature, used to offload embedding params to fast storage on edge devices. Not validated by us at any scale. | *Not yet*: candidate for a nano-scale probe in `pretraining/architecture/`. |

### 1b. Claims our measurement contract forbids us from repeating

These are the parts to **strip before any of this touches public copy** (`README`,
`RESULTS.md`, method notes). They are precisely the genre of overclaim
`tools/lint_claims.py` and `tools/claim_gate.py` exist to stop.

1. **"Gemma 4" / specific numbers.** As of this note the shipped Google line is **Gemma 3 /
   3n**; "Gemma 4", "26B A4B / 128 experts / 3.8B active", and "Composer 2.5 as teacher" are
   unsourced specifics. Treat model names and counts as *illustrative*, not facts. PLE
   belongs to Gemma **3n**, not a "Gemma 4".
2. **"Matches / carries Fable 5-level reasoning."** This is an *unfalsifiable vibe* unless
   it is a measured uplift with the full gate: ≥2 independent judge families (judge ≠
   subject), κ ≥ 0.40 (or AC1 + CI), ≥3 seeds, 95% CI excluding zero, on a decontaminated
   private split. "Feels like the teacher" is exactly the −0.118-forgetting cautionary tale
   in reverse — a number/impression that survives only until the instrument is fixed.
3. **"8 GB VRAM, frontier reasoning."** Two separate claims welded together. The most we
   could ever say is what `LowRamGate` measures: *served-quant retains BF16 next-token
   behaviour to a measured KL bound* — nothing about "frontier reasoning."

**Rule for this note:** every idea below is a **proposal**, gated like any other candidate.
If we build it, the headline it can earn is bounded by the instrument, never by the analogy
to a frontier model.

---

## 2. Brainstorm — creative takeaways mapped onto our stack

Ordered by leverage-per-effort. Each item says what is **new** vs. what already exists, the
**files** it touches, and the **gate** that would validate it.

### T1 — Two-teacher failure-patching tier in `distill_export.py`  ★ highest leverage
The thesis's one real new lever. Today `distill_export.py` is single-teacher: teacher →
gate → keep/reject. Add a **patch tier**:

1. Main teacher (cheap/high-volume, e.g. DeepSeek via OpenRouter) generates CoT+answer.
2. Execution/epistemic gate filters. **Keep passes.**
3. **For the failed subset only**, route to a stronger auxiliary teacher (GLM-5.2 / Claude),
   request a fresh higher-effort CoT, **re-verify**, keep only re-verified passes — tagged
   `provenance: patched`.
4. Passed-first-try → SFT; patched → SFT **and** mined as DPO/`train_dpo.py` pairs
   (failed-main = rejected, patched = chosen) so the student learns *debug-and-recover*, not
   just correct answers.

*New:* the failure-routing tier + per-row teacher/effort tagging. *Reuses:* the entire
verifier library as the gate. *Gate:* cross-model held-out uplift via `eval_rlvr_adapter.py`
→ `claim_gate.py`.

### T2 — Unify the verifier as a single "outcome oracle" for RL *and* distillation
We maintain reward verifiers (`provenance_bench/*_reward.py`) and distillation filters
(`distill_export.py`) as separate code paths that encode the same judgement. Collapse them:
one `agent/verifiers.py` oracle that both (a) scores GRPO rollouts in `run_rlvr.py` and (b)
gates SFT rows in `distill_export.py`. Single source of truth for "is this output correct,"
so the RL reward and the SFT filter can never silently disagree.

*New:* a thin adapter so `run_rlvr.py` and `distill_export.py` import the same verifier
objects. *Gate:* unit tests that the same case scores identically through both paths.

### T3 — On-policy self-distillation: passed RLVR rollouts → free SFT rows
Direct consequence of T2. Every GRPO rollout in `run_rlvr.py` that **passes the verifier** is
already a verified (prompt, CoT, answer) triple — i.e. a free, on-policy SFT row. Add a
`--harvest-sft` flag that writes passing rollouts to a replay JSONL, fed back as next-round
SFT seed. This is the thesis's "verified successes" loop **without paying a teacher** — the
student distills from its own best, verified behaviour.

*New:* a replay-buffer writer in `run_rlvr.py` + a `feedback_to_training.py`-style adapter.
*Risk/guard:* on-policy rows must pass `assert_decontam.py` and `lint_training_rows.py` like
any other row (see T8).

### T4 — Decontaminate teacher traces against the held-out eval  ★ the discipline the thesis ignores
The thesis never addresses that a frontier teacher may have **memorised public benchmarks**,
so its "verified" CoT can leak eval content into training. We already guard train↔eval
(`assert_decontam.py`, `assert_decontam_training.py`). Extend `TRAIN_GLOBS` to cover
**distilled outputs** and run the exact + shingle + entity-disjoint scan on every teacher
trace before it can become SFT data. This is a genuine differentiator: our distillation
would be *decontamination-gated*, which the source thesis is silent on.

*New:* register distill output paths in `assert_decontam.py:TRAIN_GLOBS`; add a `distill`
lane to CI. *Gate:* `assert_decontam.py` exit 0 on the distilled corpus.

### T5 — Make "intelligence-per-parameter / per-byte" a *measured* axis, not a slogan
The thesis's core marketing phrase is "reasoning density per active param / per byte." This
repo's edge is that we can turn that into a gated number. Add an **efficiency-frontier**
report to the eval harness: `score / active-params` and `score / served-GB` (post-quant),
each with CIs, plotted across our adapters. We already have the inputs: capability scores
(`eval_rlvr_adapter.py`, `eval_matrix/`) and served footprint (`certify_lowram.py`).

*New:* `tools/build_efficiency_frontier.py` consuming existing artifacts. *Gate:* numbers
carry CIs and the candidate/validated label; "more capable per byte" becomes a claim only
when the CI is clean.

### T6 — QAT-aware distillation co-design (train against the serving gate)
We do QAT (`training/qat.py`) and we certify low-RAM (`LowRamGate`: KL ≤ 0.05) **separately**.
Co-design them: add a **quantisation-consistency auxiliary loss** during SFT/distillation —
penalise divergence between the BF16 student and its simulated-4bit self's next-token
distribution. The student is then *trained directly against the metric the serving gate
measures*, so "graceful 4-bit degradation" is an objective, not a hope.

*New:* a KL-to-quantised-self term in `train_lora.py` (behind `--qat-consistency`). *Gate:*
the existing `certify_lowram.py` KL/top-1 bars — co-designed, not just checked after the fact.

### T7 — Verified self-consistency on the hard subset (replace "higher-effort teacher")
A cheaper variant of T1's patch tier when a stronger teacher isn't available: for failed
items, sample **N CoTs at higher temperature from the same teacher**, keep only those whose
**executions agree / pass tests** (verified majority vote), distill the agreement. This is
"higher-effort CoT" operationalised as verified self-consistency, reusing
`build_rlvr_judge_answers.py`.

*New:* an N-sample-then-verify mode in the distill path. *Gate:* same execution verifier.

### T8 — Verification-provenance weighting in the dataset builder
The thesis's "small but extremely high-signal" claim, made measurable. Tag every training
row with **how it was verified** (`passed_first_try` vs `patched_after_failure` vs
`self_consistent`) and let `build_local_sophia_dataset.py` **oversample the hard-patched
rows** (they carry the debug-and-recover signal). Extend `lint_training_rows.py` to require
the provenance tag, so we can later *measure* whether patched rows actually move the needle.

*New:* a `verification_provenance` field + sampler weights. *Gate:* `lint_training_rows.py`
enforces the tag; an ablation (patched-included vs excluded) gated by `claim_gate.py`.

### T9 — Cost-per-verified-row as a first-class distillation objective
Teacher calls cost money; the thesis's "sample-efficient" is really "verified-rows-per-
dollar." Have `distill_export.py` emit `cost_per_verified_row` and the main-vs-patch teacher
split, and let `pretraining/autopilot/` (which already enforces `--cost-ceiling`) optimise
the teacher mix. Turns "sample efficiency" into a number we actually budget against.

*New:* cost accounting in the distill loop. *Gate:* none needed — it's an operational metric,
reported not claimed.

### T10 — Nano-scale PLE / MoE architecture probe (falsifiable version of the arch claims)
We cannot validate "26B A4B beats a dense 27B." We **can**, on the `pretraining/architecture/`
toy model with a *known closed-form loss floor*, run the matched-active-compute comparison the
thesis asserts: (a) top-k MoE vs dense (already there), extended with (b) a **Per-Layer-
Embedding** variant, and report the per-active-param frontier honestly with the same
extrapolation gate the scaling study uses (~3% error bound). This is the only intellectually
honest way to touch the architecture half of the thesis.

*New:* a PLE variant in the architecture probe. *Gate:* the existing scaling/extrapolation
gate; identifiability limits reported (as the free-floor collapse already is).

### T11 — Use this thesis as the worked example in `measurement-thesis.md`
Meta-takeaway. "A 12B fine-tune that *feels like* a frontier model" is the perfect teaching
case for the Instrumented Evaluation Contract: show exactly how the claim collapses to a
gated uplift (judge families, seeds, decontam, private split) — the mirror image of the
−0.118 forgetting artifact. Strengthens the repo's central argument at zero training cost.

---

## 3. Recommended sequencing

1. **T4 + T8** first (cheap, pure discipline): decontaminate distilled traces and tag
   verification provenance — these protect everything downstream and are CI-only.
2. **T2 → T3** (unify the oracle, then harvest passing RLVR rollouts to SFT): largest
   capability-per-dollar gain, no extra teacher spend.
3. **T1** (two-teacher patch tier): the headline new lever; do it once T2/T4 make it safe.
4. **T6 + T5** (QAT-consistency loss + efficiency frontier): convert the "per-byte" story
   into co-designed training and a gated metric.
5. **T10 + T11** (architecture probe + measurement worked-example): the honest, falsifiable
   treatment of the unverifiable half.

## 4. Hard guardrails carried over

- No teacher trace enters training without passing `assert_decontam.py` (T4).
- No row enters training without `lint_training_rows.py` (habit-not-fact) passing (T8).
- No uplift is "validated" without `claim_gate.py` GO: ≥2 judge families, ≥3 seeds, clean CI.
- The most a low-RAM result may claim is the `LowRamGate` KL/top-1 bound — never "frontier
  reasoning on 8 GB."
- Model names/counts from the source thesis stay **illustrative** until first-party measured.

## 5. Implementation status (2026-06-29)

All 11 takeaways are now **built as MECHANISM** (offline-tested, gated), not measured. No
capability uplift is asserted — each is tracked OPEN in the failure ledger until a gated run.

| Takeaway | Shipped in | Offline test |
|---|---|---|
| T1 patch tier + DPO mining | `tools/distill_export.py` (`--patch-provider`) | `tests/test_distill_patch_tier.py` |
| T2 single outcome oracle | `agent/outcome_oracle.py` | `tests/test_outcome_oracle.py` |
| T3 RLVR rollout harvest | `tools/run_rlvr.py` (`--harvest-sft`) | `tests/test_rlvr_harvest.py` |
| T4 distill decontam | `tools/assert_decontam.py`, `tools/distill_export.py` | `tests/test_distill_decontam_provenance.py` |
| T5 efficiency frontier | `tools/build_efficiency_frontier.py` | `tests/test_efficiency_frontier.py` |
| T6 QAT-consistency KL | `training/qat.py`, `tools/train_lora.py` (`--qat-consistency`) | `tests/test_track_b.py` |
| T7 self-consistency | `tools/distill_export.py` (`--self-consistency-n`) | `tests/test_distill_patch_tier.py` |
| T8 provenance tag + oversample | `tools/lint_training_rows.py`, `tools/build_local_sophia_dataset.py` | `tests/test_distill_decontam_provenance.py` |
| T9 cost / teacher split | `tools/distill_export.py` summary | (same) |
| T10 nano PLE probe | `pretraining/architecture/ple.py` | `tests/test_ple_architecture.py` |
| T11 worked example | `agi-proof/measurement-thesis.md` | `lint_claims` |

**Pre-registered GPU/teacher experiments** (the two follow-ups), gated and runnable:

- **Patch-tier yield** — `agi-proof/benchmark-results/distill-patch-tier/measurement_spec.json`.
  Run: `OPENROUTER_API_KEY=… tools/distill_export.py prompts.json --provider deepseek --patch-provider glm:glm-5.2`
  → decontam (`assert_decontam` + `lint_training_rows`) → train → `eval_rlvr_adapter`
  → `claim_gate --prefix distill-patch --assert-prereg`.
- **QAT-consistency low-RAM bound** — `agi-proof/benchmark-results/qat-consistency/measurement_spec.json`.
  Run via `.github/workflows/train-runpod.yml` with `extra_train_args = "--qat --qat-consistency --qat-scheme nvfp4"`,
  then `tools/certify_lowram.py --scheme nvfp4`; compare mean_kl/top1 vs v3.

**Ledger (OPEN until gated):** `distill-recovery-levers-built-2026-06-29`,
`qat-consistency-kl-lever-built-2026-06-29`, `ple-nano-probe-negative-efficiency-frontier-built-2026-06-29`.

*End of note. Mechanisms shipped + pre-registered; no measured capability result is asserted
(`candidate_only; canClaimAGI:false`).*
