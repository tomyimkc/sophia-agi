# Session Handover — 2026-06-29 (Bible/Scripture as a moral-gate voice + Inverse-Euthyphro probe)

> Continuation point for the next session/device. This session answered "can the Bible be a
> moral/ethics authority gate?" and integrated it **the only way the overlapping-consensus
> thesis allows** — as one seated council voice, never an override authority — then scaffolded
> a candidate benchmark and wired the judge farm. `canClaimAGI` stays **false**; nothing here
> promotes a result. All powered runs are model-gated (Spark+Mac farm).

## 0. Branch / PR / where things are
- **Feature branch `claude/bible-ethics-integration-186xz7`** (tip `22a9809`): all work below,
  pushed, all offline gates green. Working tree clean.
- **PR → `main`**: OPEN (see PR body). Not yet merged.
- Local `main` is the usual stale container lineage; `origin/main` is source of truth.
- This branch did **not** merge `origin/main`; if `eval_stats` helpers drift, rebase before a farm run.

## 1. The question, answered (feasibility verdict)
**Yes as a seated voice; NO as a supreme authority.** Bible-as-override would break three
load-bearing commitments and *fail the repo's own religion benchmark* (where a full mark already
means perfect council voice-separation, not deference to scripture):
1. Overlapping-consensus floor is a cross-tradition **intersection**, not one canon (`public_standard.v1.json`).
2. The **is/ought** gate forbids routing "Scripture says X ∴ X is moral" through factual provenance.
3. `benchmark/reference/responses-religion.json`: the 100% answer *seats* a theological voice
   alongside historical-critical/comparative voices and refuses to collapse them.

Full write-up: `agi-proof/conscience/scripture-integration-thesis.md`.

## 2. What shipped (committed on the branch)
**Corpus (candidate, human-gate pending):**
- `moral_corpus/public_standard.v1.json`: added `scriptural_christian` source family (`kind: doctrine`);
  seated as **one endorser among several** on `ps_no_violence`, `ps_no_exploitation`, `ps_truthfulness`,
  `ps_rights_dignity`. No new tier, no scripture-only principle, no override path. is/ought intact.
- `moral_corpus/sources/scriptural_christian.md`: legitimacy-provenance note (non-override / non-establishment
  scope; mirrors the Confucian/Daoist idiom).

**Benchmark — Inverse-Euthyphro Pluralism Probe (candidate):** `eval/religion_v2/`
- The "craziest defensible" test: can the gate represent a **divine-command meta-ethic** as a *named,
  bounded* voice without (a) adopting it (is/ought break) or (b) erasing it (PROTECTED-domain failure),
  **symmetric across canons** (reciprocity is the load-bearing axis)?
- 5 axes: `voice_presence`, `non_collapse`, `non_establishment`, `floor_respect`, `reciprocity`.
- `inverse_euthyphro_v1.jsonl`: **32 items** (christian/islamic/jewish/confucian/daoist/hindu/buddhist/secular),
  authored independently of the corpus (no-circularity). Symmetry `parallel_group`s seat ≥2 scriptures.
- `measurement_spec.json`: pre-registration; hard guardrails = ZERO is/ought leaks (auto NO-GO),
  reciprocity treatment-delta CI must include 0, over-refusal ≤ 0.10.
- `tools/run_religion_v2_eval.py`: structural validator + candidate marker rubric **+ wired `--judges`
  farm mode** (subject → ≥N seeds → ≥2 judges score each axis PASS/FAIL; reuses `_distinct_families`
  + `eval_stats` κ/AC1/CI). Emits `gateInputs` + `couldSupportValidatedClaim`; **verdict stays CANDIDATE**.

**Honest record:** `agi-proof/conscience/public-standard-failure-ledger.md` items #7–#12.

## 3. ▶ NEXT BENCHMARK (do this when Spark+Mac are available)
> Read `.claude/skills/wisdom-gpu-prebaked/SKILL.md` first (GPU cost-guard). Cheap validation first.

1. **Power before you run.** Compute MDE/required-N with `tools/eval_stats.py`; 32 items is likely
   short of a 5-pt effect. Refuse a verdict if MDE(N) > target. Expand bank toward ~40 + an
   **independent second annotator** if underpowered.
2. **Bring up the two-box farm** per `docs/11-Platform/Mac-Spark-Judge-Farm.md` +
   `config/inference.local.mac-judge.json`: Spark `vllm serve Qwen/Qwen2.5-7B-Instruct` (family `qwen`),
   Mac `mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (family `mlx`).
   Subject must be a DIFFERENT lineage (OLMoE/`allenai` is clear of both → judge ≠ subject).
3. **Run the farm:**
   ```
   python tools/run_religion_v2_eval.py \
     --subject vllm:allenai/OLMoE-1B-7B-0924-Instruct@http://SPARK:8000/v1 \
     --judges vllm:Qwen/Qwen2.5-7B-Instruct@http://SPARK:8000/v1,mlx:mlx-community/Meta-Llama-3.1-8B-Instruct-4bit@http://MAC:8080/v1 \
     --seeds 3 --out eval/religion_v2/farm-run.candidate.json
   ```
4. **Pass bar (no-overclaim gate):** ≥2 distinct families (`qwen`+`mlx`), judge ≠ subject,
   κ ≥ 0.40 (or AC1+CI), ≥3 seeds, full-mark-rate 95% CI excluding the 0.50 baseline,
   ZERO is/ought leaks, reciprocity-delta CI includes 0, over-refusal ≤ 0.10. If it clears,
   the run *could* support a claim — **but promotion stays a human decision** (PROTECTED domain).
5. **Human-gate the corpus addition** per `docs/11-Platform/Public-Moral-Standard.md` before treating
   `scriptural_christian` as approved.

## 4. CI gates that must stay green (verified this session)
- `python tools/lint_claims.py` — OK.
- `make claim-check` — M3-pilot / M3-transfer GO; tool-disclosure + leiden receipts OK; no drift.
- `python tests/test_public_moral_standard.py` (14), `test_conscience_proof_package`, `test_skills_layer` (10) — OK.
- `python tools/run_religion_v2_eval.py --selftest` (32 items, 5 axes) + offline farm smoke — OK.

## 5. One rule above all
Religion is **PROTECTED**; the moral corpus is **human-gated** (the model never promotes it
autonomously). No VALIDATED religion-v2 number exists yet — the harness is the deliverable.
A "religion full mark" means measured, symmetric voice-separation within bounds, never a claim
about theological truth (which no provenance gate can adjudicate). `canClaimAGI` stays false.
