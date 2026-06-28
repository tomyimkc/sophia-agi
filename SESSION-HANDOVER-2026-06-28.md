# Sophia-AGI — Master Handover Prompt (2026-06-28)

> **Purpose.** You are the next AI session picking up `tomyimkc/sophia-agi` on a new
> device (DGX Spark + Mac Studio available). This document is your single-source
> briefing: what the repo is, what the last session proved, the exact state of the
> evidence, and the **next benchmark to run**, with commands and pass/fail thresholds.
> Read the "Read first" file list before acting. Do not overclaim — this repo lives or
> dies by its measurement contract.

---

## 0. Git / repo state at handover

- Branch worked on this session: `claude/session-handover-repo-analysis-pp2e6u`.
- **`HEAD == origin/main == origin/branch == 2009230`.** All work — including the last
  three commits #234/#235/#236 — is already on **remote `main`**. There is nothing to push.
- Caveat: the *local* `main` ref (`65da808`) is a stale, divergent lineage from the
  container clone (231 unrelated commits); ignore it or hard-reset it to `origin/main`.
  The remote is the source of truth and is current.
- Working tree clean.

---

## 1. What Sophia-AGI is (thesis)

Sophia ("the Wisdom Gate") is an **open, provenance-aware, verifier-gated reasoning
layer that abstains instead of fabricating**. Core deliverable: a *fail-closed gate* —
`claim → verify against sources → accept · abstain · block` — that stops LLMs from
inventing attributions or merging distinct intellectual traditions, then reasoning on
top of those errors. The project **explicitly disclaims AGI** (`canClaimAGI: false`
everywhere). Honest deliverables are: the machinery (verifiers, abstaining gate, belief
graph, governance contract), measured data under a strict **no-overclaim gate**, and a
first-class public **failure ledger** of what is *not* yet proven. A secondary
"capability arc" adds the *more-capable* half of a trustworthy agent (long-horizon
execution, retrieval/AI-search, continual learning, systems/cluster engineering) — always
in Sophia's idiom: deterministic, offline-testable, auditable.

**The no-overclaim gate (memorize this):** a number is **VALIDATED** only with
**≥2 independent judge families** in consensus (judge ≠ subject) **+** reported
inter-judge agreement (Cohen's κ ≥ 0.40, or a CI excluding chance) **+ ≥3 runs/seeds
+ confidence intervals + not-mock**. Everything else is **candidate** or **illustrative**
and must be labelled so. `agi-proof/benchmark-results/published-results.json` is the
single source of truth; `RESULTS.md` is generated from it (never hand-edit).

---

## 2. The methodology this repo is trying to prove

Beyond any single model, the repo's real thesis is the **Instrumented Evaluation
Contract (IEC)** / measurement discipline (`agi-proof/measurement-thesis.md`): *in a
small-corpus AGI-candidate pipeline the dominant source of wrong conclusions is the
measurement instrument, not the model* — so evaluation is engineered as a first-class,
fail-closed instrument. Eight pillars, each a deterministic CI check:

1. Always quantify uncertainty (CI, never a bare point estimate).
2. Power before you run — pre-register MDE + required N; refuse a verdict when MDE(N) > effect.
3. Curate items by discrimination.
4. Anytime-valid inference (confidence sequences) because the workflow peeks/iterates.
5. Triangulate ≥2 independent constructs (deterministic markers + LLM-judge panel + behavioral transfer).
6. Content-level decontamination + private split.
7. Calibrate/de-bias the judge.
8. Effect size **and** pre-registered practical magnitude.

Enforced by `tools/claim_gate.py` (the GO/NO-GO gate), `tools/eval_stats.py`
(power/MDE, bootstrap CI, Robbins anytime-valid CS), `tools/assert_decontam.py`,
`tools/lint_training_rows.py`, `tools/lint_claims.py`; pre-registered per experiment in
`measurement_spec.json` with a git-ancestry `--assert-prereg` check. Hard claim ceiling
on every artifact: `candidate_only; canClaimAGI:false; narrow corpus-bound feasibility`.

**This is the "methodology on the right track to prove it works" the operator referred
to.** The flagship demonstration: a scary `−0.118` "catastrophic forgetting" signal was
shown to be a *measurement artifact* — fixing the instrument (N 34→70→**970**), not the
model, flipped it to a clean null (Δ −0.001, both fixed-n CI and anytime-valid CS clear).

---

## 3. The last session (commits #234 → #235 → #236): Sophia-Wisdom-4B

**Goal.** Take a 4B LoRA "source-discipline" adapter (Sophia-Wisdom-4B) from scattered
pilot runs to a fully measurement-contract-gated, market-reality-checked, calibration-
fixed result — and stand up the **local hardware path** (DGX Spark + Mac Studio) to
certify it cheaply.

- **#234 `c8d45f2`** — Two-box (Spark vLLM + Mac MLX) local judge farm for the ≥2-family
  certification gate. Config + docs + smoke test. *Wired, not yet executed.*
- **#235 `d13ee7d`** — Wisdom-4B measurement-contract hardening + market reality-check +
  calibration fix. Large pilot-data drop (M3 pilot/stable/transfer/retention, 6 seeds).
- **#236 `2009230`** — `tools/certify_lowram.py`: 16-bit-vs-NVFP4 low-RAM certification
  for the Spark (Boundary-3). *Code + offline self-test shipped; real GPU pass not run.*

### What the adapter result currently is (gate verdicts)

Pre-registration: `agi-proof/benchmark-results/wisdom-market/measurement_spec.json`
(primary N=354×3, **primaryMDE 0.105**; retention guardrail N=70, MDE 0.237 — GO/NO-GO
only, never rank).

| Recipe | Verdict | primaryMag | judge winrate | retention | constructs |
|---|---|---|---|---|---|
| M3-pilot (SFT rank16) | **GO** | 1.214 | 0.815 | Δ −0.001, **N=970** (CS[−0.030,+0.028]) | 3 (marker+judge+behavioral) |
| M3-stable (rank8+KL+replay) | **GO** | 0.987 | 0.787 | Δ +0.014, N=70 (underpowered, non-critical) | 3 |
| M3-transfer (160 novel entities) | **GO** | judge 0.696 | — | — | 2 |
| M4-orpo-sft (ORPO on SFT) | **GO** | 0.955 | 0.767 | — | 2 |
| **M4-orpo (from base)** | **NO-GO** | 0.054 | 0.529 (~coin) | — | fails magnitude |

Best recipe = **M3-SFT rank16 baseline** (simplest wins; ORPO does *not* beat SFT — a
clean honest negative, gated NO-GO). Market reality-check (`reality-check.json`, N=1062):
adapter qualification-on-contested 0.978 vs Grok-4.3 / DeepSeek-V3.1 / Mistral-large
0.38–0.42 raw — but with the same scaffold the gap narrows (Mistral 0.41→0.79) and a
3-family blind semantic judge prefers the adapter only ~0.65. **Honest headline: a modest,
genuine, scaffold-independent narrow edge — not "4B beats frontier."** Calibration tax
(over-hedging settled facts) was **fixed in v3** (17.1% settled rows; generalizes to
novel entities), per `calibration-fix-result.json`.

### Proven vs still open

**Proven (within the ceiling):** source-discipline is *learnable* by a 4B model; it is a
*transferable habit* (transfer GO, 160 novel entities); *no catastrophic forgetting* on a
properly powered probe (N=970); *SFT > ORPO* (honest negative); *modest scaffold-
independent edge* over 3 strong models; *calibration tax fixed*.

**Still open / honest limits:** mostly **single-seed** per recipe (only M3-SFT is 3-seed;
retention single-seed); retention N=70 underpowered for a 5-pt threshold; judge Cohen
κ < 0.40 (prevalence paradox; reported via Gwet AC1 — **not formally validated**, no human
gold anchor); **no first-party frontier comparison** (GPT/Claude/Gemini egress-blocked);
private held-out split still pending; **the low-RAM NVFP4 certification and the local
≥2-family judge-farm run have not actually executed** — code/wiring shipped, no measured
artifact yet.

---

## 4. ▶ NEXT BENCHMARK (do this — hardware is now available)

Both the **DGX Spark** and **Mac Studio** are on the receiving device, which unlocks
exactly the two wired-but-unrun pieces from last session. Run them in this order; each
closes a specific named gap.

> **Before any GPU run, read `.claude/skills/wisdom-gpu-prebaked/SKILL.md`** — the
> anti-wastage runbook (three documented RunPod credit-burn incidents). Cheap validation
> first (`limit=24, runs=1`), watch the first ~6 min for restart loops, and always finish
> by confirming zero leaked pods.

### Benchmark A (headline, highest certification payoff) — Two-box ≥2-family VALIDATED run

Convert the M3-SFT source-discipline result from **candidate/markers** to **VALIDATED**
(headline grade) — locally and free, no metered cloud.

1. Bring up the judge farm per `docs/11-Platform/Mac-Spark-Judge-Farm.md` +
   `config/inference.local.mac-judge.json`:
   - **Spark:** `vllm serve Qwen/Qwen2.5-7B-Instruct` → family `qwen`.
   - **Mac:** `mlx_lm.server --model Meta-Llama-3.1-8B-Instruct-4bit` → family `mlx`.
   - Sanity: `python tools/run_local_judge_eval.py --config config/inference.local.mac-judge.json`
2. Judge the M3-SFT pilot answers across **≥3 seeds** with both families, then run the
   gate aggregator:
   - `python tools/judge_pilot_answers.py --judges vllm:Qwen/Qwen2.5-7B-Instruct@spark,mlx:Meta-Llama-3.1-8B-Instruct-4bit@mac …`
   - `python tools/run_lora_uplift_validation.py …`
3. **Pass bar (the no-overclaim gate):** ≥2 distinct families (`qwen`+`mlx`), judge ≠
   subject (subject lineage `allenai/OLMoE` — clear of both), κ ≥ 0.40 (or AC1 + CI),
   **≥3 seeds**, uplift 95% CI excluding zero. If it clears, promote the row in
   `published-results.json` and regenerate `RESULTS.md` via `tools/build_results_page.py`.

*Proves:* the source-discipline content uplift clears the full VALIDATED bar — the single
biggest open caveat on the Wisdom-4B claim.

### Benchmark B — Low-RAM NVFP4 certification on the Spark (Boundary-3)

First real "low-RAM, capability-retained" evidence. After a QAT train
(`tools/train_lora.py --qat`, default base `allenai/OLMoE-1B-7B-0924-Instruct`, adapter
`training/lora/checkpoints/olmoe-qat-spark`):

```
python tools/certify_lowram.py \
  --base-model allenai/OLMoE-1B-7B-0924-Instruct \
  --adapter training/lora/checkpoints/olmoe-qat-spark \
  --scheme nvfp4
```

**Pass bar** (`serving/lowram_eval.LowRamGate`): BF16-vs-NVFP4 next-token distribution
**mean KL ≤ 0.05, top-1 agreement ≥ 0.97** (protected slice: KL ≤ 0.10, agreement ≥ 0.95).
On pass, write the `certify-lowram*.json` artifact (none exists yet) per
`docs/11-Platform/Cheap-Compute-Boundary.md` — and remember an NVFP4 pass may *only*
claim "served-quant retains BF16 next-token behavior to a measured bound," nothing more.

### Then (close the residual single-seed caveats)

- **Retention/transfer at a 2nd & 3rd seed:** `…runpod_wisdom_pilot_selfreport.py
  --mode retention` / `--mode transfer` at seeds 1,2 → harden "no forgetting" (N=970) and
  "transferable habit" past their current single-seed bounds; keeps
  `claim_gate --prefix M3-transfer` GO with real power.
- **External independence upgrade (SimpleQA Verified):** `python
  tools/run_simpleqa_calibration.py` → `tools/fit_conformal_policy.py` — re-test the
  abstention/calibration habit on 1000 human-authored, non-self-authored prompts (the
  method note's flagged gap; mirrors how C1 calibration already reached VALIDATED).

---

## 5. CI gates that must stay green (do not break these)

- **fast-ci.yml** (every PR): `compileall`; `lint_claims.py`; `validate_attribution.py`;
  the measurement contract = `eval_stats.py` + `lint_training_rows.py` +
  `assert_decontam.py` + **`claim_gate.py --prefix M3-pilot`** + **`--prefix M3-transfer`**
  (both must stay **GO**); core unit tests. Local equivalent: `make claim-check`.
- **ci.yml** artifact-drift gates: `tools/build_results_page.py --check`,
  `tools/build_rag_index.py --verify`, `tools/wiki_sync.py check`,
  `tools/build_local_sophia_dataset.py --check`, `tools/check_version_consistency.py`,
  `validate_failure_ledger.py --check`, full `pytest -q` (2200+ tests).

---

## 6. Read first (in this order)

1. `agi-proof/measurement-thesis.md` + `agi-proof/preregistered-thresholds.md` — the IEC, claim levels 0–5.
2. `agi-proof/sophia-wisdom-4b-method-note.md` — the model result inside its honest bounds.
3. `agi-proof/benchmark-results/wisdom-market/` — `measurement_spec.json`,
   `recipe-benchmark.{json,gate.json}`, the `M3-*/M4-*.gate.json` verdicts,
   `reality-check.json`, `calibration-fix-result.json`.
4. `tools/claim_gate.py`, `tools/eval_stats.py` — the gate + statistics engine.
5. `.claude/skills/wisdom-gpu-prebaked/SKILL.md` — mandatory GPU cost-guard runbook.
6. `config/inference.local.mac-judge.json` + `docs/11-Platform/Mac-Spark-Judge-Farm.md`
   + `tools/run_local_judge_eval.py` — the two-box judge farm.
7. `tools/certify_lowram.py` + `serving/lowram_eval.py` + `docs/11-Platform/Cheap-Compute-Boundary.md` — low-RAM cert.
8. `agi-proof/benchmark-results/published-results.json` + `RESULTS.md` — source of truth + gate definition.
9. `agi-proof/failure-ledger.md` + `agi-proof/evidence-manifest.json` — what is *not* proven (58 open items; the headline AGI blockers).
10. `README.md` (top) + `VISION.md` + `CONTRACT.md` — thesis + scope disclaimer.

---

## 7. Project map (orientation)

- `agent/` — core: model presets, gate/grounded_gate, verifiers, calibration, graded_decision, AI-search, long-horizon engine, subagents, continual learning, security firewall.
- `okf/` — belief/provenance graph (confidence propagation, retraction, contradiction ledger).
- `agi-proof/` — governance + evidence layer (published-results, failure-ledger, pre-registration, measurement thesis, wisdom-market results).
- `provenance_bench/` — provenance-delta + calibration scoring harness, judged faithfulness.
- `eval/` + `benchmark/` — benchmark lanes + datasets (legal, agent-faithfulness, conscience, belief-revision…).
- `training/` + `pretraining/` — SFT/curriculum/LoRA adapters + DeepSeek-style pretraining-alignment + data-engineering.
- `serving/` — low-RAM serving (AirLLM-style layer streaming) + the LowRamGate.
- `storage/` — Rust systems components (sharded async kvcache, LSM/WAL, miniraft).
- `sophia_contract/` + `sophia_mcp/` — stable governance contract + fail-closed MCP gateway.
- `tools/` — every runner/gate/launcher (claim_gate, eval_stats, certify_lowram, runpod_*, build_results_page…).

**Governance status:** the HANDOVER.md §4 architecture-bets schema collision is
**RESOLVED** (split into `agi-proof/architecture-bets.json` module-wiring registry +
`agi-proof/long-context-bets.json` measurement targets). Branch cleanup effectively done
(2 remote branches). 58 failure-ledger items remain OPEN — the rule is: if OPEN hits 0,
*upgrade the public wording*, never silently relax the gate.

---

## 8. One rule above all

This repo's credibility *is* its discipline. Never report a number without its CI, seeds,
judge families, and the candidate/validated label. When in doubt, label it **candidate**
and add a failure-ledger entry. `canClaimAGI` stays `false` until a third-party hidden
eval is beaten — that is by design, not a gap to paper over.
