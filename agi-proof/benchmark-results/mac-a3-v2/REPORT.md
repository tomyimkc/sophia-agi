# Mac Studio operator — A3 philosophy council teacher v2 (decontaminated pack, halved iters)

**Date:** 2026-07-02 · **Box:** Mac Studio (M3 Ultra, MLX) · from fresh `origin/main` (c31cc162)
**Guardrails:** `canClaimAGI=false`, `candidateOnly=true`; nothing promoted (`promote_adapter.py` decides); `lint_claims` clean; never `main`; no Spark GPU / no w1–w5; **no self-hosted runner registered** (owner-gated).

## HEADLINE — two findings
1. **CONTAMINATION caught + fixed.** The v1 philosophy stage-1 pack on main had **26 of 528 rows whose prompts are the eval-ladder holdout** (`tests/benchmark-philosophy.json`, 9 cases). `build_teacher_data.py`'s built-in decontam does **not** cover that holdout (its `eval_prompt_set` = `eval/**` + wisdom_market only), so it silently trained on the test. **⇒ the v1 eval (adapter 55.6%) was train-on-test** and is not a valid generalization number. Fixed here with an explicit fail-closed decontam before building.
2. **The SFT recipe regresses this capability even CLEAN.** On the decontaminated pack + halved iters, the v2 adapter scores **44.4% vs 77.8% base (−33.3 pts)** on the true holdout. Val-loss is healthy (no collapse), so this is **structural to the recipe, not overfitting**. **Not promotable.**

## v2 pack (decontaminated, supply-capped)
- **Contamination gap is real:** `build_teacher_data` built-in guard misses the philosophy holdout. Mandatory explicit decontam against `tests/benchmark-philosophy.json` before every build.
- Sources (all decontam'd, 0 holdout overlap across 643 rows): existing stage-1 **502 clean** (528 − 26 contaminated) + **61 NEW gate-clean** council-distilled traps (70B teacher over `data/attributions.json` via `selfplay_task_forge` → `distill_council_traces`; `gatePassed:true`, 3 dropped-dirty by the gate). Stage-1 total **563** (507 train / 56 valid); stage-2 80 (72/8).
- **Supply cap (honest):** the legitimate grounded prompt supply is **~64 unique** (the forge over the current attributions graph; multi-seed adds no diversity). The cloud's ≥1500 target is **not reachable** without expanding `data/attributions.json` or a vetted external prompt bank. **No prompts were fabricated to pad** (that would recreate the memorization failure). Net new grounded rows: **+61**.

## v2 training (halved iters — the memorization lever)
`train_council_teacher.py --iters1 150 --iters2 250` (vs v1 300/500), Qwen2.5-3B-Instruct LoRA (6.65M params).
Val-loss curve (`val-loss-curve.txt`): stage-1 **2.416 → 1.396**; stage-2 **2.098 → 1.493 → 1.525** (slight tail uptick). Unlike v1 (train-loss→0 collapse), the loss is real — **halved iters fixed the memorization but not the regression**.

## v2 eval ladder (CLEAN holdout, content channel, 9 philosophy cases)
base 7/9 (77.8%) · base+gate 7/9 (77.8%) · **adapter 4/9 (44.4%)** · adapter+gate 4/9 (44.4%). **Δ(adapter−base) = −33.3 pts.**
`eval-ladder-baseline.json`, `eval-ladder-adapter-v2.json`.

## Verdict + recommendations (for the cloud)
- **CANDIDATE, NOT promoted.** Clean measurement confirms the SFT teacher recipe *degrades* base philosophy-attribution accuracy; more data + fewer iters did not help.
- **(1) Fix main:** the 26 contaminated rows in `training/teachers/philosophy/stage1` on main need a decontam PR; and `build_teacher_data`'s decontam set should include `tests/benchmark-philosophy.json`.
- **(2) Rethink the recipe, not the hyperparams:** the base is already strong here; SFT on reject-attribution traces may conflict with the eval's expected answer format. Consider preference/DPO over SFT, or a capability where the base is weak. Re-running SFT with yet-more-data is not indicated (the regression is clean).
- **(3) Scale needs corpus, not fabrication:** grounded prompts cap at ~64; expand `data/attributions.json` for real scale. `canClaimAGI=false`.
