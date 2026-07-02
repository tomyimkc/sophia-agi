# T1 — committed training-data audit vs benchmark holdouts (2026-07-02)

**Operator:** Mac Studio · from `origin/main` @ `a7a85515` · **audit only — NO data edited** (per T1).
`candidateOnly` · `canClaimAGI=false`.

## Method
Generalizes the A3 philosophy contamination find (26/528) to **all** committed packs. PR #362's
`build_teacher_data._benchmark_prompt_set` was **not yet on main** at audit time, so I constructed the
equivalent **forbidden set** = `eval_prompt_set` (eval/** + wisdom_market) **∪** the 7 domain holdouts
`tests/benchmark-{coding,history,math,personality,philosophy,psychology,religion}.json` **∪**
`tool_use_benchmark_prompt_set` **∪** `hk_advisor_benchmark_prompt_set` = **1307 normalized prompts**.
Row prompts extracted via `dataset_guard.prompt_of` (+ DPO chosen-message fallback), normalized-matched.

## Result — LEAKS (2 committed packs)

| pack | overlap | by holdout |
|---|---|---|
| **`training/corpus.jsonl`** | **65 / 528** | philosophy 26 · psychology 14 · religion 13 · history 11 · personality 1 |
| **`training/council/traces.jsonl`** | **1 / 125** | eval_prompt_set (1) |

**`training/corpus.jsonl` contaminated row indices (0-based, 65):**
`1,2,3,4,5,6,7,8,9,10,11,13,20,50,59,65,122,129,135,137,140,142,153,155,164,166,171,172,173,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,195,197,510,511,512,513,514,518,519,520,521,522,523,524,525,526,527`
(the philosophy-26 subset == the A3 v1 find, confirmed exact.)

**`training/council/traces.jsonl`:** index `12` (an `eval/**` prompt).

## Result — CLEAN (verified negatives are evidence)
- `training/moral_gate_sft.jsonl` 0/35 · `training/council/religion_repair_c4.jsonl` 0/12 ·
  `training/tool_use/sft_traces.jsonl` 0/80 · `training/tool_use/dpo_pairs.jsonl` 0/200 ·
  `training/self_evolve/distill.jsonl` 0/48.
- **Freshly-built `local_sophia_v2`** (`build_local_sophia_dataset --out <temp>`, its own guard reported
  `contamination guard: CLEAN`): the **TRAINING split is CLEAN** — `mlx/train.jsonl` **0/894**, and all
  `sft_*` / `dpo_*` / `general_instruct.jsonl` **0**. The 65 overlaps in `holdout.jsonl` and
  `mlx/valid.jsonl` are **BY DESIGN** — those files *are* the held-out eval/validation set; validating
  on the benchmark is correct, not contamination. **⇒ the build pipeline's decontam works.**

## Interpretation
The contamination lives in the **raw source `training/corpus.jsonl` (65 rows, 5 domains)** + one
`council/traces.jsonl` row — **not** in the built `local_sophia_v2` training split. The risk is any
consumer that reads `corpus.jsonl` **without** the full-benchmark decontam — exactly what leaked into
the A3 teacher pack pre-#362 (its `eval_prompt_set` missed `tests/benchmark-*.json`).

## Recommendations (cloud applies fixes; per T1 I did not edit data)
1. **Confirm #362's decontam covers all 7 domain holdouts** (not just philosophy) and apply the #362
   pattern to **every** pipeline that reads `corpus.jsonl`, not only `build_teacher_data`.
2. **Source cleanup:** remove the 65 listed rows from `training/corpus.jsonl` (a decontam PR like #362).
3. Re-source/remove `training/council/traces.jsonl` index 12.
4. Add a repo-wide CI guard: sweep all committed `training/**.jsonl` against the full forbidden set
   (this audit script) so leaks fail closed at PR time.

Machine-readable per-file results + indices: `audit_results.json` (this dir). `canClaimAGI=false`.
