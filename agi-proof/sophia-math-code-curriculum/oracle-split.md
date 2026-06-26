# Train-oracle vs evidence-oracle split

**Experiment:** `sophia-math-code-curriculum` on `Qwen/Qwen2.5-7B-Instruct`  
**Registered:** 2026-06-25 on branch `claude/sophia-math-code-curriculum` **before** GPU training.

## THE ONE RULE

| Family | Purpose | May cite as third-party / benchmark evidence? |
|---|---|---|
| **Training oracle** | Sympy canonical equivalence (`agent/math_verifier.py`), sandboxed code execution (`agent/code_verifier.py`), synthetic curriculum from `tools/gen_math_pack.py` and verifier-synthesis packs | **No** — release / curriculum gate only |
| **Evidence oracle** | Held-out MATH, GSM8K, HumanEval, MBPP style samples (sealed manifest), plus hidden reviewer pack when run | **Yes** — when run with ≥3 seeds and 95% CI excludes 0 |

Training-oracle passes must **never** be cited as MATH/GSM8K/HumanEval/MBPP headline proof.

## Training oracle (hard checks)

- **Math:** sympy parse → simplify → `accepted` / `rejected` / `abstain` (fail-closed when sympy absent).
- **Code:** subprocess in isolated temp dir, CPU/memory rlimits, wall-clock timeout, no network; hidden tests appended after model solution.
- **Synthetic generation:** sympy-verified golds (`tools/gen_math_pack.py`); entity-disjoint families; `heldout_seal_guard` blocks reading sealed eval splits.

## Evidence oracle (citable benchmarks)

- `eval/external/math-style-sample.jsonl` — MATH-style symbolic (NOT official MATH)
- `eval/external/gsm8k-style-sample.jsonl` — GSM8K-style numeric (NOT official GSM8K)
- `eval/coding/mbpp-style-sample.jsonl` — MBPP-style (NOT official MBPP)
- `eval/coding/humaneval-style-sample.jsonl` — HumanEval-style (NOT official HumanEval)
- `benchmark/code_tasks.json` — hidden-test code uplift tasks
- `provenance_bench/data/math_problems.json` — eval-family sympy RLVR held-out
- Hidden reviewer pack (when commissioned) — third-party calibration

Official third-party sets (full MATH/GSM8K/HumanEval/MBPP) may replace style samples when licensed; re-seal with `tools/seal_math_code_heldout.py` before citing.

## Contamination

- `python tools/build_local_sophia_dataset.py --check` must report **CLEAN**
- Sealed hashes: `agi-proof/sophia-math-code-curriculum/heldout-seal.manifest.json`
  (schema `sophia.math_code_heldout_seal.v2` — now carries per-file `provenance` +
  `pretrainingContaminationCaveat`)
- `canClaimAGI` stays **False** regardless of outcome

## Pretraining-contamination caveat (style samples)

The `eval/external/*-style-sample.jsonl` and `eval/coding/*-style-sample.jsonl` items are
**repo-authored style-samples** (clearly marked `authorship` in the seal manifest), NOT
the official MATH/GSM8K/HumanEval/MBPP sets. They inherit the field's pretraining-
contamination problem: a base model may have seen similar items during pretraining, so a
held-out gain on these is **suggestive, not contamination-free proof**. The per-file
`pretrainingContaminationCaveat` in the seal manifest makes this machine-readable.

A **clean external** generalization claim requires the third-party-authored pack at
[`agi-proof/third-party-heldout/`](../third-party-heldout/) (independent author,
machine-checkable oracles, salted commitment sealed before any model run). That pack is
**EMPTY by design** today; until it is filled and a gated run clears the no-overclaim bar,
the honest wording stays "AGI-candidate", not "externally validated".

