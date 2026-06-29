---
name: ci-artifact-drift
description: >
  Run BEFORE committing or pushing anything in this repo, and whenever CI is red on a generated
  artifact or a measurement-contract gate. This repo's PRs most often go red not on real bugs but
  on (a) generated artifacts that drifted from their source of truth (RESULTS.md, the RAG index,
  wiki pages, the local-Sophia dataset, the version stamp, the failure ledger) and (b) the
  no-overclaim measurement gates (lint_claims, claim_gate, eval_stats, assert_decontam,
  lint_training_rows). Use to regenerate/verify these locally so the PR lands green the first time.
metadata:
  short-description: "Pre-push guard: regenerate generated artifacts + pass the measurement-contract gates"
---

# CI artifact-drift guard

In this repo several files are **generated from a source of truth** and a CI gate fails closed if
the committed copy drifts. The other recurring red is the **measurement (no-overclaim) contract**.
Both are deterministic and reproducible locally — run them before you push so CI is green first try.

## 1. The measurement contract (fast-ci runs this on every PR)

```bash
make claim-check          # the full local equivalent of the fast-ci gate:
#   tools/lint_claims.py          (no-overclaim copy + registry/recipe receipts)
#   tools/lint_training_rows.py   (habit-not-fact training rows)
#   tools/assert_decontam.py      (eval/holdout prompts never in training)
#   tools/eval_stats.py           (power/MDE + bootstrap CI + anytime-valid self-test)
#   tools/claim_gate.py --prefix M3-pilot     (headline recipe GO/NO-GO — must stay GO)
#   tools/claim_gate.py --prefix M3-transfer  (transfer recipe GO/NO-GO — must stay GO)
```
`make claim-check-fast` is the lighter subset the pre-commit hook runs. Install the hook once with
`make hooks` (sets `core.hooksPath=.githooks`). **Never** lower a gate threshold to force a pass —
if a gate flips to NO-GO, that is a real result: label it candidate and add a failure-ledger entry.

## 2. Generated-artifact drift gates (ci.yml fails on a stale committed copy)

Regenerate (drop `--check`/`--verify`) only what your change actually touched, then re-run the
check form to confirm it's clean. If you changed none of these sources, just run the checks.

| If you changed… | Regenerate | Verify (CI form) |
|---|---|---|
| `agi-proof/benchmark-results/published-results.json` | `python tools/build_results_page.py` | `python tools/build_results_page.py --check` |
| wiki sources / `wiki/` | `python tools/wiki_sync.py sync` | `python tools/wiki_sync.py check` |
| corpus / RAG inputs | `python tools/build_rag_index.py --local` | `python tools/build_rag_index.py --verify` |
| training pack inputs | `python tools/build_local_sophia_dataset.py` | `python tools/build_local_sophia_dataset.py --check` |
| `VERSION` / version refs | — | `python tools/check_version_consistency.py` |
| `agi-proof/failure-ledger.md` | (edit by hand) | `python tools/validate_failure_ledger.py --check` |
| provenance export demo | `python tools/export_prov.py --demo` | `python tools/export_prov.py --demo --check` |

> `RESULTS.md` is **generated** — never hand-edit it. To change a result, edit
> `published-results.json` and re-run `build_results_page.py`.

## 3. One-shot pre-push sweep

```bash
python -m compileall -q agent okf tools agi-proof 2>/dev/null   # syntax (fast-ci step 1)
make claim-check                                                # measurement contract
python tools/build_results_page.py --check  && \
python tools/wiki_sync.py check             && \
python tools/build_rag_index.py --verify    && \
python tools/build_local_sophia_dataset.py --check && \
python tools/check_version_consistency.py   && \
python tools/validate_failure_ledger.py --check && echo "ARTIFACTS CLEAN — safe to push"
```
If any line fails, regenerate that one artifact (table above), re-run, then push.

## Notes

- The required CI checks are **`fast`** (fast-ci.yml) and **`ci-complete`** (ci.yml). Both must be
  green for `main-protection` to allow a merge.
- A `test`/`ci-complete` shown as *failed* may actually be a **cancelled** run (main moved mid-run).
  Confirm the job conclusion before treating it as a real failure — see the `git-discipline` skill.
