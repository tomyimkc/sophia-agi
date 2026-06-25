# W1 Hidden Eval Execution — self-authored v2

Run ID: `hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2`  
Pack ID: `selfauthored-fugu-w1-2026-06-26-v2`  
Pack SHA-256: `2fe3b97d42d2e24a8a07b2c67494a996843429db8046836473818ba24c0c39e9`  
Branch/commit before run: `claude/agi-gap-audit-roadmap-12ft30` / `de52a7786be6fa1bc191c701fc65b89db140160a`  
Backend: `deepseek`; observed model: `deepseek-v4-pro`  
Seeds: single run, no explicit seed mechanism.

## Command

```bash
python3 tools/run_hidden_eval_sophia.py private/hidden-evals/hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2/pack.json --backend deepseek \
  --responses-out agi-proof/benchmark-results/hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.responses.json \
  --private-report-out agi-proof/benchmark-results/hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.private.json \
  --public-report-out agi-proof/benchmark-results/hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.public.json \
  --manual-review-out agi-proof/benchmark-results/hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.manual-review.json
```

## Aggregate

- Nonempty answers: 8/8
- Backend failures: 0
- Auto score: 30.76/40.0 (76.9%)
- Auto passed cases: 0/8
- Strict-ready count by deterministic rubric review: 0/8
- Manual semantic pending/adjudication: 8 checks

## Claim boundary

This closes only the W1 execution-health requirement for one unspent pack (8/8 nonempty, 0 backend failures, artifacts retained with checksums). It is **not validated** under `_is_validated`: single run only, no two independent judge families, no Cohen κ, no bootstrap CI excluding 0, and manual semantic review is pending. Pack authorship is self-authored by the runner, so third-party independence remains open. `canClaimAGI` remains false.
