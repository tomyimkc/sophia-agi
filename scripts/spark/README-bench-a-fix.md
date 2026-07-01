# bench-a fix patch (2026-06-30)

`2026-06-30-bench-a-fix.patch` fixes the two failures that made the cloud-bridge bench-A run a
NO-GO on 2026-06-30 (`bridge/results/2026-06-30-claude-web-bench-a-02.json`):

1. **70B judge `n=0`** — the Mac mlx_lm.server was addressed with the `openai:` provider, which
   has no `api_key_default`; with `OPENAI_API_KEY` unset, `resolved_key()` returns `None` and the
   OpenAI client throws on every call (swallowed → `n=0`). Fix: `vllm:` provider
   (`api_key_default="EMPTY"`), keeping `/v1` on both judge URLs.
2. **A3 `FileNotFoundError`** — A2 emits pairwise verdicts; A3 wants per-family content booleans
   and nothing produced its `judgments.json`. Fix: `judge_pilot_answers.py --raw-out` +
   `tools/assemble_uplift_judgments.py` (A2.5) wired into `run_local_benchmarks.sh`.

It is generated against the `spark-bridge` branch's exact content (NOT the feature branch, whose
`run_local_benchmarks.sh` diverged — it lacks the `/v1` fix). Apply it on the Spark's bridge
checkout while that checkout is on `spark-bridge`:

```bash
cd /home/tomyimkc/sophia-bridge
git fetch origin spark-bridge claude/sophia-positioning-gaps-84kb0v
git checkout spark-bridge && git merge --ff-only origin/spark-bridge
git show origin/claude/sophia-positioning-gaps-84kb0v:scripts/spark/2026-06-30-bench-a-fix.patch | git apply --check -   # dry-run
git show origin/claude/sophia-positioning-gaps-84kb0v:scripts/spark/2026-06-30-bench-a-fix.patch | git apply -
# sanity: vllm + /v1 on BOTH judge URLs
grep -n 'JUDGES="${JUDGES' scripts/run_local_benchmarks.sh
python tools/assemble_uplift_judgments.py --selftest
git add -A && git commit -m "bench-a fix: keyless vllm judge provider + A2→A3 assembler"
git push origin spark-bridge
```

The poller ff-syncs `spark-bridge`, so once pushed the next `--bench-a --execute` runs the fixed
script. Verdict stays CANDIDATE/NO-GO unless ALL gate bars pass; `canClaimAGI` stays false.

---

## Follow-up: `2026-06-30-judge-parallel-families.patch` (apply AFTER bench-a-03 finishes)

Makes `judge_pilot_answers.py` judge the two families CONCURRENTLY (Qwen on the Spark + 70B on
the Mac at the same time) instead of sequentially — reclaims the cross-box idle gap, so wall-clock
drops from `Qwen_time + 70B_time` toward `max(Qwen_time, 70B_time)`. Verdicts are identical
(verified offline); only timing changes.

DO NOT apply while a bench-A run is in flight. Once `bench-a-03` has a result, on the Spark:

```bash
cd /home/tomyimkc/sophia-bridge
git checkout spark-bridge && git pull --ff-only origin spark-bridge
git show origin/claude/sophia-positioning-gaps-84kb0v:scripts/spark/2026-06-30-judge-parallel-families.patch | git apply --check -
git show origin/claude/sophia-positioning-gaps-84kb0v:scripts/spark/2026-06-30-judge-parallel-families.patch | git apply -
python tests/test_judge_parallel_families.py
git add -A && git commit -m "judge: families judged concurrently (both boxes at once)"
git push origin spark-bridge
```
