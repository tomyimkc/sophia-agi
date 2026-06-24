# Sophia All-Phase Benchmarks

**Status:** implemented as deterministic/offline candidate infrastructure.
**Boundary:** These are benchmark harnesses and smoke fixtures. They are **not**
proof of AGI and not headline-grade external results until real-model runs clear
Sophia's no-overclaim gate.

## Why these phases

The current repo already has provenance-delta, OKF belief graph, conscience,
MCP tools, code execution, and hidden-eval infrastructure. The all-phase suite
turns the benchmark roadmap into one CI-safe command:

```bash
python tools/run_all_phase_benchmarks.py
```

Artifact:

```text
agi-proof/benchmark-results/all-phase-benchmarks.public-report.json
```

## Phases

| Phase | Dataset | Runner | Artifact | Purpose |
|---|---|---|---|---|
| SEIB-100 | `eval/seib/seib_100_v1.jsonl` | `tools/run_seib.py` | `seib-100.public-report.json` | Epistemic integrity: false attribution, contested authorship, fabrication, tradition merge rate. |
| Belief Revision 50 | `eval/belief_revision/belief_revision_50_v1.jsonl` | `tools/run_belief_revision_benchmark.py` | `belief-revision.public-report.json` | Counterfactual retraction, cascade, stale-belief leakage, audit trail. |
| AgentBench-Sophia 30 | `eval/agentbench_sophia/agentbench_sophia_30_v1.jsonl` | `tools/run_agentbench_sophia.py` | `agentbench-sophia.public-report.json` | Advisor / Repo / Life source-discipline and audit-trace reliability. |
| GPQA-Provenance smoke | `eval/gpqa_provenance/gpqa_provenance_smoke_v1.jsonl` | `tools/run_gpqa_provenance.py` | `gpqa-provenance-smoke.public-report.json` | Provenance contract for hard science QA. Not a GPQA-Diamond score. |
| Code Provenance 30 | `eval/code_provenance/code_provenance_30_v1.jsonl` | `tools/run_code_provenance.py` | `code-provenance.public-report.json` | Coding source/dependency discipline. Not SWE-bench/LiveCodeBench. |
| SEIB-Arena-20 smoke | `eval/arena/arena_20_v1.jsonl` | `tools/run_epistemic_arena.py` | `seib-arena-20.public-report.json` | Blind-comparison preparation. Not human preference evidence. |

## Non-circularity

SEIB-100 labels are derived from `provenance_bench/data/` external-citation /
Wikidata snapshot records. The runtime gate is a treatment only. The scorer is
independent of `agent.verifiers`.

## SEIB-100 conditions & honesty metrics

SEIB-100 runs five ablation rungs so a cheap prompt nudge is never confused with
the tool/gate machinery:

`raw` · `raw+prompt` (system-prompt-only, no tools/gate) · `raw+mcp` (skill) ·
`raw+gate` (provenance gate treatment) · `sophia_full`.

Per condition it reports `provenanceAccuracy`, `falseAttributionRate`,
`fabricationRateOnContested`, `qualificationRateOnContested`, `traditionMergeRate`,
`sourceCitationRate`, and — as the honesty counterweight required by the
provenance-delta spec — **`falsePositiveCost`** (did the discipline erase a
documented gold attribution?). A degenerate gate that nukes correct answers fails
the benchmark because `falsePositiveCost` is bounded in the `ok` criterion. The
`raw+prompt` rung is distinguished from the skill/gate rungs by
`sourceCitationRate` (`prompt_to_full_citation_delta`): prompt-only answers may be
correct but cite no provenance.

## Real-model SEIB-100 via OpenRouter

`tools/run_seib.py` now supports a real-model path:

```bash
export OPENROUTER_API_KEY="..."  # never commit this

python tools/run_seib.py \
  --real-model \
  --model openrouter:openai/gpt-4o-mini \
  --limit 10 \
  --runs 1 \
  --out agi-proof/benchmark-results/seib-100-openrouter-smoke.public-report.json
```

Then run the full candidate suite:

```bash
python tools/run_seib.py \
  --real-model \
  --model openrouter:openai/gpt-4o-mini \
  --runs 3 \
  --judges openrouter:anthropic/claude-sonnet-4.5,openrouter:qwen/qwen-2.5-72b-instruct \
  --out agi-proof/benchmark-results/seib-100-openrouter-gpt4omini.public-report.json
```

Current real-mode caveat: this path uses Sophia's deterministic SEIB scorer and
benchmark-side MCP context (`mcpMode=context_from_external_eval_sources`). It records
`judgeSpecs` for later validation but does not yet use LLM judges in the scoring
loop (`llmJudgesUsed=false`). Therefore even a real OpenRouter run remains
`validated=false` until a multi-run, multi-judge scorer is added and passes the
no-overclaim gate.

## Promotion rules

A number may be promoted from "candidate" to public headline only if it clears:

- real model run(s), not deterministic mock only;
- at least 3 runs;
- at least 2 independent judge families when semantic judging is used;
- Cohen's κ >= 0.40;
- 95% confidence interval excludes 0 for a claimed delta;
- false-positive cost reported explicitly.

Until then:

```json
{
  "candidateOnly": true,
  "level3Evidence": false,
  "validated": false,
  "canClaimAGI": false
}
```

## 中文摘要

本套件把 Sophia 的下一階段基準測試落地為可在 CI 離線執行的候選證據：
SEIB-100、信念修訂、三路 AgentBench-Sophia、GPQA-Provenance smoke、Code
Provenance、以及 SEIB-Arena smoke。所有輸出皆為 `candidateOnly: true`，不可
宣稱為 AGI 證明或正式外部排行榜分數；若要成為公開 headline，需要真實模型、
多次運行、兩個以上獨立裁判家族、一致性與信賴區間檢驗。
