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

## First real-model SEIB-100 run (candidate, single run)

A genuine real-model run was executed against a **local Ollama backend**
(`ollama:llama3.2:3b`) — not a mock — over all 100 cases x 5 conditions
(500 model calls). Artifact:
`agi-proof/benchmark-results/seib-100-ollama-llama32-3b.public-report.json`.

| Condition | Prov. acc | False-attr | Fab (contested) | Qual (contested) | FP cost | Cite rate |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.66 | 0.02 | 0.62 | 0.38 | 0.20 | 0.00 |
| raw+prompt | 0.84 | 0.00 | 0.32 | 0.68 | 0.24 | 0.00 |
| raw+mcp | 1.00 | 0.00 | 0.00 | 1.00 | 0.02 | 0.74 |
| raw+gate | 0.66 | 0.02 | 0.62 | 0.38 | 0.18 | 0.07 |
| sophia_full | 1.00 | 0.00 | 0.00 | 1.00 | 0.02 | 0.93 |

Headline deltas (single run, deterministic lexical scorer): raw→full provenance
accuracy **+0.34**; raw→full contested-fabrication reduction **0.62**;
prompt→full citation delta **0.93**; full false-positive cost **0.02**.

**Boundary / honest caveats (verification pass):**

- `validated: false`, `candidateOnly: true`. This is **one run** with a
  **deterministic lexical scorer** and **no independent LLM judges**, so it does
  not meet the no-overclaim promotion bar (>=3 runs, >=2 judge families,
  kappa>=0.40, CI excluding 0).
- The lexical scorer credits **hedging** on contested cases even when the model
  names the wrong author (observed ~10/50 raw contested rows hedge without naming
  the gold author). A future LLM-judge scorer is required before any headline.
- `raw+gate` ≈ `raw` here because the source-discipline gate only fires when the
  raw text literally asserts a known-forbidden lineage (fired on 6/50 false
  cases); it is precise and narrow by design, not a general hallucination filter.
- The lift attributed to `raw+mcp` / `sophia_full` reflects **tool-grounded
  context + gate discipline**, not model intelligence — the MCP context encodes
  the externally-labeled provenance under test.

## API + stronger-local follow-up runs (candidate)

After adding `--real-model` and LLM judge support, three priority follow-ups were
run using real providers / local models. These are still **candidate-only**:

1. **OpenRouter full SEIB-100 (subject: `openrouter:deepseek/deepseek-chat`)**
   - Artifact: `agi-proof/benchmark-results/seib-100-openrouter-deepseek.public-report.json`
   - Full 100 cases × 5 conditions; deterministic SEIB scorer; no LLM judges in this full run.
   - Result: raw accuracy 0.95 → Sophia-full 0.98; contested fabrication 0.10 → 0.04;
     source citation 0.00 → 0.94; full false-positive cost 0.02.
   - `ok=false` because the strict SEIB pass criterion requires zero contested fabrication;
     this is useful negative evidence, not a failure of the harness.

2. **OpenRouter judged balanced slice (subject: `openrouter:deepseek/deepseek-chat`)**
   - Artifact: `agi-proof/benchmark-results/seib-20-balanced-openrouter-deepseek-judged.public-report.json`
   - 20 balanced cases (10 false attribution + 10 contested) × 5 conditions.
   - Judges: direct DeepSeek + OpenRouter Qwen; mean pairwise agreement 0.9737 on valid judged rows.
   - Result: all conditions scored 1.00 on this easy balanced slice; Sophia-full source
     citation rate 1.00 vs raw 0.00.
   - `ok=false` only because the fixture is 20 cases, not the full 100-case SEIB.

3. **Stronger local model balanced slice (`ollama:qwen3:30b-a3b`)**
   - Artifact: `agi-proof/benchmark-results/seib-20-balanced-ollama-qwen3-30b-a3b.public-report.json`
   - 20 balanced cases × 5 conditions; deterministic SEIB scorer.
   - Result: raw accuracy already 1.00 on this slice; Sophia-full kept accuracy 1.00,
     reduced false-positive cost 0.30 → 0.10, and raised citation rate 0.00 → 0.80.
   - `ok=false` because it is a 20-case slice and full false-positive cost is at the
     strict threshold (0.10), not below it.

**Provider caveat:** an OpenRouter `openai/gpt-4o-mini` preflight was region-blocked,
and the supplied Claude-compatible llmhub route returned empty content through the
current adapter, so they were not used for scored artifacts. No keys are stored in
the repository.
