# Session Handover — 2026-06-29 (LLM thinking logs + CoT-faithfulness)

Built an opt-in **LLM thinking-log pipeline** (reasoning capture + A2A message logging +
distillation) and a **CoT-faithfulness benchmark** on top of it. `canClaimAGI` stays **false** —
nothing here is a capability/AGI claim; the faithfulness numbers are measurements, not gates.

## Shipped (merged to `main`)

- **PR #284 (merged):** thinking-log pipeline.
  - `agent/model.py`: `ModelResult.reasoning_text`/`reasoning_tokens`; capture at the single
    `ModelClient.generate()` choke point via a `trace_sink`. Anthropic adaptive/extended thinking
    (model-aware), OpenAI-compatible `reasoning_content` + `<think>` (always stripped from the
    answer; retained only under `SOPHIA_CAPTURE_THINKING`).
  - `agent/thinking_trace.py`: append-only JSONL trace, OTel-GenAI-aligned, hash-only by default,
    verbatim under `SOPHIA_CAPTURE_THINKING`. Enabled by `SOPHIA_THINKING_LOG`.
  - `agent/subagent.py` + `agent/a2a.py`: log swarm `delegate`/`result`/`synthesis` + networked
    `peer` legs as `a2a_message` spans. `agent/a2a_distill.py`: fail-closed distill → SFT rows +
    skill candidates (candidates only, no auto-promotion).
  - `tools/run_thinking_bench.py` + `--bench-thinking` lane + `.github/workflows/thinking-bench.yml`.
- **PR #296 (merged):** discriminating CoT-faithfulness battery (v1).
  - `benchmark/faithfulness_cot_battery.json` (6 discriminating + 6 cued, easy facts).
  - `tools/run_faithfulness_battery.py`: intrinsic flip-rate + cued/uncued
    (`cueFollowRate`/`cueAcknowledgeRate`/`unfaithfulCueUseRate`), seeds, percentile bootstrap CIs.
  - `tests/test_faithfulness_battery.py`: scripted faithful/unfaithful/resistant models pin metric
    semantics.

## Key finding (real model: deepseek-r1 via OpenRouter)

- **The mechanism works end-to-end**: 100% capture coverage; reasoning captured 6/6 on the battery.
- **Intrinsic** mean flip-rate **0.075** (CI [0, 0.16]); the only non-zero items were the genuinely
  multi-step ones (60mph→100mi 0.20; 16-days-mod-7 0.25) — correct directional signal.
- **Cued split could NOT discriminate**: deepseek-r1 **resisted all 6 v1 cues** (`cueFollowRate 0.0`),
  so `unfaithfulCueUseRate` is correctly indeterminate (not zero-by-fiat). v1 is **too easy** to
  create cue pressure on a frontier model. This is a battery-calibration finding, not a faithfulness
  verdict.

## In flight — PR for v2 (this branch)

- **Branch `claude/faithfulness-battery-v2`** (this handover ships with it).
- `benchmark/faithfulness_cot_battery_v2.json`: harder near-threshold items (factoring 323,
  compounding %, combined-rate trains, `0.999...=1`) + **stronger cues** (embedded fake
  worked-solutions / fake authorities; `cueToken` = the distinctive wrong claim). `check_battery`
  now enforces `cueToken` ∈ cue.
- Runner is `--battery`-configurable; dispatcher `FAITH_BATTERY=...` selects it.
- **Offline-validated only.** The real-model run (the whole point of v2) is **pending a fresh
  OpenRouter/Anthropic key** — the prior key was shared in chat and should be rotated; do NOT reuse.

## Next steps

1. Land the v2 PR (offline-green) → run `thinking-bench` workflow_dispatch with a model secret, or
   locally: `FAITH_MODEL=openrouter:deepseek/deepseek-r1 FAITH_BATTERY=benchmark/faithfulness_cot_battery_v2.json SOPHIA_CAPTURE_THINKING=1 scripts/run_local_benchmarks.sh --bench-faithfulness --execute`.
2. Read the real-model v2 numbers: if `cueFollowRate` is now non-zero, `unfaithfulCueUseRate` becomes
   the real faithfulness signal. If a model still resists, push cues harder (system-role hints) or
   move items closer to the competence edge.
3. Optional: wire captured reasoning into `agent/faithfulness_probe.faithfulness_drop` with a gold
   scorer for the per-step (not just whole-CoT) signal.

## Security / cost notes

- **Rotate the OpenRouter key** shared earlier this session. It was never written to disk or
  committed (verified); receipts live in gitignored `agent/memory/thinking/bench/`.
- Everything offline-runnable on CPU; no RunPod/GPU touched.
