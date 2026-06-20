# The Provenance Delta — Design Spec

**Date:** 2026-06-21
**Status:** Approved → implemented
**Phase:** 1 of the "top-tier" roadmap (credibility seed)

## Problem

Sophia owns a real, machine-checked asset — the `provenance_faithful` gate
(`agent/verifiers.py`) and the guarded completion loop (`agent/guarded.py`). But
the project has never pointed that gate at the *outside world* and measured what
it buys. The AGI-candidate claim ladder lists external evaluation (item 6) and
independent replication (item 7) as **"not yet run."** Without an external,
reproducible number, "most credible / most capable / most adopted" have no
foundation.

The missing artifact is a single sentence backed by a one-command run:

> On an independent dataset of attribution claims, frontier models assert a false
> lineage X% of the time; behind Sophia's gate that drops to Z% — reproducible by
> anyone in one command.

## The circularity trap (the central constraint)

If we score "how often the gate catches violations" using cases drawn from the
gate's **own** corpus, the result is meaningless — the gate catches its own
corpus by construction. To be credible the benchmark must:

1. Take **ground truth from outside Sophia's gate** (Wikidata author relations +
   a citation-backed misattribution set).
2. Use the **gate only as the runtime treatment**, never as the judge.
3. Use an **independent judge** (gold-author comparison; optionally an
   LLM-judge that is a *different* model than the one under test) to decide
   whether an answer asserted a false attribution.

Label source and gate are kept in physically separate files and code paths.

## Ground truth (non-circular)

- `provenance_bench/data/wikidata_snapshot.json` — **true** attributions. Each
  row: `work`, `gold_author`, `wikidata_qid`, `source_url`. Committed snapshot so
  runs are deterministic/offline; refreshable via `tools/fetch_wikidata_authors.py`.
- `provenance_bench/data/misattributions.json` — **known-false** attributions
  (the lineage merges). Each row cites *why* it's false (`source_url` + `reason`),
  so the label is auditable, not Sophia's say-so. Seeded from the dispute pages
  but labeled by external citation.

The gate's `doNotAttributeTo` corpus (`data/*.json`) is **not** a label source
here; it is only the runtime treatment. Their overlap is what we *measure* as
coverage — we never assume it.

## Components (each small, single-purpose, offline-testable)

| Module | Responsibility |
|---|---|
| `provenance_bench/dataset.py` | Load the two external files → list of `Case`; `write_jsonl`. |
| `provenance_bench/judge.py` | Independent judge: extract the author an answer *asserts* for a work and whether it abstained; compare to `gold_author`. Default = lexical extractor (a screen, clearly labeled); injectable `judge_fn` hook for an LLM-judge. **Shares no code with the gate.** |
| `provenance_bench/runner.py` | Per case: produce the *raw* model answer and the *gated* answer (via `agent.guarded.guarded_complete`). Model injected as a `generate` callable (mock in tests). |
| `provenance_bench/score.py` | Aggregate the three metrics from judged raw vs gated answers. |
| `provenance_bench/report.py` | Emit `agi-proof/benchmark-results/provenance-delta.public-report.json` + a markdown table. |
| `tools/run_provenance_delta.py` | CLI entry point (build → run → score → report). |
| `tools/fetch_wikidata_authors.py` | Optional network refresh of the snapshot. |

## The three honest metrics

1. **Hallucinated-attribution rate** — of answers that assert an attribution,
   the fraction whose asserted author contradicts `gold_author`. Reported
   **alone** (raw answer) vs **behind gate** (guarded answer). The delta is the
   headline.
2. **False-positive cost** — on **true**-attribution cases (where affirming the
   gold author is correct), the fraction where the gated pipeline wrongly
   abstained/blocked/changed a correct answer. Keeps us honest: a gate that nukes
   everything would otherwise look perfect on metric 1.
3. **Coverage / recall** — of the false attributions a model gets wrong *alone*,
   the fraction the gate actually fires on. Names the gate's narrowness instead
   of hiding it.

`delta = hallucination_alone − hallucination_gated`.

## Reuse, don't rebuild

Built on existing pieces unchanged: `agent/guarded.py` (guarded loop +
repair/abstain), `agent/verifiers.py:provenance_faithful` (gate),
`agent/model.py` (multi-provider adapter incl. `mock`), and the `agi-proof/`
report convention. New code is only: two data files + dataset/judge/runner/score/
report + a thin CLI. **No change to the gate in this phase.**

## Testing

Every module offline-testable with the `mock` provider and an injected judge —
no API calls in CI. Real-model runs are an explicit, opt-in CLI invocation.
`tests/test_provenance_bench.py` is wired into CI.

## Scope guardrails (YAGNI)

- No leaderboard/submission portal (Phase 3).
- No gate-corpus expansion (separate "harden" track — measure current coverage
  honestly first).
- Textual authorship only to start (legal/code provenance is Phase 2).

## Model set (default, adjustable)

One Claude, one GPT, one Gemini, plus one local model (e.g. Qwen2.5-7B via
Ollama). The local-vs-frontier contrast — "small model + gate beats a frontier
model alone" — is the most compelling result.
