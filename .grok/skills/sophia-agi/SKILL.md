---
name: sophia-agi
description: >
  Operate the Sophia AGI provenance corpus: validate attributions, run the epistemic
  gate, score benchmarks, and route repo/agent tools. Use when the user mentions
  Sophia AGI, source discipline, provenance, attribution traps, corpus validation,
  benchmark scoring, or runs /sophia-agi. Prefer MCP tools (sophia_validate,
  sophia_gate_check, sophia_benchmark_*) when the sophia-agi MCP server is enabled.
metadata:
  short-description: "Sophia AGI corpus operator + epistemic gate"
---

# Sophia AGI skill

**Wisdom before intelligence.** Provenance-aware reasoning across philosophy, psychology, history, and religion.

## Read first

1. `AGENTS.md` and `README.md`
2. `data/attributions.json` for any cited text
3. `docs/04-Disputes/` when authorship is contested
4. `docs/09-Agent/MCP-Server.md` for MCP tool wiring
5. Portable skill (any repo): `skills/portable/sophia-source-discipline/` — install via `python tools/install_skills.py --all`

## Hard rules

- Never attribute a text to a figure in `doNotAttributeTo`.
- Treat `authorConfidence: compiled | legendary | none_extant` as uncertain — not settled fact.
- Keep traditions separate (e.g. 儒家 vs 道家) unless evidence supports a link.
- Assistant-style outputs: English + canonical Chinese terms + concise 中文 summary.
- Ask before multi-file writes; no commits unless the user asks.

## MCP tools (preferred when available)

| Tool | Use when |
|------|----------|
| `sophia_validate` | Before PR, after corpus edits |
| `sophia_gate_check` | Checking a draft answer for attribution traps |
| `sophia_benchmark_list` | Listing eval cases for a domain |
| `sophia_benchmark_score` | Scoring model responses JSON |
| `sophia_corpus_stats` | Version / example counts for release notes |
| `sophia_export_corpus` | Regenerate `training/corpus.jsonl` |
| `sophia_get_attribution` | Lookup philosophy textId |
| `sophia_get_record` | Lookup psychology/history/religion record |
| `sophia_list_disputes` / `sophia_read_dispute` | Dispute notes |

## CLI fallback

```bash
python tools/validate_attribution.py
python tools/sophia_agent.py advisor "..."
python tools/sophia_agent.py repo "..." --execute --approve
python tools/run_external_models.py --domain philosophy --providers claude-sonnet
python tools/score_benchmark.py benchmark/model_runs/MODEL.json --domain philosophy
python tools/claude_teacher.py --limit 20 --dry-run
python tools/prepare_lora_dataset.py --dry-run
```

## Workflows

### Add training data

1. Update `data/attributions.json` (and dispute note if needed).
2. Add `training/examples/NNN-slug.json` or run `claude_teacher.py`.
3. `sophia_validate` or `validate_attribution.py`.
4. Export: `python tools/export_training_jsonl.py`.

### Check an answer before publish

1. `sophia_gate_check` with `question` + `response` text.
2. If violations: fix attribution traps, add 中文 summary, re-check.

### Benchmark a model

1. `sophia_benchmark_list` for case IDs and questions.
2. Collect responses → `sophia_benchmark_score`.
3. On failures: `python tools/run_correction_loop.py --generate`.

## Agent paths

| Path | Command |
|------|---------|
| Advisor | `python tools/sophia_agent.py advisor "..."` |
| Repo | `python tools/sophia_agent.py repo "..."` |
| Life | `python tools/sophia_agent.py life "..."` |