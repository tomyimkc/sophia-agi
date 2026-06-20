# OKF Wiki — Sophia's provenance-native knowledge substrate

An [Open Knowledge Format](https://en.wikipedia.org/wiki/Knowledge_management) /
LLM-Wiki layer that turns Sophia's scattered provenance (structured `data/*.json` +
prose `docs/04-Disputes/*.md`) into **one version-controlled, machine-checkable
belief graph** the agent grows, the verifiers police, and the proof harnesses
measure. Sophia's differentiator — *source discipline* — literally **is** the
frontmatter, so OKF is an unusually good fit: every page carries `authorConfidence`,
`doNotAttributeTo`, `doNotMergeWith`, `tradition` as first-class metadata.

> Thesis: move Sophia from a provenance **dataset** to a provenance **reasoner** —
> an agent that maintains an internally-consistent, contradiction-aware model of
> who-said-what, and learns from maintaining it, without ever merging a lineage.

## Architecture

```
  raw/ ──► librarian (agent/wiki_librarian.py) ──┐  read → extract → gate → draft
                                                  ▼
  ┌──────────────────────────────────────────────────────────┐
  │ okf/ package — frontmatter codec · schema · wikilinks ·   │   data/*.json
  │ page graph · contradiction + confidence propagation        │ ◄─ (source of truth;
  └──────────────────────────────────────────────────────────┘    wiki is a projection)
        │                 │                  │                  │
   provenance         truth-maintenance   long-term         self-improvement
   VERIFIERS          (okf.graph:         MEMORY            FLYWHEEL
   (agent/verifiers)  contradictions,     (wiki_store +     (wiki_to_training →
   gate every write   confidence taint)   consolidation +   SFT/DPO)
        │                 │               recall in plan())     │
        └─────────────────┴──────────────────┴─────────────────┘
                          │
        AGI-proof: compounding curve · wiki_health · provenance lint (falsifier)
```

## The `okf/` package (dependency-free, 3.9+)

| Module | Role |
|---|---|
| `okf/frontmatter.py` | YAML-frontmatter codec — round-trippable subset, no external dep |
| `okf/schema.py` | page types, `authorConfidence` enum (mirrors `data/schema.json`), validation, confidence ranks |
| `okf/wikilinks.py` | `[[wikilink]]` parsing + slug normalization |
| `okf/page.py` | the `Page` object (frontmatter + body), load/save, edges |
| `okf/graph.py` | belief graph: dangling links, self-merges, tradition-merges, supersede cycles, **min-over-chain confidence propagation** |
| `okf/linker.py` | integrity report + backlinks + orphans + the contradiction ledger |

## Page format

```markdown
---
id: dao_de_jing
pageType: text          # text|concept|event|figure|figure_source_seat|school|tradition|domain|dispute|index|schema|memory
domain: philosophy
tradition: daoist
attributedAuthor: laozi
authorConfidence: legendary
doNotAttributeTo: [confucius, socrates, plato]
sources: ["data/attributions.json#dao_de_jing"]
links: [daoist]
---

# Dao De Jing (道德經)
...prose body...
```

## CLI surface

| Command | Purpose |
|---|---|
| `python tools/wiki_sync.py emit` / `check` | project `data/*.json` → `wiki/` pages / fail on drift (source of truth = data) |
| `python tools/wiki_validate.py` | schema + link integrity + contradictions + drift (CI gate) |
| `python tools/lint_wiki_provenance.py` | **provenance falsifier** — any `doNotAttributeTo` crossing is a failure |
| `python tools/wiki_health.py` | broken-links / orphans / contradictions / violations (long-horizon metric) |
| `python tools/wiki_ingest.py raw/x.txt --provider mock` | librarian: ingest a raw source → gated draft |
| `python tools/consolidate_runs.py` | fold agent runs into gated OKF memory pages (episodic → semantic) |
| `python tools/wiki_to_training.py` | mine wiki provenance → SFT/DPO (the flywheel) |
| `python tools/run_compounding_curve.py` | answerable-coverage vs wiki size (AGI-proof) |

## The provenance gate (the safety crux)

`agent/verifiers.py` encodes "don't merge lineages" as a **hard, machine-checked
verifier**:

- `provenance_faithful()` — fails if text asserts an attribution on a record's
  `doNotAttributeTo`. Sentence-scoped with negation/scare-quote/reported-speech
  carve-outs, so a page that *correctly* says "Confucius did **not** write the Dao
  De Jing" passes while "Confucius wrote the Dao De Jing" fails. **Zero false
  positives across the entire committed corpus; true merges caught.**
- `frontmatter_schema_valid()`, `no_broken_wikilink()`, `wiki_consistent()`.

Every agent write (`agent/wiki_store.py`, the `sophia_wiki_upsert` MCP tool) passes
this gate **even when approved** — so more autonomy can never mean more hallucinated
provenance. The MCP write tool is additionally `@audited(risk="medium")` (needs
`SOPHIA_MCP_APPROVE_WRITES=1`).

## Continual learning (no retraining)

- **Consolidation** (`agent/memory_consolidation.py`): a verified run's conclusion
  is folded into a gated OKF memory page — but a lineage-merging answer is *never*
  consolidated, so memory compounds without accumulating contamination.
- **Recall** (`agent/harness.py:_memory_recall`): the planner is handed relevant
  prior pages + `doNotAttributeTo`/`doNotMergeWith` warnings before it plans, so
  run N+1 builds on run N with frozen weights. Off-by-default `consolidate=True`
  closes the loop in `run_agent`.

## What is built (this slice)

| Layer | Status | Where |
|---|---|---|
| OKF package + frontmatter on 10 disputes + validator | ✅ | `okf/`, `tools/wiki_validate.py` |
| data ↔ wiki projection + drift CI gate | ✅ | `tools/wiki_sync.py` (58 entity/tradition pages) |
| provenance verifiers + contradiction graph + lint | ✅ | `agent/verifiers.py`, `okf/graph.py`, `tools/lint_wiki_provenance.py` |
| retrieval carries provenance + surfaces constraints | ✅ | `agent/retrieval.py` |
| librarian + wiki_store + skill + audited MCP tools | ✅ | `agent/wiki_librarian.py`, `agent/wiki_store.py`, `sophia_mcp/` |
| consolidation + plan-time recall | ✅ | `agent/memory_consolidation.py`, `agent/harness.py` |
| flywheel + health + compounding curve | ✅ | `tools/wiki_to_training.py`, `tools/wiki_health.py`, `tools/run_compounding_curve.py` |

## Honest limits (Sophia does not overclaim)

- The compounding curve's default metric is an **offline retrieval proxy**
  (`answerable-coverage@k`), not a semantic quality score; the model-backed
  quality curve is the `--provider` extension and needs manual two-pass review.
- The provenance verifier is a strong heuristic, not a parser of natural-language
  truth; it errs toward flagging assertions, with carve-outs for the corrections
  Sophia is proud of.
- `data/*.json` stays the single source of truth; `wiki/` is a generated, re-importable
  projection. Drift is a CI failure, not a silent merge.

This is the bridge from "AGI-candidate proof package" toward an auditable,
self-maintaining knowledge organism — generality you can inspect, not just trust.
See [AGI-Platform.md](./AGI-Platform.md) and [Roadmap.md](./Roadmap.md).
