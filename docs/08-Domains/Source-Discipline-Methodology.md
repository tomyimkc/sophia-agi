# Source Discipline Methodology (Philosophy → All Domains)

Philosophy training example **001** established the core pattern. Psychology and religion now use the same **source-centered** hub model.

## Core pattern (from philosophy)

1. **Named attribution** — who wrote/coined what (`attributedAuthor`, `doNotAttributeTo`)
2. **Confidence signal** — compiled, legendary, disputed, consensus
3. **Tradition / subfield boundary** — do not merge lineages
4. **Bilingual anchor** — English + canonical 中文 terms + 中文 summary
5. **Multi-source hub** — one training example can teach several benchmark traps (example 001 → 4 philosophy cases)

## Psychology center

| Asset | Path |
|-------|------|
| Source records | `data/psychology_concepts.json` |
| Hub example | `training/examples/018-psychology-source-discipline-hub.json` |
| Per-trap examples | `002`, `005`–`007`, `015` |
| Benchmark | `tests/benchmark-psychology.json` |

**Subfields:** `cognitive`, `clinical`, `pop_myth` — always tag in answers.

## Religion center

| Asset | Path |
|-------|------|
| Source records | `data/religion_concepts.json` |
| Council mode | `docs/08-Domains/Religion-Council-Debate-Mode.md` |
| Hub example | `training/examples/019-religion-source-discipline-hub.json` |
| Per-trap examples | `004`, `009`–`011`, `014` |
| Benchmark | `tests/benchmark-religion.json` |

**Answer mode:** council panel — all seats named; split theology vs history when appropriate.

## Reference pipeline

```bash
python tools/export_training_jsonl.py
python tools/build_reference_responses.py
python tools/score_benchmark.py benchmark/reference/responses-psychology.json --domain psychology
```

Target: **100%** reference score per domain before external model runs.