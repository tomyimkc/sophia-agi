---
language:
- en
- zh
license: mit
task_categories:
- question-answering
- text-generation
tags:
- philosophy
- psychology
- history
- religion
- agi
- provenance
- attribution
- source-discipline
- bilingual
- llm-safety
- ai-alignment
- knowledge-graph
- rag
- huggingface
pretty_name: Sophia AGI Corpus
size_categories:
- n<1K
---

# Sophia AGI Corpus

**Wisdom before intelligence.** Bilingual (EN + 中文) training data for provenance-aware reasoning across philosophy, psychology, history, and religion (**527** examples, v0.7+).

Includes teacher pairs + gate-filtered traces with attribution traps across four domains + OKF provenance wiki. The corpus powers the epistemic gate that drives 0% fabrication on unknown-answer cases (vs raw models 17-25%).

Pair with the [Sophia MCP server](https://github.com/tomyimkc/sophia-agi/blob/main/docs/09-Agent/MCP-Server.md), portable `/sophia-source-discipline` skill, self-extend flywheel, and verifier-gated contract for any pipeline. Live thesis + benchmarks: https://tomyimkc.github.io/sophia-agi/  Repo: https://github.com/tomyimkc/sophia-agi

**Key proof:** Sophia gate: 0% fabrication on held-out traps (deterministic scorer + multi-judge corroboration). Teacher/Grok-CLI: 100% on domain benchmarks.

## Dataset description

- **Project:** [github.com/tomyimkc/sophia-agi](https://github.com/tomyimkc/sophia-agi)
- **Format:** JSONL chat messages (`system` / `user` / `assistant`)
- **Focus:** Source discipline — correct authorship, tradition boundaries, appropriate uncertainty
- **Languages:** English with canonical Chinese philosophical terms + 中文 summaries

## Files

| File | Description |
|------|-------------|
| `corpus.jsonl` | Exported from `training/examples/` |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("tomyimkc/sophia-agi-corpus", split="train")
print(ds[0])
```

## Benchmark

Evaluate models on the **Sophia Attribution Benchmark** in the GitHub repo:

`tests/attribution_bench.json` + `tools/score_benchmark.py`

## Domains

Philosophy · Psychology · History · Religion — all active with per-domain benchmarks.

## Citation

```bibtex
@misc{sophiaagi2026,
  title={Sophia AGI: Provenance-Aware Philosophy Corpus},
  author={Sophia AGI contributors},
  year={2026},
  url={https://github.com/tomyimkc/sophia-agi}
}
```