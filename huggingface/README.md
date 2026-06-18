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
pretty_name: Sophia AGI Corpus
size_categories:
- n<1K
---

# Sophia AGI Corpus

**Wisdom before intelligence.** Bilingual training data for provenance-aware reasoning across philosophy, psychology, history, and religion (500 examples, v0.5.1).

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