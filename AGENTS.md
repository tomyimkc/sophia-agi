# Sophia AGI — AI Assistant Guide

Open corpus for **provenance-aware philosophy** and AGI-shaped epistemic reasoning.

## Read first

1. `README.md`
2. `docs/00-Index/Home.md`
3. `data/attributions.json`
4. `docs/04-Disputes/` for contested cases

## Rules

- Never attribute a text to a figure listed in `doNotAttributeTo`.
- Flag `authorConfidence: legendary` or `compiled` as uncertain — do not state as settled fact.
- Keep 儒家 (Confucian) and 道家 (Daoist) lineages distinct unless evidence supports a link.
- Training outputs: bilingual EN + 中文 summary; JSONL-compatible structure.

## Agent

- `grok-cli-teacher` — generates reviewed training pairs from dispute notes (see `.grok/agents/grok-cli-teacher.md`).

## Validation

```bash
python tools/validate_attribution.py
python tools/export_training_jsonl.py
```