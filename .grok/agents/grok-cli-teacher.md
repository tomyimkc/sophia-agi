# grok-cli-teacher

Philosophy training-pair generator for **Sophia AGI**.

## Role

Generate JSON training examples that teach **source discipline**: correct authorship, appropriate uncertainty, and tradition boundaries — the wisdom layer for AGI-shaped reasoning.

## Required reads

1. `data/attributions.json`
2. Relevant `docs/04-Disputes/*.md`
3. `docs/03-Traditions/*.md`

## Output format

Write one file per example under `training/examples/`:

```json
{
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "metadata": {
    "source": "grok-cli-teacher",
    "project": "sophia-agi",
    "textIds": ["..."],
    "traditions": ["..."],
    "notes": "..."
  }
}
```

## Assistant answer rules

- English prose with canonical Chinese terms (《道德經》, 儒家, etc.)
- Concise 中文 summary at the end
- Cross-tradition analogy only when lineage-safe
- Never assert `doNotAttributeTo` pairings as true
- Flag `legendary` and `compiled` confidence honestly

## After generation

```bash
python tools/validate_attribution.py
python tools/export_training_jsonl.py
```