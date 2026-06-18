# Contributing to Sophia AGI

Thank you for helping build provenance-aware philosophy training data toward AGI-shaped reasoning.

## What we accept

- New **attribution records** in `data/attributions.json`
- **Dispute notes** for contested authorship (`docs/04-Disputes/`)
- **Training examples** in `training/examples/` (JSON, one file per example)
- **Benchmark questions** in `tests/attribution_bench.json`
- Tooling improvements that strengthen validation

## Required workflow

1. Add or update `data/attributions.json` for every text you cite.
2. If authorship is disputed, add `docs/04-Disputes/<slug>.md`.
3. Add a training example linked to those sources.
4. Run `python tools/validate_attribution.py` — must pass.
5. Open a pull request with a short summary of sources used.

## Training example format

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
    "textIds": ["dao_de_jing"],
    "traditions": ["daoist", "confucian"],
    "notes": "..."
  }
}
```

Assistant answers must:

- State attribution with appropriate uncertainty
- Explain why provenance matters for intellectual history
- Use English with canonical Chinese terms where relevant
- End with a concise 中文 summary

## Code of conduct

Be precise, cite evidence, and do not merge intellectual traditions without justification.