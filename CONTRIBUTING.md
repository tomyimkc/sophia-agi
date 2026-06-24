# Contributing to Sophia — the Wisdom Gate

Thank you for helping build the open foundation for **source discipline** in AI.

Every record and dispute you contribute makes the gate stricter and the benchmarks more trustworthy. "Wisdom before intelligence."

> **Sole authorship & public standard.** Sophia is authored and maintained by its sole author and rights holder, **tomyimkc**, and stays a fully public, no-overclaim project. By contributing, you license your contribution under the repository's [MIT License](LICENSE) and agree the sole author retains the project's brand/trademark and commercial-licensing rights (see [TRADEMARK-POLICY.md](TRADEMARK-POLICY.md) and [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)).

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

## Phase 2 — Claude teacher examples (human review)

Machine-generated examples from `tools/claude_teacher.py` use `metadata.source: "claude-teacher"`.
Before merging a teacher batch, spot-check:

- [ ] **Attribution traps** — `doNotAttributeTo` denials are explicit (no lineage merge).
- [ ] **Confidence labels** — `compiled`, `legendary`, `none_extant` match `data/attributions.json`.
- [ ] **Domain tags** — `metadata.domain` matches philosophy / psychology / history / religion.
- [ ] **Chinese summary** — assistant ends with a concise 中文 line.
- [ ] **No invented citations** — titles and authors appear in `data/` records.
- [ ] **Corpus export** — run teacher or `python tools/validate_attribution.py` after edits.

Regenerate corpus: teacher auto-writes `training/corpus.jsonl` on completion.

## Phase 4 — Correction loop

When external benchmarks fail (`benchmark/model_runs/*.report.json`):

```bash
python tools/run_correction_loop.py --dry-run    # list failures
python tools/run_correction_loop.py --generate   # Claude drafts -> training/corrections_pending/
python tools/run_correction_loop.py --promote    # move reviewed drafts to training/examples/
```

Review `metadata.source: "correction-loop"` the same way as teacher examples.

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