# Reddit r/LocalLLaMA — post now

**Submit:** https://www.reddit.com/r/LocalLLaMA/submit  
**Flair:** Project (or Resource if available)

---

## Title

```
[Project] Sophia AGI v0.5.3 — 500-example provenance corpus + MCP skill for attribution traps
```

---

## Body

```
I keep hitting the same failure mode in chat models: they merge authors and traditions instead of checking lineage first.

Built an open project to measure and train against that:

**Sophia AGI** — 500-example corpus + per-domain benchmark + MCP tools + portable AI skill

New in v0.5.3:
- `/sophia-source-discipline` skill (works in any project)
- MCP server: validate, epistemic gate, benchmark score, attribution lookup
- LoRA experiment harness (holdout eval)

Domains:
- Philosophy (9 traps: Confucius/Laozi, Socrates/Plato, Mencius, Zhuangzi, Symposium…)
- Psychology (subfield tags: cognitive vs clinical vs pop_myth)
- History (myth labeling)
- Religion (council-panel answer format — scored on structure, not theology)

Harness is reproducible in CI — heuristic markers + bilingual 中文 checks.

**Leaderboard (same contract for every model):**
- Reference teacher: 100% all domains
- Claude Sonnet: 100% all domains
- Looking for GPT-4o / Grok / local model runs — PRs with scored JSON welcome

Links:
- Thesis site: https://tomyimkc.github.io/sophia-agi/
- GitHub: https://github.com/tomyimkc/sophia-agi
- HF dataset (500 examples): https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus
- Release: https://github.com/tomyimkc/sophia-agi/releases

Run locally:
python tools/run_external_models.py --all
python tools/score_benchmark.py your_file.json --domain philosophy

Not claiming AGI — "source discipline" before belief scales. Feedback on missing traps especially welcome.
```

---

## First comment (optional)

```
Happy to add a llama.cpp / ollama template if anyone wants to run a local 7B/70B through the harness — reply with your stack.
```