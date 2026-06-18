# Reddit r/LocalLLaMA — post now

**Submit:** https://www.reddit.com/r/LocalLLaMA/submit  
**Flair:** Project (or Resource if available)

---

## Title

```
[Project] Sophia AGI — open benchmark for attribution traps (Confucius ≠ Dao De Jing, 9 philosophy cases)
```

---

## Body

```
I keep hitting the same failure mode in chat models: they merge authors and traditions instead of checking lineage first.

Built an open project to measure and train against that:

**Sophia AGI** — corpus + per-domain benchmark + optional agent

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
- HF dataset (20 examples): https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus

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