# Reddit drafts

## r/LocalLLaMA

**Title:** `[Project] Sophia AGI — benchmark for attribution discipline (Confucius ≠ Dao De Jing, Freud ≠ cognitive dissonance)`

**Body:**

```
Built an open benchmark + training corpus for a problem I keep hitting in RAG/chat: models merge authors and traditions.

Sophia AGI tests per-domain "source discipline":
- Philosophy: authorship traps (9 cases now — Mencius, Zhuangzi, Symposium added)
- Psychology: subfield tags (cognitive vs clinical vs pop_myth)
- History: myth labeling
- Religion: council-panel format with named seats

Harness: heuristic markers + bilingual 中文 checks — reproducible in CI.

Leaderboard (same contract for all models):
- Reference teacher: 100% all domains
- Claude Sonnet (via my API proxy): 100% all domains
- Looking for GPT-4o / Grok / local model submissions

Links:
- Thesis site: https://tomyimkc.github.io/sophia-agi/
- GitHub: https://github.com/tomyimkc/sophia-agi
- HF dataset: https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus

Run locally:
python tools/run_external_models.py --all
python tools/score_benchmark.py your_responses.json --domain philosophy

Happy to add llama.cpp template if someone wants to run a 7B through it.
```

## r/MachineLearning

**Title:** `[P] Sophia AGI — provenance-aware reasoning benchmark across philosophy, psych, history, religion`

**Body:** (shorter — link thesis + leaderboard table screenshot)

```
We measure whether LLM answers respect authorship, tradition boundaries, and subfield tags before reasoning.

20 bilingual training examples, 4 domain benchmarks, open scoring harness. Thesis exposition + UI council design doc on the site.

Repo: https://github.com/tomyimkc/sophia-agi
Paper-style overview: https://tomyimkc.github.io/sophia-agi/

Seeking collaborators for lm-eval-harness integration and science/law domain pilots.
```