# Show HN — post now (copy-paste)

**Submit:** https://news.ycombinator.com/submit  
**Checklist:** Logged in → paste below → URL field = thesis site → Submit → post first comment within 2 minutes.

---

## URL field

```
https://tomyimkc.github.io/sophia-agi/
```

(If 404, use `https://github.com/tomyimkc/sophia-agi` temporarily — switch URL when Pages propagates.)

---

## Title

```
Sophia AGI – benchmark for stopping LLMs from merging Confucius and Laozi
```

---

## Text (body)

```
LLMs routinely merge intellectual lineages: Confucius → Dao De Jing, Socrates → Republic, Freud → cognitive dissonance, nirvana → eternal heaven.

Sophia AGI (σοφία, wisdom) is an open corpus + per-domain benchmark + agent that enforces "source discipline" — named attribution, tradition boundaries, and confidence signals before reasoning.

Live today:
• 4 domains: philosophy, psychology, history, religion
• 20 bilingual training examples (Hugging Face)
• Per-domain leaderboards with explicit pass/fail markers
• Thesis-style site documenting UI council design
• Three-path agent: advisor | repo | life

Claude Sonnet scores 100% on all domains on our harness. We want GPT/Grok/local model runs — PRs with scored JSON welcome. <!-- claim-ok: scoped to "our harness" -->

Not claiming AGI — claiming we need a provenance gate before belief scales.

Thesis: https://tomyimkc.github.io/sophia-agi/
Repo: https://github.com/tomyimkc/sophia-agi
Dataset: https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus

Feedback wanted: (1) benchmark traps we're missing, (2) lm-eval-harness fit, (3) next domains (science, law).
```

---

## First comment (post right after submit)

```
Author here.

Council panel (religion) = named seats in the answer format — we score structure, not theology.

Philosophy benchmark: 9 traps (Mencius, Zhuangzi, Symposium added in v0.4.2).

Local agent: python tools/serve_web.py → thesis site + POST /api/ask

Contributions: good first issues in repo; attribution records especially welcome.
```

---

## After posting

1. Watch https://news.ycombinator.com/newest for your post (5–10 min)
2. Reply to every comment for the first 2 hours
3. Post Reddit draft next: `reddit-localllama-draft.md`