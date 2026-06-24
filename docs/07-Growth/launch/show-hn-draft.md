# Show HN — draft (ready to post)

**When:** Tuesday–Thursday, 8–10am US Eastern (best HN window).  
**URL:** https://tomyimkc.github.io/sophia-agi/ (after Pages enabled) or https://github.com/tomyimkc/sophia-agi

---

## Title (pick one)

1. **Sophia AGI – benchmark for stopping LLMs from merging Confucius and Laozi**
2. **Show HN: Open corpus + benchmark for provenance-aware LLM answers**
3. **Sophia AGI – source discipline before AGI-scale belief propagation**

## Post body

```
LLMs routinely merge intellectual lineages: Confucius → Dao De Jing, Socrates → Republic, Freud → cognitive dissonance, nirvana → eternal heaven.

Sophia AGI (σοφία, wisdom) is an open corpus + per-domain benchmark + agent that enforces "source discipline" — named attribution, tradition boundaries, and confidence signals before reasoning.

What's live today:
- 4 domains: philosophy, psychology, history, religion
- 20 bilingual training examples on Hugging Face
- Per-domain leaderboards with explicit pass/fail markers
- Thesis-style site with UI council design record
- Three-path agent: advisor | repo | life

Claude Sonnet scores 100% on all domains on our harness; GPT/Grok runs welcome (PRs with scored JSON). <!-- claim-ok: scoped to "our harness" -->

Not claiming AGI — claiming we need a provenance gate before belief scales.

Site: https://tomyimkc.github.io/sophia-agi/
Repo: https://github.com/tomyimkc/sophia-agi
Dataset: https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus

I'd love feedback on: (1) benchmark traps we're missing, (2) whether this belongs in lm-eval-harness, (3) domains to add next (science, law).
```

## First comment (post immediately after submit)

```
Author here. Quick clarifications:

- "Council panel" answers (religion) require named seats — we score format, not theology.
- Philosophy benchmark now has 9 traps including Mencius/Zhuangzi/Symposium attribution.
- `python tools/serve_web.py` runs the thesis site + live /api/ask locally.

Good first issues tagged in repo — attribution records and benchmark traps especially welcome.
```