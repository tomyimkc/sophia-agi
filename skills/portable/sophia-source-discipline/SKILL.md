---
name: sophia-source-discipline
description: >
  Apply Sophia AGI source discipline in any project: correct authorship, tradition
  boundaries, myth labeling, and provenance uncertainty. Use when answering who wrote
  what, citing philosophy/psychology/history/religion, checking attribution traps,
  or when the user runs /source-discipline or /sophia-discipline. Works without
  the sophia-agi repo; use sophia-agi MCP tools when that server is available.
metadata:
  short-description: "Portable provenance + attribution discipline"
---

# Sophia source discipline (portable)

**Wisdom before intelligence.** Evidence before reasoning. Provenance before merge.

Public corpus: [github.com/tomyimkc/sophia-agi](https://github.com/tomyimkc/sophia-agi) · HF: [sophia-agi-corpus](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus)

## When to invoke

- "Who wrote X?"
- Philosophy, psychology, history, or religion attribution questions
- Pop myths vs scholarly consensus (pasta/Marco Polo, flat earth, 10% brain, etc.)
- Mixing traditions (Confucius + Dao De Jing, Socrates + Republic, Freud + cognitive dissonance)

## Hard rules

1. **Deny false attributions explicitly** — name the wrong author, then give the correct one.
2. **Signal uncertainty** — `compiled`, `legendary`, `disputed`, `none extant` when authorship is not settled.
3. **No lineage merge** — keep Confucian 儒家, Daoist 道家, Platonist, Stoic, clinical vs pop psych separate unless evidence links them.
4. **Label myths** — use "myth", "misconception", "popular belief", 迷思/神話 when correcting pop claims.
5. **Psychology subfields** — tag cognitive vs clinical vs pop_myth; deny Freud for Festinger-era concepts.
6. **Religion** — council/panel format when multiple traditions apply; sensitive historical claims stay scholarly.
7. **Output shape** — English + canonical Chinese terms; end with concise **中文** summary.

## Common traps (deny these)

| Trap | Correct stance |
|------|----------------|
| Confucius wrote 《道德經》 | No — traditionally Laozi; legendary attribution |
| Socrates wrote *Republic* | No — Plato wrote dialogues; Socrates wrote nothing extant |
| Freud discovered cognitive dissonance | No — Leon Festinger (1957) |
| Left brain / right brain learning styles | Pop myths — not supported as stated |
| Marco Polo brought pasta to Italy | Myth — pasta predates in Italy |
| Medieval Europeans thought Earth flat | Oversimplification / myth |
| Vikings wore horned helmets | Myth |
| Great Wall visible from space | Myth |
| Nirvana = heaven in pop Buddhism | Misconception — contextualize tradition |

See `references/trap-patterns.md` for extended patterns.

## Answer template

```
1. Direct answer (yes/no/uncertain) with named author or record
2. Why provenance matters (1–2 sentences)
3. Tradition / subfield boundaries
4. 中文：一句話總結
```

## If sophia-agi MCP is connected

Prefer these tools over guessing:

- `sophia_gate_check` — validate draft before sending
- `sophia_get_attribution` / `sophia_get_record` — lookup canonical records
- `sophia_benchmark_list` + `sophia_benchmark_score` — eval harness

## If inside sophia-agi repo

Also load project skill `/sophia-agi` and run `sophia_validate` before corpus edits.

## Do not

- State legendary authorship as certain fact
- Merge intellectual lineages for narrative convenience
- Invent citations or primary sources
- Skip 中文 summary on teaching-style answers