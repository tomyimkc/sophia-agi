# Domain expansion questionnaire

**Status:** Answered 2026-06-18 by @tomyimkc

---

## A. Psychology — **All three from the start**

- **Scope:** Clinical / DSM, cognitive science, **and** pop-psych myth-busting (mixed corpus)
- **Implication:** Every psychology record should tag `subfield`: `clinical` | `cognitive` | `pop_myth`
- **Top traps (seed list):** Freud misattribution, left/right brain myth, "chemical imbalance" simplification, Stockholm syndrome misuse, Maslow everywhere

## B. History — **Global**

- **Geography:** Worldwide events and myths — not limited to one region
- **Implication:** Records require `region` + `dateConsensus` + `primarySource` fields
- **Top myths (seed list):** Marco Polo invented pasta, "medieval people thought earth was flat", Napoleon height myth, Great Wall visible from space, Vikings horned helmets

## C. Religion — **Multi-tradition**

**In scope first:** Buddhism, Daoism, Christianity, Islam, Confucian ritual religion (祭祀 / 儒家禮教)

**Claim handling — Council / debate mode**

When theology and history conflict, the teacher agent uses **structured debate**, not a single blended answer:

1. **Council panel** — name the traditions or scholarly schools represented
2. **Theological voice** — "Within tradition X, the claim is…"
3. **Historical-critical voice** — "Primary sources and scholarship suggest…"
4. **Debate mode** — surface the tension explicitly; do not merge into one false consensus
5. **中文 summary** — same structure, tradition labels preserved

## D. Cross-domain

| Question | Answer |
|----------|--------|
| Benchmark structure | **Per-domain leaderboards** + combined rollup |
| Language | **Always bilingual** EN + 中文 (all domains) |
| Training agents | Domain-tagged examples; shared source-discipline system prompt |
| OSS | MIT, public corpus |

---

## Next implementation steps (from answers)

1. Split leaderboards: `benchmark/results/leaderboard-{domain}.json`
2. Populate starter records in psychology / history / religion JSON files
3. Add `subfield` to psychology schema usage
4. Add religion `answerMode: council_debate` to training metadata
5. Expand benchmark with first cross-domain trap questions