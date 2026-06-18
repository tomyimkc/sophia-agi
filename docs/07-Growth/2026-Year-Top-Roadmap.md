# 2026 Roadmap — Toward category-leading repo

Goal: make **Sophia AGI** the default name for *provenance-aware reasoning* — cited in papers, listed in eval harnesses, and starred because it solves a problem every LLM user has felt.

> **Reality check:** "Top repo of the year" on GitHub usually means **10k–50k+ stars** and sustained press. Sophia wins by **owning a category** (source discipline / attribution benchmark), not by claiming "AGI."

---

## Where you are now (v0.4.1)

| Done | Item |
|------|------|
| ✅ | 4 active domains, 19 training examples, HF corpus live |
| ✅ | Per-domain leaderboards; reference + Claude 100% |
| ✅ | Three-path agent (advisor · repo · life) |
| ✅ | Thesis web UI + UI council + GitHub Pages deploy |
| ✅ | CI, validation, benchmark harness |

**Gap:** visibility, corpus scale, external citations, contributor funnel.

---

## North-star metrics (Dec 2026)

| Metric | Minimum viable | Category leader |
|--------|----------------|-----------------|
| GitHub stars | 2,000 | 10,000+ |
| HF dataset downloads | 500/mo | 5,000/mo |
| External model runs on leaderboard | 8 models | 20+ models |
| Training examples | 100 | 500+ |
| Active contributors | 5 | 25+ |
| Citations / mentions | 3 blogs or papers | Listed in 2 eval suites |

Track weekly in GitHub Discussions → **Metrics** thread.

---

## Phase A — Launch window (Jun–Aug 2026)

**Objective:** First wave of legitimate attention. Thesis site is the landing page.

### Week 1–2 (immediate)

- [ ] Enable GitHub Pages (Settings → Pages → GitHub Actions) — **your click**
- [x] Tag `v0.4.2` on GitHub; link thesis URL in README
- [ ] Hacker News Show HN — draft ready: `docs/07-Growth/launch/show-hn-draft.md`
- [ ] Reddit — drafts ready: `docs/07-Growth/launch/reddit-localllama-draft.md`
- [ ] Post on X/LinkedIn: one thread on **lineage-merge failure** + leaderboard proof
- [x] GF-01–05 implemented in corpus; community issues GF-10+ via `tools/create_github_issues.py`

### Week 3–4

- [ ] Run GPT-4o, Gemini, Grok, Llama API on all 4 domains → update leaderboards
- [ ] Blog post (Medium / dev.to / personal site): *"We benchmarked attribution discipline"*
- [ ] Add **comparison table** to thesis Chapter V (auto from manifest)
- [ ] Discord or GitHub Discussions: `#benchmark-submissions`, `#corpus-contributions`

**Exit criteria:** 200+ stars, 3+ external benchmark entries, 2+ PRs from strangers.

---

## Phase B — Authority (Sep–Nov 2026)

**Objective:** Researchers and OSS devs treat Sophia as the attribution benchmark.

### Corpus & benchmark

- [ ] 50+ training examples (10 per domain minimum)
- [ ] 30+ philosophy attributions in `data/attributions.json`
- [ ] 10+ dispute notes in `docs/04-Disputes/`
- [ ] 20+ benchmark cases per domain (not just traps — edge cases)
- [ ] Publish `docs/WHITEPAPER.md` (8–12 pages, PDF export optional)

### Integrations (high leverage)

- [ ] **`pip install sophia-gate`** or documented LangChain / LlamaIndex plugin
- [ ] PR to **lm-evaluation-harness** or **Open LLM Leaderboard** subset (even 1 task)
- [ ] Hugging Face **Dataset Viewer** + leaderboard dataset card
- [ ] Optional: hosted demo (Fly.io / Railway) for `/api/ask` — Pages stays static

### Community

- [ ] Monthly **"Attribution Trap of the Month"** issue → community PR
- [ ] Contributor ladder: reviewer → domain maintainer (philosophy, psych, etc.)
- [ ] Video (5 min): thesis site walkthrough + one failed GPT answer vs Sophia gate

**Exit criteria:** 1k+ stars, 5+ contributors, 1 external eval listing or paper citation.

---

## Phase C — Category ownership (Dec 2026)

**Objective:** "Source discipline" and "Sophia benchmark" become searchable phrases.

- [ ] 100+ training examples; nightly corpus export to HF
- [ ] **Self-correcting loop:** failed eval → auto-draft training example → human review
- [ ] Science + law domain pilots (schema only + 5 cases each)
- [ ] Annual report page on thesis site: *Sophia Attribution Report 2026*
- [ ] Conference submission: NeurIPS workshop / EMNLP system demo (provisional)
- [ ] Partnership pitch: one fine-tuning shop or RAG vendor lists Sophia gate in docs

**Exit criteria:** 2k–10k stars depending on viral luck; HF downloads growing MoM; inbound PRs without asking.

---

## Ongoing weekly rhythm (non-negotiable)

| Day | Action |
|-----|--------|
| Mon | Check stars, HF stats, open issues; triage 15 min |
| Wed | Merge 1 corpus or benchmark PR (yours or community) |
| Fri | One public artifact: tweet, issue, leaderboard update, or doc |
| Monthly | Re-run external models; refresh manifest + thesis stats |

---

## What *not* to do (Philosophy lineage voice)

- Do not rebrand as "AGI solved" — wisdom / provenance is the wedge
- Do not merge domains in marketing (keep philosophy ≠ religion ≠ pop psych)
- Do not launch without benchmark evidence on the thesis site
- Do not ignore 中文 audience — bilingual traps are your differentiator

---

## Priority stack rank (if time is limited)

1. **HN + thesis URL** (biggest single spike)
2. **External model leaderboard** (credibility)
3. **Good First Issues → real issues** (contributor funnel)
4. **Whitepaper + eval harness listing** (long-term authority)
5. **pip package / plugin** (developer adoption)

---

## Links

- [90-Day Launch Playbook](90-Day-Launch.md) — tactical checklist (update Phase 1 ✅ items)
- [Open Intelligence Plan](../06-Roadmap/Open-Intelligence-Plan.md) — technical phases
- [UI Council Decisions](../10-Web/UI-Council-Decisions.md) — public face of the project