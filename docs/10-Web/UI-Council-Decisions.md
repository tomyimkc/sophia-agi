# UI Council — Web design decisions

**Council panel (all seated):** UX Research · Design Systems · Accessibility · Engineering · Philosophy lineage voice

## Question

How should the Sophia AGI public website look and behave?

---

**UX Research voice:** The audience is researchers, OSS contributors, and skeptical ML engineers. They need **thesis-level depth** first — not a landing-page slogan. Structure as: Abstract → Problem → Framework → Method → Results → Agent. Scannable chapter nav with persistent table of contents.

**Design Systems voice:** Visual language = **scholarly monograph**, not startup SaaS. Serif body (literature), sans UI chrome. Palette: ivory paper `#f8f5f0`, ink `#1a1a2e`, accent bronze `#9a7b4f`. Council panels get bordered “seats” with named headers — reuse religion council UX pattern for **UI decisions themselves** (meta-council).

**Accessibility voice:** 18px base, 1.65 line-height, skip link, focus rings, bilingual text without breaking CJK line height. Charts must have text labels, not color-only.

**Engineering voice:** Static `web/` ships on GitHub Pages. Optional `tools/serve_web.py` adds `/api/ask` for live agent (advisor | repo | life). Leaderboards load from embedded `manifest.json` — regenerate via `tools/build_web_data.py`.

**Philosophy lineage voice:** The site must **practice source discipline** — cite repo paths, show benchmark evidence, never claim “AGI” without defining epistemic gate. Tagline: *Wisdom before intelligence* (σοφία).

**Debate / tension:** Thesis depth vs. mobile brevity → **progressive disclosure**: executive abstract on load, chapters expandable, agent panel collapsed on small screens.

## Decision

| Element | Ruling |
|---------|--------|
| Layout | Single-page thesis with chapter nav |
| Council UI | Visible panel documenting these decisions |
| Agent | Three tabs: Advisor · Repo · Life |
| Data | Live leaderboards + version from manifest |
| Deploy | `python tools/serve_web.py` locally; `web/` static for Pages |

**中文：** 網站採論文式章節結構；視覺為學術手稿風；理事會面板呈現設計決策；三路代理可選 API 串接。