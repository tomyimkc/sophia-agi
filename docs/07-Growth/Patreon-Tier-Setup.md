# Patreon Custom Tier Setup Guide (自訂會員等級)

This document contains everything you need to create / maintain the custom membership tiers on your Patreon page.

**Your Patreon page:** https://www.patreon.com/c/aideveloper_tomyim

## Current Tier Structure (已設定)

These are the **exact tier titles** currently configured on your Patreon (use them as-is for the sync to work perfectly):

1. **Wisdom Seed (智慧種子)** — HKD99 / month
2. **Gatekeeper (守門人)** — HKD299 / month
3. **Verifier (驗證者)** — HKD699 / month
4. **Wisdom Council (智慧議會)** — HKD999 / month

The machine-readable definition lives in `data/patreon/tiers.json`. The sync script uses the `patreon_title` values for grouping and ordering.

## How to Create / Edit Tiers on Patreon

1. Go to your creator dashboard: https://www.patreon.com/portal/membership
2. Click **"Edit tiers"** or **"Tiers & benefits"**.
3. For each tier:
   - Set the **Tier name** exactly as above (e.g. `Wisdom Seed (智慧種子)`)
   - Set the monthly price
   - Paste the **description** and **benefits** below
   - (Optional) Upload a small thematic image
   - Save

## Copy-Paste Content for Each Tier

Use the **Short description** in the "Short description" field on Patreon.

Use the **Long description** + **Benefits** in the main description area.

All content is provided in both **English** and **Traditional Chinese (繁體中文)**.

---

### 1. Wisdom Seed (智慧種子) — HKD99 / month

**Short description (shown in tier card):**
Support the development of honest, provenance-aware AI reasoning. Wisdom before intelligence.

**Long description (English):**
Support the development of honest, provenance-aware AI reasoning. Wisdom before intelligence.

Thank you for supporting Sophia — the Wisdom Gate.

Your contribution helps fund the time and compute needed to build and validate the fail-closed provenance gate, run reproducible benchmarks, and pursue third-party validation — the most important missing piece for credible progress.

**Long description (繁體中文):**
支持誠實且具來源意識的 AI 推理開發。智慧先於智能。

感謝您支持 Sophia — the Wisdom Gate。

您的支持有助於資助建構及驗證 fail-closed 來源閘門、執行可重現的基準測試，以及推動第三方驗證工作 — 這是建立可信進展最重要的缺失一環。

**Benefits (English):**
- Your name will be publicly listed in SPONSORS.md
- Thanked in release notes
- The satisfaction of funding rigorous, source-disciplined AI research

**Benefits (繁體中文):**
- 姓名公開列於 SPONSORS.md
- 在版本發行說明中獲得鳴謝
- 支持嚴謹、來源紀律 AI 研究的滿足感

---

### 2. Gatekeeper (守門人) — HKD299 / month

**Short description (shown in tier card):**
Early access + deeper insight into the work behind the gate.

**Long description (English):**
Early access + deeper insight into the work behind the gate.

You get everything from Wisdom Seed, plus a closer look at the actual development. As a Gatekeeper you help sustain day-to-day progress while receiving early visibility into new tools, verifiers, and benchmarks before public release.

**Long description (繁體中文):**
搶先體驗 + 深入了解閘門背後的工作。

您將獲得智慧種子的所有權益，同時更深入了解實際開發過程。作為守門人，您幫助維持項目的日常進展，並能搶先看到新工具、驗證器和基準測試（在公開發布前）。

**Benefits (English):**
- Everything in Wisdom Seed
- Early access to new tools, scripts and benchmark previews
- Private Patreon posts with deeper technical notes
- Priority consideration for feedback on new features

**Benefits (繁體中文):**
- 包含智慧種子所有權益
- 搶先體驗新工具、腳本及基準測試預覽
- Patreon 私人貼文，提供更深入的技術筆記
- 新功能回饋享有優先考慮

---

### 3. Verifier (驗證者) — HKD699 / month

**Short description (shown in tier card):**
Help decide the direction of future domains and expansions.

**Long description (English):**
Help decide the direction of future domains and expansions.

Verifiers play an active role in shaping what Sophia works on next. In addition to earlier access, you participate in quarterly direction surveys. Your input directly helps decide which domains (philosophy, psychology, history, law, new fields...) should be expanded.

**Long description (繁體中文):**
協助決定未來領域與擴展的方向。

驗證者在塑造 Sophia 下一步工作方向上扮演重要角色。除了較早的存取權限，您還可以參與季度方向調查。您的意見將直接影響下一個應擴展的領域（哲學、心理學、歷史、法律、新領域等）。

**Benefits (English):**
- Everything in Gatekeeper
- Quarterly direction survey — help choose which domains to expand next
- Name mentioned in relevant technical updates and posts
- Occasional private notes on verification work

**Benefits (繁體中文):**
- 包含守門人所有權益
- 季度方向調查 — 協助選擇下一個要擴展的領域
- 在相關技術更新與貼文中提及姓名
- 偶爾收到驗證工作的私人筆記

---

### 4. Wisdom Council (智慧議會) — HKD999 / month

**Short description (shown in tier card):**
Small-group involvement and direct voice in the project.

**Long description (English):**
Small-group involvement and direct voice in the project.

The highest public tier. Wisdom Council members get a seat at the table. You receive everything from lower tiers plus direct interaction with the maintainer through small quarterly discussions. Your perspective carries real weight on the long-term roadmap.

**Long description (繁體中文):**
小組參與及在項目中擁有直接發聲機會。

這是公開等級中最高的階層。智慧議會成員將獲得參與討論的機會。您將享有較低等級的所有權益，並透過每季度小型討論與維護者直接交流。您的觀點對長期路線圖具有實質影響力。

**Benefits (English):**
- Everything in Verifier
- Invitation to small quarterly live discussion / office hours
- Stronger voice on roadmap and domain priorities
- Public credit in major releases and documentation

**Benefits (繁體中文):**
- 包含驗證者所有權益
- 獲邀參與每季度小型線上討論 / 辦公時間
- 對路線圖及領域優先順序擁有更強的發聲權
- 在主要版本及文件中獲得公開致謝

---

## After Creating / Updating the Tiers on Patreon

1. Make sure the tier titles on Patreon exactly match:
   - `Wisdom Seed (智慧種子)`
   - `Gatekeeper (守門人)`
   - `Verifier (驗證者)`
   - `Wisdom Council (智慧議會)`

2. (Optional but recommended) Add at least one test subscriber to each tier.

3. Run the sync locally to pull the latest supporters into the repo:

   ```bash
   python tools/sync_patreon_supporters.py --update --write-json
   ```

4. Verify the output in `SPONSORS.md` and `data/patreon/supporters.json`.

5. (Recommended) Configure the GitHub Action secrets so the list auto-syncs (see `SPONSORS.md` maintainer notes).

## Tier Configuration (machine readable)

The source of truth is:
- `data/patreon/tiers.json`

The sync script reads the `patreon_title` and `ordering` to group supporters correctly and produce nicely ordered output.

**Whenever you change tier titles or pricing on Patreon, also update `data/patreon/tiers.json`.**

## Notes & Integrity

These tiers follow the project's core principles (documented in `SPONSORS.md`):
- No sponsor can buy "truth"
- No sponsor name goes into training data
- Higher tiers can influence *direction*, never *verdicts*

Good luck, and thank you for building real support infrastructure for this work!

---

## Posting GitHub Content to Patreon (reverse sync)

Use `tools/post_to_patreon.py` to push content **from the repo to your Patreon page** (the opposite of supporter syncing).

### Basic usage

```bash
# Preview only (highly recommended first)
python tools/post_to_patreon.py \
  --title "New cluster simulator landed" \
  --file CHANGELOG.md \
  --min-cents 9900 \
  --dry-run

# Actually post (visible to patrons at Wisdom Seed tier and above)
python tools/post_to_patreon.py \
  --title "New cluster simulator landed" \
  --file CHANGELOG.md \
  --min-cents 9900
```

Common patterns:

- `--changelog` — automatically use the top section of CHANGELOG.md
- `--public` — make the post visible to everyone (not just patrons)
- `--min-cents 9900` — only patrons pledging at least HKD99 (Wisdom Seed+)
- `--file docs/some-announcement.md` — any Markdown file in the repo
- `--content "..."` — pass content directly on the command line

The script converts Markdown → basic HTML and uses the same Patreon creator credentials from your `.env`.

### Tips for good posts

- Use `--dry-run` and review the HTML preview before publishing.
- After posting you can still edit the post on Patreon (add images, formatting, etc.).
- For tier-specific announcements, choose the right `--min-cents` value:
  - 0 or omit → all active patrons
  - 9900 → Wisdom Seed and above
  - 29900 → Gatekeeper and above
  - etc.

See the script itself for more examples:

```bash
python tools/post_to_patreon.py --help
```

This lets you announce new GitHub features, releases, or sponsor updates directly to your patrons without leaving the repo workflow.

#### Helper mode for huge supporter expansions (`--supporter-post`)

When your supporter list gets very large, use this dedicated helper:

```bash
# Preview the generated post (safe)
python tools/post_to_patreon.py --supporter-post --dry-run

# Post a thank-you update visible to all active patrons
python tools/post_to_patreon.py --supporter-post

# Tier-specific (example: only Wisdom Council and above)
python tools/post_to_patreon.py --supporter-post --min-cents 99900
```

**What it does automatically:**
- Reads your latest `data/patreon/supporters.json` (make sure you ran the sync first)
- Builds a clean thank-you post using your exact bilingual tier titles
- For small tiers → lists the names
- For large tiers (your "huge expand") → shows a sample + count + link to the full list in GitHub SPONSORS.md
- Adds a nice intro + call-to-action
- Picks a sensible title (e.g. "Thank You to Our Patreon Supporters — 2026-06")

You can override the title with `--title "My custom title"`.

This is the best tool for regularly feeding content back to a growing supporter base.