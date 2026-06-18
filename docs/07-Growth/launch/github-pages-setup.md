# GitHub Pages setup (one-time)

The workflow `.github/workflows/pages.yml` deploys `web/` on every push to `main`.

## Steps (requires repo admin — **your click**)

1. Open https://github.com/tomyimkc/sophia-agi/settings/pages
2. **Build and deployment → Source:** select **GitHub Actions** (not "Deploy from branch")
3. Push to `main` or run workflow manually: **Actions → Deploy thesis site → Run workflow**
4. Wait ~2 minutes; URL appears on Settings → Pages

**Expected URL:** https://tomyimkc.github.io/sophia-agi/

## Verify

- Abstract loads with version + training example count
- Chapter V shows 4 leaderboards
- Chapter VII "Ask Sophia" shows CLI fallback (static Pages has no `/api/ask`)

## Live agent (optional)

```bash
python tools/serve_web.py
# http://127.0.0.1:8765 — full /api/ask
```

Hosted agent (Fly.io / Railway) is a Phase B item — needs your deployment preference.