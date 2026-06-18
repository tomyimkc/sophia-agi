# GitHub Pages setup (one-time)

Workflow `.github/workflows/pages.yml` builds `web/` and pushes to the **`gh-pages`** branch.

## Why gh-pages branch?

If **Deploy thesis site** failed with:

`Failed to create deployment (status: 404) ... Ensure GitHub Pages has been enabled`

GitHub Actions “deploy-pages” requires **Source: GitHub Actions** in Settings. Many repos only show **Deploy from a branch** until Pages is enabled — the `gh-pages` branch method avoids that 404.

---

## Steps (your clicks)

### 1. Workflow permissions

**Settings → Actions → General → Workflow permissions**

- Select **Read and write permissions** → **Save**

### 2. Run deploy workflow

**Actions → Deploy thesis site → Run workflow** (branch `main`)

Wait for green checkmark. This creates/updates the **`gh-pages`** branch with your site files.

### 3. Enable Pages (branch source)

**Settings → Pages → Build and deployment**

| Field | Value |
|-------|--------|
| **Source** | Deploy from a branch |
| **Branch** | `gh-pages` |
| **Folder** | `/ (root)` |

Click **Save**.

### 4. Open site

After 1–3 minutes:

**https://tomyimkc.github.io/sophia-agi/**

---

## Verify

- Abstract shows version + training example count
- Chapter V shows leaderboards
- Chapter VII falls back to CLI hints (static Pages has no `/api/ask`)

## Live agent (local)

```bash
python tools/serve_web.py
# http://127.0.0.1:8765
```

---

## Optional: GitHub Actions source later

If your Settings → Pages later shows **Source: GitHub Actions**, you can switch — not required for the thesis site to work.