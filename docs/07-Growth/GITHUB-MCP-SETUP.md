# GitHub MCP setup (Cursor / Grok)

Lets the agent create issues, read repos, and manage `tomyimkc/sophia-agi` without `gh` CLI.

> **Do not commit your PAT.** Replace `YOUR_GITHUB_PAT` locally only.

---

## 1. Create a GitHub token

**Fine-grained (recommended):** https://github.com/settings/personal-access-tokens/new

| Field | Value |
|-------|--------|
| Resource owner | `tomyimkc` |
| Repository access | Only `sophia-agi` (or all repos) |
| Permissions | **Issues** Read+write, **Contents** Read+write, **Metadata** Read, **Actions** Read |

**Classic alternative:** https://github.com/settings/tokens/new — scopes: `repo`, `workflow`

Copy the token (`github_pat_...` or `ghp_...`).

---

## 2. Configure MCP (manual — avoid expired install links)

**Do not use** one-click install deeplinks if you see *"Unknown or expired link"* — edit JSON by hand.

### Option A — This repo only

File: `sophia-agi/.cursor/mcp.json` (already scaffolded)

Replace `YOUR_GITHUB_PAT` with your token.

### Option B — All projects (global)

File: `C:\Users\tomyim\.cursor\mcp.json` (create if missing)

```json
{
  "mcpServers": {
    "github": {
      "url": "https://api.githubcopilot.com/mcp/x/issues",
      "headers": {
        "Authorization": "Bearer YOUR_GITHUB_PAT"
      }
    }
  }
}
```

### Toolset URLs (pick one)

| Need | URL |
|------|-----|
| Issues only (launch tasks) | `https://api.githubcopilot.com/mcp/x/issues` |
| Issues + repos | `https://api.githubcopilot.com/mcp/` (default) |
| Read-only | append `/readonly` to any URL |

---

## 3. Cursor UI (preferred for token)

1. **Cursor Settings → Tools & Integrations → MCP**
2. Find **github** (or add server)
3. Click **pencil** icon → paste PAT (not stored in git)
4. **Restart Cursor completely**

---

## 4. Verify

In chat, ask:

```
List open issues on tomyimkc/sophia-agi
```

Green dot next to `github` in MCP settings = connected.

---

## 5. What agent can do after connect

- Create GF-10 / GF-20 / GF-30 issues
- Update repo About/topics via API
- Check Actions workflow runs
- Open PRs (add `pull_requests` toolset URL)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Unknown or expired link | Skip deeplinks; edit `mcp.json` manually |
| 401 / auth failed | Regenerate PAT; check Issues permission |
| MCP not in tool list | Full Cursor restart |
| Grok CLI session | Open `sophia-agi` as workspace; MCP loads from `.cursor/mcp.json` |

## Docker local server (optional)

Requires Docker Desktop:

```json
"github": {
  "command": "docker",
  "args": ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
  "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_PAT" }
}
```