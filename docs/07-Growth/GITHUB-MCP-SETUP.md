# GitHub MCP setup (Cursor / Grok)

Lets the agent create issues, read repos, and manage `tomyimkc/sophia-agi` without `gh` CLI.

> **Do not commit your PAT.** Replace `YOUR_GITHUB_PAT` locally only.

---

## 1. Create a GitHub token

Official permission names and access levels: [Permissions required for fine-grained personal access tokens](https://docs.github.com/en/rest/authentication/permissions-required-for-fine-grained-personal-access-tokens) and [Managing your personal access tokens → Repository permissions](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token#repository-permissions).

In the token UI, each permission is a **Repository permissions** row. The dropdown is **No access**, **Read-only** (`read`), or **Read and write** (`write`). `write` always includes `read`.

### Fine-grained (recommended)

Create: https://github.com/settings/personal-access-tokens/new

Pre-filled (issues MCP minimum — still pick repo `sophia-agi` manually):

https://github.com/settings/personal-access-tokens/new?name=Sophia+AGI+MCP+issues&description=Cursor+GitHub+MCP+issues+toolset&target_name=tomyimkc&issues=write

| Field | Value |
|-------|--------|
| Resource owner | `tomyimkc` |
| Repository access | **Only select repositories** → `tomyimkc/sophia-agi` |
| Expiration | Your choice (90 days is fine) |

#### Repository permissions — pick **exact UI names**

| Permission (UI label) | Issues-only MCP (`/mcp/x/issues`) | Default MCP (`/mcp/`) | REST `create_github_issues.py` |
|-----------------------|-----------------------------------|-------------------------|--------------------------------|
| **Metadata** | **Read-only** (required; no write option) | **Read-only** | **Read-only** |
| **Issues** | **Read and write** | **Read and write** | **Read and write** |
| **Contents** | No access | **Read-only** (read files/README) | No access |
| **Actions** | No access | **Read-only** (check CI runs) | No access |
| **Pull requests** | No access | **Read and write** (if opening PRs) | No access |
| **Administration** | No access | No access | **Read and write** only if setting repo **topics** (`PUT /repos/.../topics`) |

**Minimum for your launch task (create GF-10…GF-40 issues):**

1. **Metadata** → **Read-only**
2. **Issues** → **Read and write**

That covers `POST /repos/tomyimkc/sophia-agi/issues` and applying labels (labels are under **Issues**, not a separate permission).

Leave every other repository permission at **No access** unless you expand the MCP toolset URL.

### Classic alternative (simpler, broader)

https://github.com/settings/tokens/new — check scope **`repo`** only (covers issues + metadata). Add **`workflow`** only if you need Actions status via classic token.

Copy the token (`github_pat_...` or `ghp_...`). **Do not commit it.**

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
| 401 / auth failed | Regenerate PAT; ensure **Issues → Read and write** + **Metadata → Read-only** on `sophia-agi` |
| 403 `X-Accepted-GitHub-Permissions` | See [permissions doc](https://docs.github.com/en/rest/authentication/permissions-required-for-fine-grained-personal-access-tokens); raise the listed permission |
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