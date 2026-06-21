# Issue/PR triage & project automation

This repo auto-triages issues and PRs via [`.github/workflows/triage.yml`](workflows/triage.yml).

## What runs automatically (no setup)

- **Issue labeling** — `github/issue-labeler` applies area labels (`corpus`,
  `benchmark`, `tooling`, `documentation`, `psychology`, `history`, `religion`,
  `bug`, `enhancement`) by matching regexes in
  [`.github/issue-labeler.yml`](issue-labeler.yml) against the title/body.
- **PR labeling** — `actions/labeler` applies the same area labels by changed paths
  per [`.github/labeler.yml`](labeler.yml).
- **Dependabot** — weekly grouped dependency PRs from [`.github/dependabot.yml`](dependabot.yml).

## Optional: auto-add new issues to a Project board

The `add-to-project` job is dormant until you configure two things:

1. **Create a Projects v2 board** (user or org level):
   ```bash
   gh auth refresh -s project,read:project       # one-time: grant project scope
   gh project create --owner tomyimkc --title "Sophia AGI roadmap"
   gh project list --owner tomyimkc              # note the board URL
   ```
2. **Wire it to the workflow:**
   - Repo variable `PROJECT_URL` = the board URL
     (`gh variable set PROJECT_URL --body "https://github.com/users/tomyimkc/projects/N"`).
   - Repo secret `ADD_TO_PROJECT_PAT` = a fine-grained PAT with **read/write access to
     your Projects** (`GITHUB_TOKEN` cannot modify user/org projects)
     (`gh secret set ADD_TO_PROJECT_PAT`).

Once both are set, every newly opened issue is added to the board automatically.
Projects v2 has built-in workflows (Settings → Workflows) to move cards on
status change and close them when a linked PR merges.
