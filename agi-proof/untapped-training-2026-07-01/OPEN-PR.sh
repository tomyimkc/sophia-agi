#!/usr/bin/env bash
# Push the candidate-tools branch and open a PR.
# The commit (d6bd80a8) already exists locally on branch feat/agi-proof-candidate-tools.
# It contains ONLY the 28 agent-authored files (10 tools + 9 tests + 9 docs) — none of your
# 383 in-progress working-tree changes, and not the .obsidian vault or personal notes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BRANCH=feat/agi-proof-candidate-tools
git checkout "$BRANCH"

# 1) push the branch
git push -u origin "$BRANCH"

# 2) open the PR — needs the GitHub CLI (https://cli.github.com), authed via `gh auth login`
gh pr create \
  --base main \
  --head "$BRANCH" \
  --title "feat(agi-proof): candidate evidence + untapped training-signal tools" \
  --body-file agi-proof/untapped-training-2026-07-01/PR-BODY.md

# If you don't use gh: after the push, open
#   https://github.com/tomyimkc/sophia-agi/compare/main...feat/agi-proof-candidate-tools
# and paste PR-BODY.md as the description.
