#!/usr/bin/env bash
# Push the oscillatory-crosspollination branch and open a PR.
# The HEAD commit of branch feat/oscillatory-crosspollination is authored as tomyimkc and
# contains ONLY the 18 agent-authored files (6 tools + 6 tests + 6 docs) — none of your
# in-progress working-tree changes, and not the .obsidian vault or personal notes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
BRANCH=feat/oscillatory-crosspollination
git checkout "$BRANCH"
git push -u origin "$BRANCH"
gh pr create --base main --head "$BRANCH" \
  --title "feat(agi-proof): oscillatory cross-pollination tools (O1-O5)" \
  --body-file agi-proof/oscillatory-crosspollination-2026-07-01/PR-BODY.md
# No gh? After the push, open:
#   https://github.com/tomyimkc/sophia-agi/compare/main...feat/oscillatory-crosspollination
# and paste PR-BODY.md as the description.
