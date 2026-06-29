# Claude Code setup for Sophia-AGI

This folder configures how Claude Code (CLI **and** Claude-Code-on-the-web) operates in this
repo. It exists so a fresh session bootstraps from hard-won lessons instead of starting blind.

## What's here

```
.claude/
  settings.json              # hooks (SessionStart unlock+orient, PreToolUse git guard)
  hooks/
    session_start.sh         # auto-unlock git-crypt + print orientation at session start
    git_write_guard.sh       # advisory nudge before push/merge/rebase (never blocks)
  skills/
    git-discipline/          # PLAINTEXT  — pre-flight before any git write / merge / CI work
    ci-artifact-drift/       # PLAINTEXT  — regenerate+verify generated artifacts & gates pre-push
    session-handover/        # PLAINTEXT  — bootstrap from / write the handover docs
    sophia-agi/              # encrypted  — corpus + epistemic gate + training substrate
    runpod-mcp/              # encrypted  — manage RunPod via the official MCP server
    wisdom-gpu-prebaked/     # GPU cost-guard runbook (anti-credit-burn)
```

The three **process** skills are plaintext on purpose (see "Encryption" below); the **repo/IP**
skills are git-crypt-encrypted.

## How skills auto-trigger (the answer to "how do agents trigger these automatically")

There are two distinct mechanisms — use both:

1. **Model-chosen (skills).** A skill is *not* a script that fires itself. At each turn the agent
   reads every skill's frontmatter `description` and invokes the one whose triggers match the
   task. **Auto-trigger quality == description quality.** Each skill here front-loads concrete
   trigger verbs and situations ("Run BEFORE any git write… commit, push, branch, rebase, merge…",
   "even if the conversation already contains a diagnosis"). To add a skill that reliably
   self-selects: name the exact user phrasings and actions in the first two sentences of the
   description, including the "even if it looks unnecessary" cases.

2. **Deterministic (hooks in `settings.json`).** Hooks run on events regardless of what the model
   decides — this is true automatic execution:
   - `SessionStart` → `hooks/session_start.sh`: unlocks git-crypt and injects the orientation
     (read-first files, live git snapshot, guardrails) into context every session.
   - `PreToolUse` (matcher `Bash`) → `hooks/git_write_guard.sh`: when a command is about to
     push/merge/rebase, injects a one-line reminder to run the `git-discipline` pre-flight.
     It is **advisory** (always exits 0); to make it *block* until the pre-flight passes, change
     the hook to exit non-zero on the unsafe case.

   Hooks are how you guarantee discipline fires even when the model forgets. The skills carry the
   depth; the hooks guarantee the trigger.

## Encryption (git-crypt) and why web sessions used to start blind

`.gitattributes` encrypts `.claude/skills/**`, `AGENTS.md`, `CONTRACT.md`, `.grok/**`,
`docs/superpowers/**`. On Claude-Code-on-the-web, git-crypt is **not** unlocked by default, so
those files arrive as ciphertext and the skills were effectively dead. Two fixes:

- **Auto-unlock.** Export the symmetric key once from a local unlocked clone and store it as an
  **environment secret** so every session self-unlocks:
  ```bash
  git-crypt export-key /tmp/k && base64 -w0 /tmp/k   # macOS: base64 -i /tmp/k
  ```
  Add the output as the `GITCRYPT_KEY_B64` environment secret in your web environment
  (https://code.claude.com/docs/en/claude-code-on-the-web). `session_start.sh` decodes it,
  installs git-crypt if missing, and runs `git-crypt unlock`. Locally, if already unlocked it's a
  no-op. (This key decrypts all encrypted files — treat it like a password; never commit it.)
- **Plaintext process skills.** `git-discipline`, `ci-artifact-drift`, `session-handover` carry no
  IP and are exempted from encryption in `.gitattributes`, so they work even with no key.

> Note: `skills/wisdom-gpu-prebaked/SKILL.md` is currently committed **unencrypted** despite the
> `.claude/skills/**` rule (added before the rule / never re-staged). It's only a cost-guard
> runbook, but if you want it private, re-stage it while unlocked: `git rm --cached
> .claude/skills/wisdom-gpu-prebaked/SKILL.md && git add .claude/skills/wisdom-gpu-prebaked/SKILL.md`.

## MCP servers (`.mcp.json` at repo root)

| Server | Command | Needs |
|---|---|---|
| `sophia-agi` | `python sophia_mcp/server.py` (`PYTHONPATH=.`) | repo Python deps; tools `sophia_validate`, `sophia_gate_check`, `sophia_benchmark_*` |
| `runpod` | `npx -y @runpod/mcp-server@latest` | `RUNPOD_API_KEY` env (set it as a web environment secret, **not** only a GitHub Actions secret) |

After setting `RUNPOD_API_KEY`, restart the session so the server picks it up; verify by listing
pods (`mcp__runpod__*`). See the `runpod-mcp` skill for detail. RunPod **GPU training** jobs still
go through GitHub Actions, never local SSH (see `AGENTS.md` / `wisdom-gpu-prebaked`).

## Environment secrets to set on the web environment

| Secret | Purpose |
|---|---|
| `GITCRYPT_KEY_B64` | auto-unlock the encrypted skills/docs each session |
| `RUNPOD_API_KEY` | enable the `runpod` MCP server (interactive pod management) |

## Relationship to other agent tools

`.agents/skills/` (git-operations, multi-agent-merge-preflight, pr-merge-verification) holds the
canonical, fuller playbooks + runnable scripts, shared with Cursor/GLM/etc. The `git-discipline`
skill is the Claude-Code entry point that routes to those scripts — single source of truth, no
duplicated logic. `.cursor/` and `.grok/` are the equivalents for those tools.
