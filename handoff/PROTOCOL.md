# Cloud <-> Mac git coordination channel

This branch (`mac-handoff`) is the **message bus** between the cloud Claude session and the Mac Studio
operator (Claude Code / OpenClaw). There is NO live link between them — they coordinate **only** by
committing + pushing to this branch and reading each other's files on pull. `canClaimAGI` stays false.

## Directories (each side writes ONLY its own dir -> no merge conflicts)
- `handoff/from-mac/`  -- the **Mac** writes here (status, results, an updated hook, questions).
- `handoff/to-mac/`    -- the **cloud** writes here (priorities, acks, answers).
- `handoff/inbox-from-mac/` / `handoff/inbox-to-cloud/` -- optional append-only message drops
  (one small JSON file per message, named `<seq>-<topic>.json`); never edit the other side's files.

## Heartbeat / status (the "what's going on" signal)
- The Mac OVERWRITES `handoff/from-mac/STATUS.json` every work cycle (schema in that file). This is how
  the cloud knows what the Mac is doing, the Spark/GPU state from the Mac's view, and any blockers.
- The cloud overwrites `handoff/to-mac/STATUS.json` with current priorities + acks.
- Cloud reads on its 6h heartbeat trigger and on demand; the Mac reads each cycle. Latency = push cadence.

## Commit rules
- Commit message prefix: `handoff(mac): ...` or `handoff(cloud): ...`.
- Pull before you push; if the other side touched the SAME file (shouldn't happen — separate dirs),
  keep both and reconcile in your own dir. Never force-push this branch.
- Push small + often. Status is a snapshot (overwrite); messages are append-only files.

## Secret handling (HARD RULE)
- This repo is PUBLIC. **Never** commit plaintext secrets (tokens, keys, the Telegram bot token).
- The OKF decision layer + personal hooks stay out of the public repo. If the Mac needs to ship an
  updated hook through here, push it **encrypted** (git-crypt / age) as `from-mac/okf_hook.enc`; the
  cloud will note its presence + sha and forward to the owner (cloud cannot decrypt without the key).
- If you must reference a secret, reference its NAME/location, never its value.

## What the cloud will act on automatically (via the 6h heartbeat)
- Read `from-mac/STATUS.json`; if a job FAILED, root-cause + log to the ledger.
- Keep the Spark idle-backlog fed (bounded cert/eval only; never `--run-train` without owner approval).
- Forward any `from-mac/okf_hook.enc` presence to the owner.
- It will NOT push to the work branch or open PRs without cause; it reports status to the owner.
