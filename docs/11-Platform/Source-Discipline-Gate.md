# Source-Discipline Gate (Sophia → OpenClaw)

*Added in v0.7.2.* Sophia's "never merge lineages" rule, enforced on an external
[OpenClaw](https://github.com/openclaw/openclaw) gateway's agent replies.

Full design: [`docs/superpowers/specs/2026-06-20-source-discipline-gate-design.md`](../superpowers/specs/2026-06-20-source-discipline-gate-design.md).

## What it is

`provenance_faithful` (`agent/verifiers.py`) is Sophia's machine-checked source-discipline
verifier — a ~2 ms local-regex check (no model call) that fails text asserting an
attribution forbidden by a record's `doNotAttributeTo` ("Confucius wrote the Dao De Jing")
while passing the correct debunk ("Confucius did **not** write the Dao De Jing").

This feature exposes that verifier as a tiny CLI so an OpenClaw plugin can call it across
processes — making OpenClaw agent replies obey Sophia's discipline.

## The Sophia side (this repo)

`tools/source_discipline_cli.py` — reads draft text on **stdin** (or `--text`), prints
`{"passed": bool, "reasons": [...], "violations": [...]}`. Exit `0` on a completed check;
nonzero only on internal error (so a caller can fail open).

```bash
echo "Confucius wrote the Dao De Jing." | python tools/source_discipline_cli.py
# {"passed": false, "reasons": ["forbidden attribution asserted: confucius -> dao_de_jing"], "violations": ["confucius -> dao_de_jing"]}
```

Dependency-free, offline, `okf/`-independent. Tested in `tests/test_source_discipline_cli.py`.

## The OpenClaw side (outside this repo)

A plain-JS OpenClaw plugin at `~/.openclaw/plugins/sophia-source-discipline/` registers a
`before_agent_finalize` hook that:

1. spawns this CLI with the draft reply (`event.lastAssistantMessage`) on stdin;
2. on `passed: false`, returns `{action: "revise", retry: {...}}` so the producing model
   makes one corrected pass (bounded by `maxAttempts`);
3. **fails open with a loud log** if the checker is unreachable (availability over a missed
   narrow check), so a downed checker never stalls replies.

It requires `plugins.entries.sophia-source-discipline.hooks.allowConversationAccess = true`
and a `config.sophiaRepoPath` pointing at this repo.

## Honest limits

Output-gating only; ~31 curated records (high precision, **not** a general hallucination
catcher); the `revise` correction needs a working model on the gateway. Adds nothing to,
and claims nothing about, the AGI-candidate proof package.
