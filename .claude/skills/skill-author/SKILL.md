---
name: skill-author
description: >
  Run WHENEVER you hit a non-trivial issue a future session would repeat — a bug whose root cause
  was non-obvious, a repo footgun, a surprising rule/guardrail, a wasted GPU/CI cycle, a wrong
  attribution you almost shipped, or any "I wish I'd known that earlier" moment. Use this to WRITE a
  new skill or UPDATE an existing one so the lesson auto-triggers next time, and to log the issue in
  the failure ledger. Trigger even if the fix felt obvious in hindsight, even mid-task, and even if
  you think "it's just a one-off" — recurring waste in this multi-agent repo is the #1 cost. Also
  fires on "capture this", "make a skill for this", "so we don't hit this again", "/skill-author".
metadata:
  short-description: "Turn an issue you just hit into a new/updated, auto-triggering skill (verifier-gated when checkable)"
---

# Skill author — issue → skill (this repo)

**An issue you don't capture, the next session repeats.** This skill turns a problem you just hit
into a durable, auto-triggering skill. It is the human-readable half of the self-improving skill loop
described in `docs/09-Agent/Skill-RSI-Thesis-and-Review.md`.

## When to invoke (be eager)

- You debugged something whose root cause was non-obvious, or you tripped a repo guardrail.
- A command/test/CI/GPU run failed and the reason is reusable knowledge.
- You almost shipped an overclaim, a wrong attribution, or a stale-snapshot git mistake.
- The user says "capture this", "make a skill for this", "so we don't hit this again".

Invoke even when the lesson feels obvious, even mid-task, even if it seems like a one-off.

## Step 1 — pick the layer (this decides the format)

| The lesson is… | Layer | Where it goes |
|---|---|---|
| A **process / discipline** rule ("before X do Y", "this repo's footgun is Z") | **A — Agent Skill** | new/updated `.claude/skills/<name>/SKILL.md` (this format) |
| A **checkable classification** ("is this output an attribution trap / destructive intent / unsafe claim?") | **C — Skill Forge** | a spec for `tools/sophia_skill_forge.py` so it passes the **verifier gate** |
| A **harness workflow** the Sophia agent should follow | **B — registry** | `skills/registry/<name>.json` (`agent/skills.py` schema) |

If it is checkable, **prefer Layer C** — a skill that ships through `synthesize_gate` is verifier-gated
and cannot overclaim. Never hand-author a "detector" skill when the forge can earn it on held-out data.

## Step 2 — search before you write (de-dup keeps triggering sharp)

1. List existing skills: `.claude/skills/*/SKILL.md`, `skills/registry/*.json`,
   `skills/registry/forge_index.json`.
2. If one already covers this, **UPDATE it** — sharpen the `description`/`whenToUse` triggers or add a
   row to its trap/workflow table. A near-duplicate skill *weakens* triggering (the router splits
   votes). Updating beats adding.

## Step 3 — write it with trigger discipline (so it actually fires)

The model only sees the `description` at selection time (progressive disclosure), so the description
**is** the trigger. Follow the gold standard in `.claude/skills/git-discipline/SKILL.md`:

- First two sentences: **exact phrasings + concrete actions/verbs** that should fire it.
- Name the **negative-override** cases — "even if it looks unnecessary", "before X not after".
- Include the **slash alias** (`/<name>`) and any MCP tool names.
- One distinctive token not shared with sibling skills (raises precision).

For a Layer-C forge skill instead, write a spec JSON and run it:

```bash
python tools/sophia_skill_forge.py spec.json --register-smoke
# promoted only if the synthesized verifier clears held-out validation; a rejection is a valid outcome.
```

## Step 4 — record the issue as evidence

Append to `agi-proof/failure-ledger.md` (or the structured `agi-proof/failures.jsonl` if present): the
issue, the skill id you wrote/updated, and — for a forge skill — the promotion verdict. **A rejected
forge skill is still logged**: "no skill shipped, verifier failed validation." Failures are claim
evidence here; do not hide them.

## Step 5 — keep it plaintext if it carries no IP

Process skills must stay readable to a fresh web session with no git-crypt key. If your new skill is a
process/discipline skill, add a `.gitattributes` exemption next to the existing ones:

```
.claude/skills/<name>/** !filter !diff
```

IP skills (corpus/training/infra internals) stay encrypted — do **not** exempt those.

## Do not

- Do not weaken the verifier gate to make a checkable skill "pass" — route rejections to the ledger.
- Do not add a near-duplicate skill when an update would do.
- Do not write a vague `description` ("helps with debugging") — it will never trigger.
- Do not overclaim in the skill body; `tools/lint_claims.py` still applies.
