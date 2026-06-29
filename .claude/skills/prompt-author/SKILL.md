---
name: prompt-author
description: >
  Run WHENEVER you are about to write a prompt that another agent (or a future session) will act
  on — a task handoff, a PR description, a sub-agent brief, a z.ai/GLM/Copilot assignment, a /loop
  or workflow prompt, or a session-handover. Use it so the prompt is PRECISE and machine-checkable:
  it states how "done" is verified, bounds its scope, gives an abstention/failure path, grounds
  itself in concrete artifacts, and makes no overclaim. The same claim -> verify -> accept/abstain
  discipline Sophia applies to facts, applied to prompts. Also fires on "write a prompt", "give me
  a handoff", "draft a PR body", "prompt for z.ai", "/prompt-author".
metadata:
  short-description: "Author precise, verifier-gated task/PR/handoff prompts (this repo's discipline)"
---

# Prompt author — write a prompt the verifier would pass

**A vague prompt is an overclaim about the future.** "Make it better" asserts a result with no way
to check it — exactly the failure Sophia's gate exists to stop, one level up. This skill makes a
prompt *checkable* before you send it. The machine bar is `agent/prompt_quality_verifier.py`
(`score_prompt` / `prompt_quality_ok`); this is its human-readable half.

## When to invoke (be eager)

- Writing a PR body → also fill `.github/pull_request_template.md`.
- Handing work to another agent (z.ai / GLM / Copilot) or a future GPU session.
- Briefing a sub-agent, a `/loop`, or a workflow stage.
- The user says "write a prompt", "give me a handoff", "draft a PR", "prompt for X".

## The five dimensions (all four required ones must hold)

| Dimension | Ask | Markers that satisfy it |
|---|---|---|
| **success_criterion** (required) | How is "done" *verified*? | a test, `make claim-check` GO, a metric + CI, an expected output, κ≥0.40, "excludes zero", a receipt |
| **bounded_scope** (required) | What is in / out? | "only", "do NOT touch", a single deliverable, the exact branch, named files |
| **abstention_path** (required) | What on blocked/uncertain? | "if blocked, report NO-GO", "abstain", "ask first", "stays candidate", fail-closed |
| **no_overclaim** (required) | No unqualified superlative / AGI / safety claim | reuses `tools/lint_claims.py` FORBIDDEN; mark illustrative copy `claim-ok` |
| **grounding** (recommended) | Concrete artifacts, not "the thing" | file paths, `#PR`, branch names, `tools/…`, `agent/…` |

## Procedure

1. Draft the prompt.
2. Score it: `python -m agent.prompt_quality_verifier` style, or
   `from agent.prompt_quality_verifier import score_prompt`. Read `reasons`.
3. Fix every missing **required** dimension; add grounding unless the prompt is trivially short.
4. For any number you ask the recipient to PRODUCE, demand its CI + seeds + ≥2 judge families OR a
   CI excluding zero, and tell them it stays **candidate** until a gate clears (no overnight
   point-estimate becomes a headline).
5. For cross-agent work: name the recipient's **own** branch (never `main`, never another agent's
   feature branch), and say "open a PR; don't rebase mine."

## This repo's load-bearing guardrails to bake into any execution prompt

- `make claim-check` must be GO; `canClaimAGI` stays false; no artifact drift.
- RunPod GPU jobs go through the GitHub Action only (never local SSH).
- The optimiser/agent may edit policy/data/hyperparameters — **never** the verifier, gate, eval,
  reward, or constitution (the `sophia_autoresearch` firewall).
- religion / history are PROTECTED.

## Anti-patterns (the verifier will flag these)

- "Improve the model as much as possible." → no success_criterion, no scope, no abstention.
- "Build the world's first AGI / make it safe." → overclaim (and contradicts the failure ledger).
- "Fix it." / "the thing" → no grounding; the recipient guesses and wastes a cycle.

## Boundary

This scores prose for checkability; it is **not** a capability or safety claim and makes no AGI
claim. A prompt that passes is well-formed, not guaranteed-correct. For the thesis and the Layer-C
forge integration, see `docs/09-Agent/Prompt-Author-Skill-and-Verifier.md`.
