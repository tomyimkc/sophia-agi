---
name: sophia-karpathy-engineering
description: >
  Engineering discipline for changes in the sophia-agi repo (and any codebase):
  think before coding, simplicity first, surgical changes, goal-driven verified
  execution. Use when writing, reviewing, or refactoring code so changes stay
  minimal, traceable, and provably done. Adapts Andrej Karpathy's four
  anti-pitfall guidelines to Sophia's no-overclaim, fail-closed charter. Invoke
  on /karpathy, /engineering-discipline, or before any non-trivial code edit.
metadata:
  short-description: "Karpathy's 4 coding guidelines, fused with Sophia's no-overclaim discipline"
  source: "Adapted from github.com/multica-ai/andrej-karpathy-skills (MIT)"
---

# Sophia × Karpathy — engineering discipline

**Wisdom before intelligence applies to *code*, too.** Four guidelines that cut
the LLM coding pitfalls Karpathy named, tuned to this repo's charter: every change
should make reasoning *more* checkable, never less ([VISION.md](../../../VISION.md)).

Why this repo needs them: sophia-agi is large and verification-first. Speculative
abstractions, drive-by refactors, and unverified "done" claims are exactly what
the no-overclaim gate exists to prevent — so hold code to the same bar as claims.

## 1. Think before coding
*Don't assume. Don't hide confusion. Surface tradeoffs.*

- State assumptions openly before editing. If a request has multiple readings,
  present options (use `AskUserQuestion`) instead of choosing silently.
- Recommend the simpler approach when one exists.
- **Sophia overlay:** if you're unsure a change is correct, **abstain and ask** —
  the same fail-closed reflex the gate enforces. A wrong guess that compiles is
  the code equivalent of a confident hallucination.

## 2. Simplicity first
*Minimum code that solves the problem. Nothing speculative.*

- No unrequested features, no abstractions for a single use, no config nobody
  asked for, no error handling for impossible states.
- If it's 200 lines where 50 would do, rewrite it.
- **Sophia overlay:** prefer **dependency-free + deterministic** (the charter for
  `pretraining/`, `okf/`, `skills/`). A new third-party dep must earn its place;
  optional deps are lazily imported and degrade fail-closed (see `agent/model.py`,
  `pretraining/gpt/`).

## 3. Surgical changes
*Touch only what you must. Clean up only your own mess.*

- Preserve adjacent code, formatting, and existing style. Don't refactor working
  code you weren't asked to touch.
- Remove only imports/functions **your** change orphaned. Note unrelated dead
  code; don't delete it unprompted.
- **Sophia overlay:** never relax a gate, threshold, or pre-registration to make
  a diff pass. If a guard is in the way, that's a signal to **stop**, not edit
  (`tools/lint_claims.py`, `tools/validate_failure_ledger.py` exist for this).

## 4. Goal-driven, verified execution
*Define success criteria. Loop until verified.*

- Turn the task into measurable checks before starting; for multi-step work,
  outline the plan with verification checkpoints.
- **Run the check**, don't assume it passes. Report what you ran and its output.
- **Sophia overlay — the no-overclaim rule for code:** "done" means a test or
  command was executed and observed. If you couldn't run it (no torch, no GPU,
  no key), say so explicitly and mark it `candidate`/unverified — never imply a
  green that wasn't observed. Mirror `RESULTS.md`: only verified numbers headline.

## Quick checklist (before you say "done")

- [ ] Assumptions stated / ambiguity surfaced (not silently resolved)
- [ ] Smallest diff that solves it; no speculative code or deps
- [ ] Only intended files touched; style matched; no gate weakened
- [ ] Success criteria defined **and run**; output reported honestly
- [ ] Unverifiable steps labelled as such, not claimed as passing

## If inside sophia-agi

- Match the package's dependency posture (pure-Python where the charter says so).
- Run the nearest test (`python -m pytest tests/test_<area>.py -q`) and any
  linter the area ships (`tools/lint_claims.py`, `tools/validate_*`).
- Keep `canClaimAGI: false` on generated artifacts; don't introduce capability
  language a measurement gate hasn't cleared.

## Do not

- Add abstraction or configurability "for later"
- Refactor or reformat code outside the task's blast radius
- Weaken a verifier, gate, or pre-registered threshold to pass
- Claim a test passed without running it, or hide that you couldn't run it
