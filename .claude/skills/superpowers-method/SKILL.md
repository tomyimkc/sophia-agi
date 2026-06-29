---
name: superpowers-method
description: >
  Use when implementing a non-trivial new feature/experiment in this repo — especially anything
  research- or measurement-shaped (a new benchmark, verifier, training/eval lane, steering or
  council experiment, MCP/skill surface). This repo develops with a spec-driven discipline
  (spec -> plan -> task-by-task implementation) recorded under docs/superpowers/. Trigger when
  the user says "build/implement X", "add a benchmark/experiment/lane", "write a spec/plan",
  references docs/superpowers, Spec A/B/C/D, or a "superpowers" sub-skill — so new work follows
  the established two-tier, pre-registered, fail-closed pattern instead of ad-hoc code.
metadata:
  short-description: "Spec-driven (spec->plan->task) development discipline for new features/experiments"
---

# Superpowers method (spec-driven development for Sophia)

Non-trivial work here is built **spec → plan → task-by-task implementation**, with the spec and
plan committed under `docs/superpowers/` (git-crypt encrypted IP — auto-unlocked by the
`SessionStart` hook). This skill is the standing discipline; the specs/plans are the source.

## Read first

- `docs/superpowers/specs/` — the approved **design specs** (the "what + why + contract").
- `docs/superpowers/plans/` — the matching **implementation plans** with checkbox (`- [ ]`)
  tasks, file lists, and interfaces (the "how, task by task").
- The plan headers name the required sub-skill: **`superpowers:subagent-driven-development`**
  (preferred) or **`superpowers:executing-plans`** — implement one task at a time, commit after each.

## The pattern every plan enforces (copy it for new work)

1. **Two-tier architecture.** A **deterministic, pure-stdlib CI core** (no torch/numpy/network —
   CI has no pip step) plus an **opt-in real path** (torch/transformers/Ollama, lazy-imported,
   never a CI assertion). Mirror `tools/run_rlvr.py`: the GPU/real path runs locally, is
   skip-guarded in CI.
2. **Plain-script tests (NO pytest).** Each `tests/<file>.py` starts with the
   `ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))` guard, defines
   `def test_*() -> None` with bare `assert`, and a `main() -> int` that runs a `tests=[...]`
   list printing `ok {name}` then `PASS N <suite> tests`, ending
   `if __name__ == "__main__": raise SystemExit(main())`. Wire each new test file into
   `.github/workflows/ci.yml` with its own `python tests/<file>.py` line.
3. **Pre-register before you run.** Fix every threshold (effect size, κ floor, N/MDE, off-target
   bound) in the spec **before** any run. **A NULL/negative result is a legitimate, pre-registered
   outcome** — never relax a threshold to manufacture a positive (the no-overclaim contract).
4. **Reuse, don't fork.** Plans list the exact existing functions to call (e.g.
   `provenance_bench/consensus.py:cohen_kappa`, `agent/steering/stats.py`, `council_deliberate`).
   Add-only: don't modify shipped A/B/C/D logic; keep prior suites green.
5. **Fail-closed + least privilege.** Missing key/model → ABSTAIN, never a silent fallback.
   Secrets only in gitignored `.env`; never in code/spec/report/manifest/CI.
6. **Commit after every task**, on the feature branch the user named.

## Writing a new feature

1. Draft a spec in `docs/superpowers/specs/YYYY-MM-DD-<slug>-design.md` (context, goal, scope,
   the contract, pre-registered thresholds). Get it to "approved (design)".
2. Draft the plan in `docs/superpowers/plans/YYYY-MM-DD-<slug>.md` — checkbox tasks, each with
   **Files** (create/modify), **Interfaces** (produces/consumes), and a failing test first.
3. Implement task-by-task via the `superpowers:` sub-skill; run the plain-script tests + wire CI.
4. Before pushing, run the `ci-artifact-drift` skill (`make claim-check` + the drift gates) and,
   before any git write, the `git-discipline` skill.

## Existing program (already built — reuse, don't rebuild)

Specs A–D are implemented: Spec A personality-measurement gate (`agent/personality_map.py`),
Spec B activation-steering + behavioral PIF (`agent/steering/*`), Spec C personality council +
anti-gaming (`agent/council_personas.py`), Spec D capability-retention + MCP packaging
(`agent/steering/capability.py`, `sophia_mcp/tools_impl.py`). Check `docs/superpowers/` for the
design contract before extending any of them.
