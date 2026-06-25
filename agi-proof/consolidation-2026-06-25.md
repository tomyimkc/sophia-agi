# Sophia AGI — Branch Consolidation Report

**Date:** 2026-06-25
**Agent:** consolidation
**canClaimAGI:** **False** (candidate-only on everything)
**main HEAD at start/end:** `6d33968` (Merge PR #100, error-memory-rag)

This report inventories active feature branches, records what is verified-done vs
blocked, and tracks the PRs opened to consolidate merge-ready infrastructure into
`main`. No training-success, uplift, or AGI claims are made or merged.

---

## 1. Active-branch inventory (in-scope workstreams)

All four target branches diverged from the *old* main (`ec34f09`, PR #99) and were
9 commits behind current main. Each was reconciled by merging `origin/main` in
(clean auto-merge, no conflicts), re-verified, and pushed.

| Branch | Pre SHA | Post-merge SHA | Ahead | PR | State |
|---|---|---|---|---|---|
| claude/error-memory-rag | 8a91ab9 | — | 0 (merged) | #100 | **MERGED to main** (pre-existing) |
| claude/sophia-team-orchestrator | 5821ec4 | 7c0570a | 9 | #102 | PR open (CI) |
| claude/sophia-7b-train-verify | 2a0a7b5 | e69f8f5 | 8 | #103 | PR open (CI) |
| claude/sophia-math-code-curriculum | 080b821 | 1b9537a | 8 | #104 | PR open (CI) |
| claude/team-agents-mode | b187afa | — | 4 | — | **Superseded by #102** (do not merge) |
| chore/math-code-sft-workflow-on-main | 2c2b726 | — | 1 | #101 (merged) | Redundant (workflow already on main) |

Other `claude/*` and `feat/*` branches (40+ total) are 56–220 commits behind main and
out of scope for this consolidation; left untouched.

## 2. Already merged to main (prior to this session)

- **PR #100** — error-memory-rag: failure store, precision-gated error-RAG, sealed
  v2 eval (N=40), eval CLI. Ledger row `error-memory-rag-pr-100-2026-06-25` =
  **Partial** (deterministic oracle only; live-model eval still pending).
- **PR #101** — math-code RunPod SFT workflow on default branch.
- **`dcb4eae`** — runpod-sophia-7b-sft workflow_dispatch on main.

Net: the GPU-trigger workflows for both training branches are ALREADY on main; the
open PRs add the verifiers/data/launchers behind them.

## 3. PRs opened this session

### PR #102 — Team-orchestrator (Workstream A2)
- **IS:** runtime team-deliberation orchestrator (`agent/team_agents.py`), MCP +
  CLI wiring, sealed team/HK/long-task benchmarks, eval harnesses with prompt-parity
  + panel-independence reporting. Runs against existing `sophia-v3` candidate adapter
  (no new LoRA weights).
- **IS NOT:** validated team/council LoRA, AGI, proven council uplift. Only numbers
  are mock-path + deterministic external scorer.
- **Ledger:** real MLX eval OPEN (`team-agents-*-2026-06-25`).
- **CI fix applied:** lazy-import `team_agents` in `tools/council_deliberate.py` so
  the dependency-light `validate-build` job stays numpy-free (commit 7c0570a).

### PR #103 — Sophia-7B train/verify (Workstream B1)
- **IS:** sealed held-out pack + seal tooling, decontaminated flywheel (guard CLEAN),
  RunPod 3-seed SFT/DPO pipeline, pre-registration, honest blocker report.
- **IS NOT:** any claim SFT/DPO completed. `sftSeedsCompleted: 0`, `dpoSeedsCompleted: 0`.
  No promote/accept, no adapters, no uplift numbers.
- **Ledger:** OPEN / BLOCKED on infrastructure (RunPod SSH egress).

### PR #104 — Math/code curriculum (Workstream B2)
- **IS:** sympy/exec hard-oracle verifiers + tests, 144-row verified curriculum,
  sealed held-out + guard, QLoRA wiring, RunPod launcher, pre-registration, blocker report.
- **IS NOT:** uplift/RLVR results or promote verdicts. `seedsCompleted: 0/3`.
- **Ledger:** OPEN (`math-code-curriculum-preregistered-2026-06-25`).
- **CI fix applied:** install `sympy` in the full-pytest `test` job (commit 1b9537a)
  so the curriculum tests run instead of erroring at import-time collection.

## 4. Blocked items (Workstream C — document only)

| Blocker | Affected | Unblock (user action) |
|---|---|---|
| RunPod SSH egress timeout from Mac/Cursor shell | 7b, math-code | Dispatch GHA workflows (`runpod-sophia-7b-sft.yml`, `sophia-math-code-sft-runpod.yml`) with `RUNPOD_API_KEY` secret — do NOT run launch scripts from the Cursor agent shell |
| Apple Silicon + mlx_lm | team-orchestrator real eval | Run eval on local MLX hardware |
| VECTARA_* unset | 7b Stage 5 evidence | Provide Vectara credentials |
| Hidden-pack serving endpoint | 7b, math-code evidence | Provide model endpoint + creds |

## 5. Consolidated failure-ledger status (per workstream)

| Workstream | Ledger ID | Status |
|---|---|---|
| error-memory-rag | error-memory-rag-pr-100-2026-06-25 | Partial (live-model eval pending) |
| team-orchestrator | team-agents-*-2026-06-25 | Open (mock + deterministic scorer only; real MLX eval pending) |
| sophia-7b-train-verify | stage2-runpod-blocker | Open / BLOCKED (infra; 0 seeds) |
| math-code-curriculum | math-code-curriculum-preregistered-2026-06-25 | Open / BLOCKED (infra; 0/3 seeds) |

## 6. Honest headline

The repo's **inference-time guardrail + evaluation substrate** is the consolidated,
mergeable asset: a precision-gated error-RAG (already on main), a runtime team
orchestrator with sealed benchmarks, hard-oracle math/code verifiers, sealed/
decontaminated 7B and math-code training packs, and pre-registered RunPod pipelines.
**No GPU training run has completed for any candidate** — every training branch is
blocked on RunPod SSH egress and credentials, with 0 completed seeds and honest
BLOCKED reports. The repo can verify, decontaminate, seal, and stage training; it
cannot yet show post-training uplift. `canClaimAGI` stays **False**.

## 7. Recommended next actions (ordered)

1. Review/merge **PR #102** (team-orchestrator) once CI is green — pure runtime infra.
2. Review/merge **PR #103** and **PR #104** (training infra only; honest "no results" titles).
3. Run 7B SFT via **GitHub Actions** `runpod-sophia-7b-sft.yml` (RUNPOD_API_KEY secret) —
   the documented unblock for the SSH egress failure.
4. Run math-code SFT via `sophia-math-code-sft-runpod.yml` similarly.
5. Run the team-orchestrator real MLX eval on Apple Silicon to move its ledger row off Open.

## 8. canClaimAGI: False
