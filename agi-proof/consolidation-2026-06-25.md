# Sophia AGI — Branch Consolidation Report

**Date:** 2026-06-25
**Agent:** consolidation
**canClaimAGI:** **False** (candidate-only on everything)
**main HEAD after consolidation:** `d6032f9` (Merge PR #104)

This report inventories active feature branches, records what is verified-done vs
blocked, and documents the PRs landed to consolidate merge-ready infrastructure
into `main`. No training-success, uplift, or AGI claims are made or merged.

---

## 1. Active-branch inventory (in-scope workstreams)

All four target branches diverged from the *old* main (`ec34f09`, PR #99). Each was
reconciled by merging `origin/main` in, re-verified (lint_claims, decontam, tests),
pushed, CI-gated green, and merged in order.

| Branch | PR | CI | Merge SHA | Result |
|---|---|---|---|---|
| claude/error-memory-rag | #100 | green | (pre-session) | **MERGED** |
| claude/sophia-team-orchestrator | #102 | green | `299ad14` | **MERGED** |
| claude/sophia-7b-train-verify | #103 | green | `65da808` | **MERGED** |
| claude/sophia-math-code-curriculum | #104 | green | `d6032f9` | **MERGED** |
| claude/team-agents-mode | — | — | — | **Superseded by #102** (not merged) |
| chore/math-code-sft-workflow-on-main | #101 | — | (pre-session) | Redundant (workflow already on main) |

Other `claude/*` and `feat/*` branches (40+ total) are 56–220 commits behind main and
out of scope for this consolidation; left untouched.

## 2. Merged to main this session

### PR #102 — Team-orchestrator (Workstream A2) → `299ad14`
- **Landed:** runtime team-deliberation orchestrator (`agent/team_agents.py`), MCP +
  CLI wiring, sealed team/HK/long-task benchmarks, eval harnesses (prompt-parity +
  panel-independence), HK-advisor trace distillation / DPO-pair mining. Runs against the
  existing `sophia-v3` candidate adapter + a mock path — **no new LoRA weights**.
- **Did NOT land:** any validated team/council LoRA, AGI evidence, or uplift. Only
  numbers are mock-path + deterministic external scorer.
- **CI fix:** lazy-import `team_agents` in `tools/council_deliberate.py` so the
  dependency-light `validate-build` job stays numpy-free (`7c0570a`).

### PR #103 — Sophia-7B train/verify (Workstream B1) → `65da808`
- **Landed:** sealed held-out pack + seal tooling, decontaminated flywheel (guard CLEAN),
  RunPod 3-seed SFT/DPO pipeline (`train_dpo.py`), pre-registration, honest blocker report.
- **Did NOT land:** any claim SFT/DPO completed (`sftSeedsCompleted: 0`,
  `dpoSeedsCompleted: 0`), no promote/accept, no adapters, no uplift numbers.
- **Conflict resolution:** unified `tools/build_local_sophia_dataset.py` SFT-sources
  (kept 7B's function refactor + #102's HK-advisor source) and the failure-ledger union.

### PR #104 — Math/code curriculum (Workstream B2) → `d6032f9`
- **Landed:** sympy/exec hard-oracle verifiers + tests, 144-row verified curriculum,
  sealed held-out + guard, QLoRA wiring, RunPod launcher, pre-registration, blocker report.
- **Did NOT land:** uplift/RLVR results or promote verdicts (`seedsCompleted: 0/3`).
- **CI fix:** install `sympy` in the full-pytest `test` job (`1b9537a`).
- **Conflict resolution (significant):** unified `tools/runpod_train.py` so it supports
  **all three** training paths — 7B SFT/DPO (`--dpo-pairs`, `--sft-adapter-archive`,
  `--ssh-login-timeout-s`) **and** math-code sealed-curriculum SFT
  (`--adapter-dir`, `--train-only`, minimal recipe). This also fixed a latent break:
  the math-code GHA workflow (already on main via #101) called `--adapter-dir`/`--train-only`
  flags that main's `runpod_train.py` did not yet define. Also unified `dataset_guard.py`
  (seal-manifest + team manifests), `failure-ledger.md`, and `preregistered-thresholds.md`.

Also already on main pre-session: PR #100 (error-memory-rag), PR #101 (math-code RunPod
workflow), `dcb4eae` (7B RunPod workflow).

## 3. Verification performed (per branch, before each merge)

- `tools/lint_claims.py` → OK on every commit
- `tools/build_local_sophia_dataset.py --check` → contamination guard CLEAN
- Full local `pytest` suite green on each resolved tree (team-orch 1208, 7B 1184,
  math-code 1238 passed; z3-solver + numpy + sympy installed to match CI)
- `tools/runpod_train.py --dry-run` rendered correctly for SFT / SFT-only / DPO modes
- GitHub Actions `ci-complete` required check green on #102, #103, #104
- Push-verify (`git ls-remote` == local HEAD) after every push

## 4. Blocked items (Workstream C — document only, NOT forced)

| Blocker | Affected | Unblock (user action) |
|---|---|---|
| RunPod SSH egress timeout from Mac/Cursor shell | 7B, math-code GPU stages | Dispatch GHA workflows (`runpod-sophia-7b-sft.yml`, `sophia-math-code-sft-runpod.yml`) with `RUNPOD_API_KEY` secret — do NOT run launch scripts from the Cursor agent shell |
| Apple Silicon + mlx_lm | team-orchestrator real eval | Run eval on local MLX hardware |
| VECTARA_* unset | 7B Stage 5 evidence | Provide Vectara credentials |
| Hidden-pack serving endpoint | 7B, math-code evidence | Provide model endpoint + creds |

## 5. Consolidated failure-ledger status (per workstream)

| Workstream | Ledger ID | Status |
|---|---|---|
| error-memory-rag | error-memory-rag-pr-100-2026-06-25 | Partial (live-model eval pending) |
| team-orchestrator | team-agents-*-2026-06-25 | Open (mock + deterministic scorer only; real MLX eval pending) |
| sophia-7b-train-verify | sophia-7b-train-verify-data-flywheel / stage2-runpod-blocker | Open / BLOCKED (infra; 0 seeds) |
| math-code-curriculum | math-code-curriculum-preregistered-2026-06-25 | Open / BLOCKED (infra; 0/3 seeds) |

All four entries carry `canClaimAGI: false`. No ledger entry was closed as proven.

## 6. Honest headline

The repo's **inference-time guardrail + evaluation/training substrate** is now
consolidated on `main`: a precision-gated error-RAG, a runtime team orchestrator with
sealed benchmarks, hard-oracle math/code verifiers, sealed/decontaminated 7B and
math-code training packs, and pre-registered RunPod SFT/DPO pipelines (now unified in
one `runpod_train.py`). **No GPU training run has completed for any candidate** — every
training path is blocked on RunPod SSH egress and credentials, with 0 completed seeds
and honest BLOCKED reports. The repo can verify, decontaminate, seal, and stage
training; it cannot yet show post-training uplift. `canClaimAGI` stays **False**.

## 7. Recommended next actions (ordered)

1. Run **7B SFT** via GitHub Actions `runpod-sophia-7b-sft.yml` (set `RUNPOD_API_KEY`
   repo secret) — the documented unblock for the SSH egress failure. Then DPO.
2. Run **math-code SFT** via `sophia-math-code-sft-runpod.yml` (3 seeds) the same way.
3. Run the **team-orchestrator real MLX eval** on Apple Silicon to move its ledger row off Open.
4. Provide **VECTARA_*** + a served hidden-pack endpoint to unlock 7B third-party evidence.
5. Delete/retire the superseded `claude/team-agents-mode` branch (its work is in #102).

## 8. canClaimAGI: False
