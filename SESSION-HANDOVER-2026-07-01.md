# Sophia-AGI — Session Handover 2026-07-01 (real-time grounding loop + Phase-0 benchmark)

> Newest dated handover wins. Predecessors: `SESSION-HANDOVER-2026-06-29.md`, `-2026-06-28.md`.

## 1. Git / repo state at handover
- `origin/main` @ `488a59db` (source of truth). This handover lands via its own PR branch.
- Working tree carries the **intentional Obsidian local-only md graph diff** (~381 files, `<!-- OBSIDIAN-LINKS -->` blocks) + restored untracked files. This is expected — **never `git add -A`**, stage explicit files only.
- Merged this session: **#320**, **#321**. Handed off (another session): **#322**.

## 2. What this session did (verdicts within the no-overclaim ceiling)
- **#320 (`5ed0cadb`)** — Verifier-gated real-time grounding loop, **candidate-only, `canClaimAGI:false`**. Reuses the existing verifier stack to link a text world model to live web data and fact-check it instantly. Fast reversible loop (fact-check verdict → conformal nonconformity → content/temporal-decontam + valid-time → external belief store, never weights; `mark_stale` = re-verify primitive). Slow loop (habit-shaped GRPO rows + reversible LoRA-delta ledger, **dry-run**; GPU is a RunPod/Actions seam only). Modules: `agent/realtime_grounding.py`, `agent/streaming_decontam.py`, `agent/realtime_consolidation.py`; CLIs `tools/run_realtime_{grounding,consolidation}.py`. Offline + deterministic + tested.
- **#321 (`488a59db`)** — Phase-0 benchmark for **claim C1 (verifier-as-truth-filter)**. `agent/realtime_benchmark.py` + `tools/run_realtime_benchmark.py` + pre-registered `data/realtime/benchmark/measurement_spec.json`. **Verdict: candidate-underpowered.** On `eval/fact_check/heldout_v1` (N=53): verifier arms precision/recall 1.0 vs `accept_all` 0.585 fabrication / `raw_rag` 0.45; full-vs-raw_rag Δ+0.377 (McNemar p=0) but **underpowered** for the pre-registered 0.10 MDE (MDE@53≈0.27). Load-bearing metric = precision/passAt1, **not** meanReward.
- **#322 (`feat/epistemic-substrate`, another session)** — triaged its red `CodeQL` check: 3 high `py/clear-text-logging` alerts (#437/#438/#439 on `verify_verifiers.py`/`vov_selftest.py`) are **likely false positives** (the logged `receipt` holds only floors/precision/verdicts, no secret) and **non-blocking** (CodeQL isn't a required check). Posted a triage note; did **not** dismiss alerts or edit the branch (owner's decision; branch is in a live worktree).

## 3. Proven vs still open
- **Proven (offline, candidate):** the loop's wiring, the C1 verifier floor, and the no-overclaim gating behaviour (the harness refuses to certify at N=53 and flags the coupled-verifier risk). Ledger: `realtime-grounding-loop-candidate-only-2026-07-01`.
- **Open:** C1 powered + independent (needs N≥393 AND a **sealed audit set** the verifier is never tuned on AND ≥2 judge families — the verifier acing its own pack is circular); **C2** (live passAt1 uplift) and **C3** (drift / forgetting / reward-hack==0) both need a model → Phase-1/3.

## 4. ▶ Next step (single most valuable)
Phase-1: run the fast loop `--online` against a **frozen time-stamped snapshot**, measure ingestion precision/recall + live passAt1 vs baselines with pre-registered N + CIs.
- Build a temporally-frozen dated-QA pack (labels true/false/unknowable + `sourceTimestamp`), N≥`eval_stats.required_n_for_mde(0.10)` (~393), entity-disjoint + sealed.
- `python3 tools/run_realtime_benchmark.py --online` (keyless LiveFactBackend) → C1 live.
- **Pass/fail:** GO iff full-arm precision CI-clean above floor AND full > `raw_rag` by ≥ practical threshold (McNemar-significant), on the sealed set, ≥2 judge families κ≥0.40. Else honest NO-GO / candidate.
- C2/C3 require a model run — GPU via **GitHub Actions → RunPod only**, never local; gate any LoRA delta through `claim_gate` before merge; extend decontam to time.

## 5. Read-first
- `data/realtime/benchmark/measurement_spec.json` — pre-registration + guardrails (control-sanity, baseline-contrast, coupled-verifier audit, temporal decontam).
- `agi-proof/failure-ledger.md` → `realtime-grounding-loop-candidate-only-2026-07-01`.
- `agent/realtime_grounding.py`, `agent/realtime_benchmark.py`.
- `rlvr-harness-traps` skill — metric = passAt1/VSC not meanReward; pin a multi-seed sweep to one commit.

## 6. Don't-break
- Required CI checks: **`fast`** + **`ci-complete`**. Measurement gates: `lint_claims`, `lint_training_rows`, `assert_decontam`, `eval_stats`, `claim_gate --prefix M3-pilot`/`M3-transfer` (must stay GO), `validate_failure_ledger --check` + `tests/test_failure_ledger.py`.
- `canClaimAGI` stays **false**. Never quote meanReward as the capability number.
- Branch new PRs off `origin/main` (a stale local HEAD → add/add conflicts vs main's squash). Stash the Obsidian md diff (`git stash push --include-untracked`) before any rebase/switch. Copilot auto-reviews PRs; its unresolved threads block merge (`required_review_thread_resolution`).
