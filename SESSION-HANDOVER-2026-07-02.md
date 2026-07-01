# Session Handover — 2026-07-02

> Mac-operator session. Continuity pointer, not a duplicate of state. `origin/main` is the
> source of truth. See `agi-proof/failure-ledger.md` for the open-work list, `CHANGELOG.md`
> for what landed. canClaimAGI stays false; every number below is honest-or-NO-GO.

## 1. Git / repo state
- `origin/main` HEAD: the `#327` (security) + `#328` (CodeQL notes) merges on top of `#319`/`#306`/`#304`.
- Working checkouts of note: the **Spark cluster** (`~/sophia-bridge`, `spark-bridge` poller) and the
  **w1–w5** branch (`claude/sophia-w1-w5-live-g04zs1`, realtime-facts-pack growth) are **ACTIVE** —
  do not disturb the single Spark GPU or that branch/pack.
- Local `feat/codebase-memory-mcp-phase0` still carries the Phase-0 `tools/cbm/` controls (unmerged;
  see §3 open item).

## 2. What this session did (all merged to main)
- **OLMoE NVFP4 v6 QAT — fused-expert co-adaptation VALIDATED, honest NO-GO.** Fixed the blind spot
  where `attach_qat` reached **0/32** fused MoE experts; the trained v6 cert is **top1 0.9609**
  (< 0.97 bar → NO-GO) with **mean_kl 0.0341** (≤ 0.05 ✓). That is the largest single jump in the
  QAT program (v5 0.8828 → v6 0.9609) and the first *valid* measurement where experts trained against
  the serving grid. **Shippable path:** v6 + conformal abstention ≈ **93% coverage @ answered_top1 0.983**.
  Landed on `#319` via three fixes: fused-expert reach (`e8d16fab`), `_torch_nvfp4` bucketize perf
  69h→2.7h (`505daa89`), skip gradient-checkpointing under QAT (`6814f449`). Ledger:
  `nvfp4-v6-coadapt-cert-2026-07-01`.
- **Merged 3 stuck PRs:** `#304` (dive-into-llms gates + T3 calibration), `#306` (holdout-enforced
  invention RLVR task), `#319` (Spark cloud-bridge bench-A + council/security/operator + the QAT work).
  Resolved conflicts (RLVR-task unions, ledger), fixed 23 review threads on `#319`, and repaired
  latent pre-existing test failures it had masked (skill-index drift, dangling `prompt-author`
  manifest surface, a physics council-seed that tripped the algebraic-identity gate).
- **Cleared the CodeQL security tab → 0:** `#327` fixed both HIGH `clear-text-logging` alerts
  (`security_audit` no longer reads the matched token at all; `runpod_focus_frontier` logged the
  env-var *name*, renamed the dest); `#328` cleared the 17 notes (unused imports/locals,
  implicit-concat → explicit `+`, documented `except`, const-comparison, un-quoted a real `Callable`).
- **Hooks:** fixed the `.claude/hooks/session_start.sh` git-crypt lock-primitive — `grep -q GITCRYPT`
  misread a genuinely LOCKED clone (ciphertext starts with a leading NUL) as "already-unlocked" and
  never auto-unlocked; replaced with a deterministic 10-byte hex compare (`00474954435259505400`).

## 3. Proven vs still open
- **Proven (honest):** v6 co-adaptation lifts NVFP4 next-token agreement and passes the KL bar; the
  abstention serve path (~93%@0.983) beats v5's frontier. **Not proven:** the strict `top1 ≥ 0.97`
  bar (short by 0.009) — a v7 (β-sweep / more epochs / `--keep-top-experts`) or ship-with-abstention.
- **Open (non-GPU, non-w1–w5, safe next work):** `codebase-memory-mcp` Phase 1 — the Phase-0
  controls exist on `feat/codebase-memory-mcp-phase0` but need the errant mac-handoff commits stripped
  (`git rebase --onto origin/main <base>`) before a clean Phase-0 PR; then pin+verify the binary and
  wire `index_guard` as the `.mcp.json` entrypoint.
- **Open (GPU/judge-gated — belongs to the cluster / w1–w5, do NOT start here):** v7 QAT, the virtue
  real-evals, world-model real-model ablation, calibration-scaling, chem-bio, the m3-3family judge farm.

## 4. ▶ Next step
Land the `codebase-memory-mcp` Phase-0 controls as a clean PR (strip the unrelated commits first),
then Phase 1. Or write a v7 QAT recipe doc (no GPU) for when the cluster frees. Everything strictly
off the Spark GPU and the `w1–w5` branch.

## 5. Read-first
`agi-proof/failure-ledger.md` (rows `nvfp4-v6-coadapt-cert-2026-07-01`, `fused-expert-qat-coadapt-reach-2026-07-01`),
`docs/06-Roadmap/QAT-v6-Recipe-Proposal.md`, `SESSION-COORDINATION.md` (live GPU claims),
`docs/superpowers/plans/2026-07-01-codebase-memory-mcp-phase0-security-controls.md`.

## 6. Don't-break
Required CI checks `fast` + `ci-complete`; `make claim-check` (no-overclaim + gate GO/NO-GO);
the generated-artifact drift gates (skill index, wiki, results page, failure-ledger validator).
Never flip a ledger row to a positive claim without a pre-registered gate + artifact + sha256.
