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

---

# Appendix — web session (later on 2026-07-02): recommendation-adoption sprint

> Branch `claude/sophia-workflow-agi-recommendations-2z8pbt` (all pushed). Separate session
> from the Mac-operator one above; be aware BOTH exist. My seam work (W1/W2/W3/W5) may brush
> against the ACTIVE `claude/sophia-w1-w5-live-g04zs1` branch at merge time — reconcile there,
> not by rebasing their branch.

## A1. What landed (one commit each, tests green, lint_claims OK throughout)
- `c225e23` chart-vs-code verification + ranked recommendations
  (`docs/09-Agent/flowcharts/Workflow-AGI-Recommendations-2026-07-02.md`).
- `e860d04` **R5/W1**: `provenance_bench/prm_step_reward.py` + `run_rlvr --task step
  --step-reward prm --prm-derivations --prm-cap` — PRM wired as an RLVR arm; symbolic oracle
  authoritative, PRM fills abstains only, capped; containment invariants in the dry-run report.
- `27e2fd1a` **R6/W3**: `build_local_sophia_dataset --provenance-weighting[/-floor/-repeat]`
  — declared PACK_PROVENANCE map → curriculum order + 1:1 weight sidecar + optional
  replication; default build byte-identical.
- `dbef2a52` **R3/W5 seam CLOSED**: real `build_hidden_state_featurizer` (MLX final-layer
  residual stream, lazy, fail-closed on x86) + `train_vector_probe`/`evaluate_vector_probe`
  + new `mac-mlx-bench.yml` (self-hosted macOS runner lane).
- `57c0e3c7` **R2/W2 bridge**: `tools/build_calibration_dpo_pack.py` — balanced honesty DPO
  pairs (one-sided packs refused), registered as `dpo_calibration.jsonl`; Platt baseline +
  post-train re-audit pre-registered in its own report.
- `b59fc296` **R4 vehicle**: `mac-mlx-bench.yml suite=claim-router-ablation`
  (sophia-full vs sophia-claim-router, 18-case abstain pack, mlx adapter backend).
- New plaintext skill `remote-session-fallbacks` (Bash-classifier outage playbook,
  post-API-push resync, git-crypt-in-container, cluster-dispatch-from-web).

## A2. In flight — check FIRST
- **rlvr-runpod run #64** (offline smoke, provenance/gate/seed0): `waiting` on the owner's
  `runpod-paid` approval → actions/runs/28558110902. Owner pre-authorized smoke → then the
  3-seed live sweep (reward=gate AND multiaxis).
- **spark-gpu run #2** (offline reward-wiring smoke, NO model load / NO GPU allocation —
  safe next to the active bridge work): `queued` until the Spark runner is free/online →
  actions/runs/28558129937.
- **mac-mlx-bench**: not yet dispatched; first `suite=featurizer` (validates Mac runner +
  flips W5/energy-head readiness True there), then `suite=claim-router-ablation` (R4 evidence).

## A3. Next steps (plan of record)
1. Approve #64 → smoke green → dispatch live sweep seeds {0,1,2} × reward {gate, multiaxis}.
2. Post-train re-audit MUST include answerable-coverage (abstention-collapse check);
   pass@1/VSC load-bearing, never meanReward; ledger row `rlvr-live-run-not-yet-gated-2026-06-21`.
3. Mac bench R4 deltas → if router wins, `evaluate_update()` with answerable-coverage
   protected BEFORE flipping `use_claim_router` default.
4. R5 live arm needs a derivations JSONL; R2 training needs records that carry prompt+answer.

## A4. Traps rediscovered
`rlvr.public-report.json` is rewritten by every run incl. dry-runs — `git checkout --` before
commit. Red `cleanPositive` on step-task offline invariants = missing sympy, not a bug.
`sophia-security-audit/SKILL.md` shows M after unlock (clean-filter artifact) — never stage.

## A5. Second wave (same session, after the Agents-A1 study): A-series IMPLEMENTED + live sweep

Paper study: `agi-proof/agents-a1-horizon-scaling-2026-07-02/README.md` (3e8d721d). All seven
proposals then implemented, each tested + linted (57 passed / 2 MLX-skipped overall):
- `82c5831f` **A4** `provenance_bench/rl_data_curation.py` + run_rlvr `--advantage-shaping papo
  --lambda-neg` (mixed-outcome filter, dynamic-sampling predicate, PAPO shaping; offline invariants
  in the dry-run report).
- `72732300` **A2** `tools/distill_sva_mlx.py` — SVA math core (top-k truncated reverse-KL, rho
  monitor, hard routing, Eq.-6 aggregation) CI-tested; MLX step = Mac-bench seam.
- `2a1c8745` **A1** `tools/build_trajectory_pack.py` — run_case/long-horizon records ->
  loss-masked (s,a,o,v) trajectories; failures -> DPO negatives; five acceptance gates.
- `82c5591f` **A5** `tools/selfplay_task_forge.py` — masked-entity multi-hop + doNotAttributeTo /
  authorConfidence traps over data/attributions.json; decontaminated; seed-deterministic.
- `2b8ef965` **A6** long-horizon v2: `.notes.jsonl` durable memory, `kind:model` steps with
  gateCheck verification events, ENFORCED resourceManifest (violated -> scoreable:false).
- `6a12a05c` **A3** `tools/train_council_teacher.py` (two-stage specialist SFT; protected seats
  refused; candidate-only) + **A7** `tools/stage_decomposition_report.py` (fail-visible stage
  regressions; dual official/reproduced baseline provenance).

**R1 state: offline smoke run #64 SUCCEEDED on the pod path** (pod rented+deleted, artifacts
uploaded, 2026-07-02T01:55Z). The 3-seed live sweep is DISPATCHED and pending runpod-paid
approvals: provenance x {gate, multiaxis} x seeds {0,1,2} on this branch (includes the A4/R5
run_rlvr additions, inert on the gate/multiaxis path). After the sweep: ingest_rlvr_eval gates
per run + the calibration/abstention re-audit INCLUDING answerable-coverage before touching
ledger row `rlvr-live-run-not-yet-gated-2026-06-21`. pass@1/VSC load-bearing, never meanReward.
