# AGI-Foundation Roadmap — what to do next

> **Status:** advisory strategy note (2026-06-29). `canClaimAGI: false` (unchanged, by design).
> This document does not promote any result; it prioritizes the *open* work in
> `agi-proof/failure-ledger.md` toward the repo's stated north star. Nothing here relaxes
> the measurement contract — the gate decides validity, never prose.

## 1. What this repo already is (don't re-prove it)

Sophia is a **measurement-gated, provenance-aware reasoning layer that abstains instead of
fabricating** — a fail-closed gate (`claim → verify → accept · abstain · block`) under the
strictest no-overclaim discipline in the codebase. The deeper thesis is the **Instrumented
Evaluation Contract (IEC)**: in small-corpus work the instrument produces wrong conclusions
before the model does, so evaluation is engineered as a first-class, deterministic, CI-enforced
artifact (`agi-proof/measurement-thesis.md`).

Engineering reality (verified against code, not docs): the verifier/gate stack, `SophiaContract`
golden-vector governance, the OKF belief/provenance graph, and the continual-learning loop are
real and tested (~453 test files). Sparse-serving GPU/RDMA tiers and Rust Raft are honestly
*labeled* scaffolding. **The moat is the discipline + the public failure ledger, not feature count.**

## 2. The single thing capping the whole repo: the independence gap

Every result that clears (or nearly clears) the VALIDATED bar is either **self-authored** or
**single-subject**:

| Result | Grade | The independence limit |
|---|---|---|
| Provenance-delta hallucination cut (Δ 12.5% [5.6, 19.4]) | VALIDATED | one *weak* base model; pattern decays on stronger bases; N≤48; self-authored pack |
| SimpleQA selective prediction (+15.8% @20% cov, N=1000) | VALIDATED | external gold ✓ but only 2 subject models; calibration result, not capability |
| Legal-holding faithfulness (100%, κ=1.0) | VALIDATED | N=8 constructed; validates the tier, not subtle reasoning |
| M3-SFT source-discipline (Qwen 0.72 / Llama-70B 0.81) | CANDIDATE | κ=0.24 prevalence-deflation (not fixable); self-authored corpus |

**`canClaimAGI` flips only when a third-party hidden eval is beaten** — and *no* current run is
both independently authored and run across multiple subject families. This is the ceiling. Close
it before adding capability surface.

## 3. Priority order (highest leverage first)

### P0 — Bank the two free local benchmarks already wired (this week, $0)
Hardware is live (Spark + Mac, 10GbE/Tailscale). Both are ~95 min, owned-hardware, no metered cloud.
- **Benchmark A — two-box ≥2-family judging** of M3-SFT. *Do not chase formal VALIDATED* — κ=0.24 is
  prevalence-deflation math. Report the win-rate panel + Gwet AC1 + CI as the honest ceiling and keep
  the CANDIDATE label. (`tools/judge_pilot_answers.py` → `tools/run_lora_uplift_validation.py`.)
- **Benchmark B — NVFP4 low-RAM certification.** v3 is best so far (mean_KL 0.045 ✓, top-1 0.906 ✗).
  Push top-1 ≥ 0.97 with more QAT epochs / a calibrated protected slice, or accept "next-token faithful
  in aggregate" as the measured ceiling and write the artifact. (`tools/certify_lowram.py --scheme nvfp4`.)

### P1 — Fix the instrument before trusting more numbers (this is the thesis)
The deterministic fabrication scorer carries ~20% label error (ledger #55–56): it over-flags correct
debunks and over-credits hedged names. A noisy instrument silently caps every downstream claim.
**Audit and re-anchor it against a human-gold slice** before pushing more answers through it. Per the
IEC, instrument debt is the highest-priority class of bug in this repo.

### P2 — Close the independence gap (the real "AGI-foundation" move)
Stand up a **third-party hidden eval**: an unspent, *externally authored* pack with an independent
reviewer signature, run across **≥3 distinct subject model families**. Beating that — under the
existing gate — is the one thing the contract says flips `canClaimAGI`. Concrete spec:
`agi-proof/independence-eval-plan.md`.

### P3 — Measure ONE long-horizon capability
Long-horizon autonomy and learning-under-shift are **pre-registered but never run**
(`agi-proof/preregistered-thresholds.md`) — and they are the Level-3+ gate. A foundation for AGI
needs at least one *measured* long-horizon result, even a narrow, deterministic, offline-testable one.
This is where net-new engineering belongs, not in widening the verifier surface.

### P4 — Resist scope creep
The swarm already has a **published null result** (no benefit over solo on fact-check). Do not add more
swarm / verifier surface to chase a story. Deepen evidence on what exists; let complementary process
branches (skills/hooks) land separately.

## 4. The operating rule (unchanged)

Never report a number without its CI, seeds, judge families, and candidate/validated label. If an
open ledger item closes, **upgrade the public wording — never silently relax the gate.** When in
doubt, label CANDIDATE and add a ledger entry. The credibility *is* the discipline.

## 5. One-line strategy

**Treat the failure ledger as the roadmap. Independence (P2) + one measured long-horizon result (P3)
are the two items that actually move this from "trustworthy-reasoning layer" toward "foundation for AGI";
P0/P1 are the cheap, owned-hardware steps that bank what you've already built.**
