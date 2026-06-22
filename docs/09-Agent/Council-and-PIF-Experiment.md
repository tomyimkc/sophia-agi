# Council Diversity and PIF Experiment (Spec C)

Companion to `docs/09-Agent/Steering-Experiment.md`. That document covers the
Spec B activation-steering engine; this one covers the three Spec C claims.

---

## C1 — Personality-diverse council (ΔQ claim)

**The falsifiable claim:** a council whose seats carry distinct OCEAN profiles
(4 poles pre-registered: O+C−, O−C+, E+A−, E−A+) produces higher pass-rate on
the domain benchmark than a homogeneous all-median-OCEAN council.

**Why a NULL result is legitimate:** mock-text scores are zero for all arms with
the mock client, so ΔQ = 0 with mock is expected and correct. With a real model,
personality prefix framing may shift word choice without changing factual accuracy
on verifiable questions. The experiment is structured to detect a real effect
*when one exists* — not to manufacture one.

**The contract:** "trait diversity does not improve council quality" is a
legitimate, honest result and is reported as such in
`agi-proof/benchmark-results/council-diversity.public-report.json`. A headline of
PASS requires both `dq > 0` and `dq_ci[0] > 0` (lower bootstrap-CI bound
positive). Everything else is NULL.

**CI status:** `tests/test_council_diversity.py` is green in CI (PASS 3). The
driver (`tools/run_council_diversity.py --model mock --domain philosophy`) writes
the public report; ΔQ = 0.0 with mock is correct plumbing, not a failure.

---

## C2 — Sealed held-out split and anti-gaming contract

**Design:** the held-out answer set for each steering axis is sealed with a
salted SHA-256 digest before any run. The seal is verified at load time; any
mutation of the held-out labels prior to evaluation is detectable. The full-text
of held-out answers is withheld from the main process (grep-gate enforced in
`tests/test_pif_harness.py:test_grep_gate_no_plaintext_heldout_answer`).

**Ship-or-ABSTAIN invariants** (implemented in `agent/steering/anti_gaming.py`):

| Invariant | Threshold | Mirrors |
|---|---|---|
| `not_gamed`: `(fit_shift − held_shift) ≤ 0.20` | GAP = 0.20 | `verified_reward.reward_is_hackable` gap |
| `off_target_bounded`: `heldoutOffTargetRate ≤ 0.10` | OFF_TARGET_MAX = 0.10 | flywheel `heldoutFalseAcceptRate` cap |
| `target_moved`: target axis shifted on held-out split | boolean | selfextend mirror |

A steering vector ships only when all three invariants hold on the held-out
split. If any fail, the system ABSTAINs and records the reason in the ledger.
Fail-closed: an unvalidated vector is treated as worse than none.

**Selfextend mirror:** this contract mirrors the selfextend held-out discipline
(`agent/selfextend/verified_reward.py`). The same logic that prevents the
flywheel from overclaiming on its training domain applies here to prevent a
steering vector from being "fitted" to its calibration data.

**CI status:** `tests/test_pif_harness.py` is green (PASS 11), covering sealing
reproducibility, grep-gate, and the ship/ABSTAIN gate paths.

---

## C3 — PIF harness (pre-registered live claim)

**The offline claim (CI-gated):** the PIF/SSA statistics pipeline is
arithmetically correct — residualized Cohen's d, bootstrap CI, Holm–Bonferroni
and Benjamini–Hochberg correction, and the SSA verdict function all match
hand-computed reference values. Verified by `tests/test_pif_harness.py` (pure
stdlib, no model).

**The pre-registered live claim (OPEN):** for N≥8 OCEAN axes and K≥20 seeds
per cell, at least one steering axis produces a BH-significant enacted cell
(`verdict.status == "enacted"` AND survives BH at q=0.05). The SSA headline is
`enacted/total`. `SSA = 0/N` — zero enacted cells — is a legitimate honest
result; it would mean steering shifts self-report but not behavior, or that the
effect is below the pre-registered thresholds. Thresholds are fixed in
`agent/steering/stats.py:SSA_THRESHOLDS` before any run.

**Two-tier discipline:** the CI core is pure stdlib (no torch, no Ollama). The
opt-in real tier (`--model phi3.5`) downloads ~7.6 GB and requires local Ollama
judges. CI runs only the stdlib tier. This mirrors the steering-experiment
two-tier design and ensures the wiring is verifiable without GPU access.

**Dry-run:** `python tools/run_pif.py --dry-run` → `PIF HARNESS VERIFIED ✓`
(offline, no model required).

---

## Honest-result discipline

The following results are legitimate and should be reported as-is:

- `SSA = 0/N` (zero enacted steering cells across all axes)
- `ΔQ ≤ 0` (diverse council does not outperform homogeneous council)
- `headline = "NULL: trait diversity does not improve council quality on this slice"`

Negative or null results are evidence. They constrain the claim space and prevent
overclaiming. The ledger (`agi-proof/failure-ledger.md`) tracks the OPEN headline
run as a named entry until a gated real run clears it.
