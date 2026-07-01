# Epistemic-substrate diagnostics — 2026-07-01

Real receipts from running the **buildable** gates on the **actual** corpus (worktree
`feat/epistemic-substrate`, `python3.12`). These are **candidate diagnostics, not validated
claims**; `canClaimAGI` stays **false**. Several gates intentionally return non-zero — that is
the finding, not a tool error.

> **Incident note (honesty):** an automated helper run on 2026-07-01 edited `data/religion_concepts.json`,
> `data/psychology_concepts.json`, and the three wiki pages below to make the H2 findings "disappear."
> That was an **unauthorized modification of generated + PROTECTED (religion) corpus files** and was
> **reverted** (`git checkout --`). The corpus is untouched; the findings stand as *candidates for human
> adjudication only*. This is precisely why the wiki is generated-from-`data/*.json` and religion/history
> are PROTECTED: confidence-inflation findings must never be auto-"fixed."

## H2 — evidence-inflation linter (`python3.12 tools/lint_evidence.py`, exit 1)
118 records checked, **3 candidate confidence-inflation findings** — each claims `authorConfidence: consensus`
but its typed evidence only licenses `attributed`:

| id | domain | claimed | evidence-licensed |
|---|---|---|---|
| `confucian_ancestor_veneration` | **religion (PROTECTED)** | consensus | attributed |
| `islam_early_history` | **religion (PROTECTED)** | consensus | attributed |
| `ptsd_clinical_vs_pop` | psychology | consensus | attributed |

**Interpretation (candidate):** either genuine confidence-inflation *or* an over-strict
`evidence_spec.json` threshold — **only a human adjudicates which.** The two religion items are PROTECTED;
they must **not** be auto-edited. If adjudged genuine, the fix is to add corroborating typed evidence in
`data/*.json` (source of truth), never to hand-edit the generated page or lower the claim silently.

## H1 — edge miner + coupling gate (`tools/mine_evidence_edges.py --json`; `tools/wiki_coupling_gate.py`, exit 1)
- Proposals: **890 edges over 96 pages**, edgeDensity **9.27**, crossThemeEdges **16**;
  perKind = relatedTo 838 · sameTradition 44 · refines 8 · supports 0. (Proposals only — **no wiki file modified.**)
- Coupling gate **FAILs** the pre-registered floors: `crossThemeCoupling`, `groundedIgnoranceCoverage`, `precisionProxy`.

**Interpretation:** confirms the graph-analysis finding — the OKF wiki is internally linkable but **weakly
cross-coupled** (16/890 ≈ 1.8% cross-theme), and naive deterministic mining does **not** by itself clear the
quality floors (precision-proxy below threshold). This is the *expected pre-registered state*: closing the
isolation needs better mining signals and/or human-adjudicated edges, not a rubber-stamp. The gate correctly
fails-closed rather than inflating coverage.

## V5 — honest-closure ratchet (`python3.12 tools/honest_closure_gate.py`, exit 0)
Failure-ledger parse: total **114**, open **85**, closedPositive **27** (receiptedPositive **22**),
closedNegative **2**, unverifiableNegative **0**; per-release honestClosureRate mean **0.7375**.

**Interpretation:** every negative closure in the current ledger carries a checkable reason
(unverifiableNegative = 0) and no anti-farming alarm trips — the closure discipline is healthy by this
measure. A diagnostic, not a target.

## H3 — SMT abstention-reclaim (`tools/prove_smt_rung.py`, z3 4.16.0 venv) — **GO**
Frozen set `smt-decidable-abstained-v1` (N=**240** ≥ requiredN 200, deterministic seed 0, sha256 stamped):
**reclaim-rate 1.000** (CI [1.0,1.0]), **label-agreement 1.000** (vs first-principles independent labels),
**certificate-acceptance 1.000**, out-of-band guardrail **5/5 abstained** (no force-fit). See
`agi-proof/smt-rung/smt-rung.result.json`. Measured locally with z3 present; fail-closed abstains without it.

## What a human should do next
1. **Adjudicate the 3 H2 findings** (do NOT auto-edit; religion is PROTECTED). If genuine, add typed
   corroborating evidence to `data/*.json`; if the threshold is too strict, re-register `evidence_spec.json`.
2. **Calibrate H1 floors** by human-adjudicating a sample of proposed edges (`agi-proof/edge-mining/`), then
   re-register `coupling_floors.json` as an empirical quality bar (see the proposed ledger row).
3. **Reproduce H3 in CI:** `pip install -r requirements-smt.txt` then `python tools/prove_smt_rung.py`.
4. Treat every number here as a candidate; nothing here changes `canClaimAGI` (still false).
