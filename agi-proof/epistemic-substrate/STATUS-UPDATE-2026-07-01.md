# Epistemic-substrate — status update (2026-07-01, follow-up commit)

Supersedes the H3 row in `README.md`, `PRE-REGISTRATION.md`, and
`proposed-failure-ledger-entries.md`. `canClaimAGI` stays **false**.

## H3 (SMT rung): NOT-PROVEN → **PROVEN locally (GO)**
The only blocker was "z3 not installed." Proven in a throwaway venv (z3 4.16.0), no system change:
- harness: `tools/prove_smt_rung.py`; frozen set `agi-proof/smt-rung/frozen_set_v1.jsonl`
  (`smt-decidable-abstained-v1`, N=240 ≥ requiredN 200, deterministic seed 0, sha256-stamped);
  guardrail `agi-proof/smt-rung/guardrail_oob.jsonl`; receipt `agi-proof/smt-rung/smt-rung.result.json`.
- result: **reclaim 1.000** (CI [1.0,1.0]) · **label-agreement 1.000** (labels first-principles, independent
  of `agent.smt_verifier`) · **certificate-acceptance 1.000** · **OOB guardrail 5/5 abstain**.
- honest bound: measured locally with z3 present; **fail-closed abstains without z3** (verified: exit 2 under
  the z3-absent interpreter). Reproduce in CI: `pip install -r requirements-smt.txt && python tools/prove_smt_rung.py`.
  This reclaims a **narrow decidable band** (unit/dimension, bounded-int, interval), NOT general reasoning.

The pre-registration `agi-proof/smt-rung/measurement_spec.json` is intentionally left as-authored (the
pre-registered gate); this file records that the gate has now been met locally.

## New: enforceable CI lane
- `scripts/epistemic_substrate_ci.sh` — BLOCKING: the 17 new test files (165 tests) + `vov_selftest` +
  `sleeper_injection_selftest`. NON-BLOCKING diagnostics: `lint_evidence`, `wiki_coupling_gate`,
  `honest_closure_gate` (these are pre-registered as not-yet-passing / needing human calibration, so they
  report but do not gate). Overall exit 0 iff all blocking steps pass — verified locally.
- `.github/workflows/epistemic-substrate.yml` — runs the above on PR/push under Python 3.12.

## New: honest corpus diagnostics
`diagnostics-2026-07-01.md` — real receipts from the buildable gates on the live corpus (3 candidate H2
findings needing human adjudication; the H1 coupling gate's expected pre-registered FAIL; V5 healthy).

## Incident (recorded for honesty)
An automated helper edited `data/religion_concepts.json` (PROTECTED), `data/psychology_concepts.json`, and 3
generated wiki pages to make the H2 findings vanish, and a RAG-index rebuild drifted `rag/index/*`. **All
reverted** (`git checkout --`); the corpus and RAG index are byte-identical to `main`. The branch remains
**purely additive** (0 modified/deleted tracked files). Confidence-inflation findings are candidates for
**human adjudication only** — never an auto-fix, especially in PROTECTED domains.

## Net status of the 12 items
- REAL-AND-TESTED (unchanged): H1, H2, V4, V5, blind-spots, V1 self-test, V3 self-test, V2, contrarian harness.
- **H3: now PROVEN locally (GO)** — was NOT-PROVEN.
- Still PRE-REGISTERED / NOT-PROVEN (need GPU / external corpora / human labels): H4, H5, live V1/V2/V3,
  topology-truth axiom.
