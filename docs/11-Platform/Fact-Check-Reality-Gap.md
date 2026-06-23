# Fact-Check Reality Gap v1

Status: implemented as an **offline deterministic candidate eval** plus optional keyless live backends. It is not Level-3 AGI evidence and is not a hidden-reviewer run.

## What changed

Sophia's out-of-wiki gate no longer has to choose between "found in internal wiki" and silence. The new path is:

1. decompose generated text into atomic claims;
2. run deterministic verifiers for math, dates, code, DOI, and URL existence;
3. query external source adapters when the wiki has no coverage;
4. require entailment, source-family independence, and calibrated confidence floors;
5. emit accepted external claims only as quarantined learning candidates;
6. recheck candidates before provisional promotion; canonical OKF/wiki writes remain disabled.

## Files

| Purpose | Path |
|---|---|
| Keyless/fixture source adapters | `agent/live_sources.py` |
| Eval metrics, ECE/Brier, Wilson CI, empirical floors | `agent/fact_check_eval.py` |
| Held-out labeled pack (N=43; true/false/unknowable) | `eval/fact_check/heldout_v1.jsonl` |
| Offline source fixtures | `eval/fact_check/fixtures_v1.json` |
| Eval CLI | `tools/run_fact_check_live_eval.py` |
| Quarantine/recheck flywheel | `agent/fact_check_flywheel.py`, `tools/run_fact_check_flywheel.py` |
| Reflexive AGI self-gate | `agent/reflexive_self_gate.py`, `tools/run_reflexive_self_gate.py` |
| Candidate artifacts | `agi-proof/fact-check-live/`, `agi-proof/self-gate/` |

## Commands

Offline deterministic candidate eval:

```bash
python tools/run_fact_check_live_eval.py
```

Optional keyless live run using Crossref/Wikidata/URL resolution:

```bash
python tools/run_fact_check_live_eval.py --live \
  --out agi-proof/fact-check-live/fact-check-live-eval.live.local.json
```

Learning candidate quarantine + independent recheck:

```bash
python tools/run_fact_check_flywheel.py
```

Reflexive no-overclaim scan:

```bash
python tools/run_reflexive_self_gate.py
```

## Current offline candidate result

From `agi-proof/fact-check-live/fact-check-live-eval.public-report.json`:

- N = 43 held-out labeled cases.
- Fabrication rate = 0.0 on false/unknowable cases.
- Over-abstention on true cases = 0.0 in the fixture run.
- Correct abstention on unknowable cases = 1.0.
- Resolved-answerable accuracy = 0.9697.
- Resolved-only calibration: ECE = 0.0856, Brier = 0.0113.
- Empirical floors derived for ≤1% target fabrication:
  - normal risk: 0.80
  - high risk: 0.84

Boundary: this is a **candidate/offline fixture run** (`candidateOnly=true`, `level3Evidence=false`). It proves wiring, metrics, and fail-closed behavior; it does not prove open-world AGI.

## Source rules

- Wikidata/Crossref are external keyless backends, not Sophia-internal evidence.
- Crossref verifies DOI existence, not the truth of a paper's claims.
- Wikidata authorship retrieval is narrow and structured; non-authorship claims need other adapters.
- Fixture entailment is an offline cached annotation for deterministic CI, not model consensus.
- High-risk economics / AGI-incentive claims require at least 3 independent entailing source families and confidence above the high-risk floor.

## Verifier synthesis slot

`agent.fact_check_gate.fact_check_claim(..., synthesized_verifier=...)` now has an optional slot for an **already admitted** synthesized verifier. The verifier must have passed the existing `agent/verifier_synthesis.py` contract: AST sandboxing and held-out validation. If it is absent, non-applicable, or errors, the layer returns `held` and active grounding continues fail-closed.

## Learning flywheel boundary

`tools/run_fact_check_flywheel.py` extracts `learningCandidate` records from accepted claims, appends them to quarantine, and independently rechecks them. Passing candidates are only `promoted_provisional`; `canonicalWikiWrite=false` remains a hard boundary. This prevents poisoning, staleness, and circular grounding.

## Reflexive self-gate

`tools/run_reflexive_self_gate.py` scans Sophia's own README/results/AGI-proof docs for status overclaims. It accepts candidate/no-overclaim wording and rejects unqualified "Sophia is proven AGI"-style claims. Current report keeps `canClaimAGI=false`.

中文摘要：Reality Gap v1 已把 out-of-wiki claim 連到 deterministic verifier、keyless Wikidata/Crossref/URL backend、N=43 held-out eval、ECE/Brier/derived floor、quarantine flywheel、以及 Sophia 自我 no-overclaim gate。現階段是 candidate/offline wiring，不是 Level-3 AGI 證據。
