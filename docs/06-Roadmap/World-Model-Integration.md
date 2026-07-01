# World-Model Integration — error-centric intelligence, applied to the gate

`canClaimAGI` **false.** These are **candidate research prototypes**, offline-invariant tested, not
validated capabilities. They translate the 2026 world-model / error-centric-intelligence theses into
this repo's verifier-gated idiom.

## The theses (grounded)
- **Nature, "change in tack":** LLMs lack three abilities for AGI — *generalization*, *representation*
  (a **world model** that anticipates the consequences of decisions), and *selection*.
- **Nature (2026), "Statistical approximation is not general intelligence"** + skeptic pieces:
  curve-fitting on observations ≠ intelligence.
- **arXiv 2510.15128, "Error-Centric Intelligence: Beyond Observational Learning":** five mechanisms —
  **error detection → verification → intervention → causal discovery → world models** — with *error* as
  the driver and *intervention* (not observation) as what reveals causal structure.

## The honest map
This repo is already the front half of error-centric intelligence (error detection, verification,
selection/abstention) on the OUTPUT side. The gaps are **world model** (anticipate, don't just catch),
**intervention/causal discovery**, and **generalization**. These prototypes attack the first two; the
generalization gap is logged as still-open, not papered over.

## The prototypes (all offline-invariant tested)
| # | Module | Thesis mapping | What it does |
|---|---|---|---|
| 1 | `agent/verifiability_model.py` | *representation* — anticipate consequences | logistic world model of P(verifier passes) from knowability features; abstains **proactively** (before generating), generalizes the feature not the string |
| 2 | `tools/causal_ablation.py` | *intervention → causal discovery* | injects/removes a knowability feature and ranks each by its **causal** effect on abstention (null padding ~0) |
| 3 | `tools/counterfactual_traps.py` | *intervention* | minimal knowable↔unknowable counterfactuals; **intervention_consistency** = does abstention FLIP with the edit? (world-model gate 1.0 vs surface 0.0) |
| 4 | `agent/constrained_world_model.py` | *world model grounded in reliable dynamics* | forward-simulates A=L+E; predicts the determined value, **abstains when under-determined**, flags violations — a physics-informed world model using a verifier as the law |
| 5 | `agent/active_query.py` | *selection* + verification-by-query | turns abstention into a **targeted request for the specific missing evidence** (names the source it lacks) |
| 6 | `agent/process_world_model.py` | *anticipate during the process* | running validity confidence over reasoning steps; flags the first wrong step **before** the final verifier would (early abstain/re-route) |

## How they compose with the existing stack
- #1 upgrades the conformal/abstention gate from **reactive** to **anticipatory** (decide by forecasting the verifier verdict).
- #2 + #3 upgrade `sophia_autoresearch` + the replication pack from **observational** to **interventional** — the strongest move against A4 (first-party-only) and the world-model claim.
- #4 makes the deterministic verifiers (finance/chemistry laws) into a **simulatable** world model, not just a checker.
- #5 makes abstention **useful** (active information-seeking, not a dead end).
- #6 makes process supervision **predictive** (early error localization), extending `okf_trace.locate_wrong_step`.

## Honest limits (do not bury)
- All six are **candidate** — feature-based/tiny by design, tested on synthetic + small hand sets, not
  on a trained model. Each needs a real ablation + pre-registration before it earns `adopted:true` in
  `recipe_spec.json`.
- The **generalization** ability (abstract rules across tasks) is **not** solved here — logged as the
  deepest open gap in the ledger.
- A world model is a research direction, not a config flag; these are the honest first rungs.

## First rungs to promote
Prototype-to-promotion order: **#3 counterfactual traps** (cheap, testable now, strengthens A4) →
**#1 verifiability model** (highest new capability, trains on data you already generate) → #2/#4/#6 as
deeper bets. Each moves candidate→adopted only via an ablation + a passing gate (`tools/lint_recipe.py`).
`canClaimAGI` stays false.
