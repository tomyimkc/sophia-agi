# Agent-Trajectory Evaluation & Specialized-Domain Data Packs

> Status: **candidate** — shipped as fail-closed machinery with offline tests; the
> LLM-judge tiers are *measured under the no-overclaim gate*, not asserted. No
> headline number is published from this work yet.

## Why this, why now (market signal)

DeepSeek / High-Flyer's 2026 hiring drive — 33 roles across seven categories — is
an explicit pivot from "use a model" to **agentic AI plus owning the data/eval
stack**. The loudest signal is three brand-new role titles: **Agent Deep Learning
Algorithm Researcher, Agent Infrastructure Engineer, and Agent Data Evaluation
Expert**, alongside specialized-domain data roles for **medicine and law**
([Bloomberg](https://www.bloomberg.com/news/articles/2026-03-24/deepseek-s-latest-job-postings-highlight-pivot-to-agentic-ai),
[SCMP](https://www.scmp.com/tech/big-tech/article/3358394/deepseek-hiring-spree-chinese-ai-firm-seeks-newcomers-it-pursues-agi)).

The market is paying top-of-band for **agent-trajectory evaluation** and
**domain-specific (law/medicine) data discipline** — which is exactly what
Sophia's thesis already is, lifted from a single claim to a whole agent run. This
work makes that fit explicit. It hits three of the seven hiring categories (deep
learning research / model-data strategy / specialized-domain data) and both new
"Agent" role titles, mostly by **recombining modules the repo already had**.

## What shipped

### 1. Agent-trajectory evaluator — `agent/trajectory_eval.py`

Sophia's existing gates score a *single* answer. An agentic system emits a
*trajectory* — tool calls, observations, intermediate assertions — and can pass
every single-answer check at the end while the fabrication happened **mid-plan**:
a claim asserted at step 4 that no observation up to step 4 supported, then
reasoned on top of. `evaluate_trajectory` scores a run **step by step** and emits
a fail-closed verdict.

Per claim-bearing step:

| Verdict | Meaning |
| --- | --- |
| `blocked` | the claim trips a hard Sophia provenance violation (merged lineage, fabricated attribution, false arithmetic) — via `agent.guarded.check_claim` |
| `ungrounded` | the claim has **no** supporting evidence available at this step — the mid-plan fabrication this exists to catch |
| `unverified` | evidence exists but the offline judge cannot confirm entailment — abstained, not condemned |
| `grounded` | supported by available evidence and trips no gate |
| `skipped` | no asserted claim to check |

Grounding has two modes, both fail-closed: **explicit `cites`** give a narrow,
audited warrant (forward / self / observation-less citations earn nothing, as in
`proof_carrying_reasoning`); **no `cites`** grounds the claim against every prior
observation the agent actually had in context, so a missing citation is not a
penalty — only the *absence of any supporting observation* is.

The trajectory verdict: `blocked` if any step is blocked, `abstain` if any step is
ungrounded/unverified (the run is **withheld, not certified**), `accept` only if
every claim-bearing step is grounded. It also reports `faithfulnessScore`
(grounded / claim-bearing steps) and `firstUnfaithfulStep`.

**The support judge is pluggable.** The default is a deterministic, offline,
embedding-free **lexical** judge that confirms strong overlap and otherwise
*abstains* (it never calls a paraphrase a fabrication). Inject
`make_entailment_judge(spec)` for a measured LLM run — and, like every judged tier
in this repo, any headline number from it must clear the no-overclaim gate
(≥2 judge families, κ ≥ 0.40, ≥3 runs, CIs) before it is published.

### 2. Medical citation-discipline pack — `agent/medical_faithfulness.py`

The medicine sibling of `agent/legal_faithfulness.py`, because the market is
hiring domain-data discipline for *both* law and medicine. Same two failure modes:

- **Existence (deterministic, always runs).** A model invents `PMID 99999999` or a
  plausible DOI — the medical analogue of *Mata v. Avianca*. `medical_citation_exists`
  checks PMIDs / DOIs / NICE guideline IDs against a register
  (`data/medical_register.json`; a live PubMed/Crossref/guideline resolver in
  production) and fails closed on anything unresolvable.
- **Faithfulness (judged, measured).** A *real* trial cited for a claim its result
  does not establish (e.g. a secondary-prevention statin trial cited to start a
  statin in an asymptomatic low-risk adult). Pluggable judge; default abstains so
  the wiring is provable offline. **Citation review only — not medical advice.**

### 3. Surfaces

- **MCP tools:** `sophia_trajectory_eval`, `sophia_medical_citation_check`
  (`sophia_mcp/server.py`, impls in `sophia_mcp/tools_impl.py`).
- **Skill:** `agent_trajectory_eval` (`skills/trajectory_eval.py`) — maps the
  evaluator's fail-closed verdict onto the skills contract (`accept`→`ok`,
  `blocked`→`flagged`, else `held`).
- **Tests:** `tests/test_trajectory_eval.py`, `tests/test_medical_faithfulness.py`.

## Honest limits (failure ledger discipline)

- The default trajectory judge is **lexical**, so it confirms only high-overlap
  grounding and abstains otherwise; it does **not** detect semantic entailment or
  subtle unfaithfulness without an injected LLM judge — and that judge is
  unvalidated here.
- The medical register is a **tiny illustrative snapshot**, not a citator; it
  proves the wiring, not coverage. A fabricated-but-plausible citation outside the
  snapshot is reported as non-existent (correct here, but only because the snapshot
  is small) — production needs the live resolver.
- No third-party agent-faithfulness benchmark has been run yet. The natural next
  step (see `agi-proof/`) is a held-out trajectory pack + ≥2 judge families to put
  a measured number behind the evaluator.

## Tier 2 — shipped: agent-faithfulness benchmark

A deterministic, reproducible benchmark now scores the evaluator itself:

- **Pack:** `benchmark/agent_faithfulness.json` — 13 hand-labelled trajectories
  (grounded / fabrication / weak-evidence / provenance-violation / no-claims /
  empty). Labels are set by case *design*, independent of the evaluator, and the
  pack is stamped first-party seed.
- **Scorer:** `provenance_bench/agent_faithfulness.py` — verdict accuracy (+ Wilson
  95% CI), detection precision/recall/F1 on the "should-not-certify" class, and
  culprit-step localization accuracy.
- **Runner / report:** `tools/run_agent_faithfulness_bench.py --write` →
  `agi-proof/benchmark-results/agent-faithfulness.public-report.json`.
- **Result (seed):** verdict accuracy **1.0** (Wilson-95 **[0.77, 1.0]**, N=13),
  detection **F1 1.0**, localization **6/6** — deterministic, no judge. The wide CI
  lower bound is the honest cost of N=13; this is *not* a multi-family-gated headline.

Because the default judge is lexical and deterministic, this benchmark needs no
model and no multi-family gate — the remaining honesty caveat is **label
provenance** (first-party), disclosed in the pack and report. The OPEN item is a
**third-party trajectory pack** to remove that caveat.

## Judged held-out benchmark (entailment judge, under the gate)

The deterministic bench proves the wiring; it cannot settle whether a claim is
*entailed* by evidence when surface forms diverge. That tier now exists:

- **Sealed held-out pack:** `benchmark/agent_faithfulness_heldout.json` (N=9),
  sha256-committed in `agi-proof/hidden-reviewer-packs/agent-faithfulness-heldout.seal.json`.
  Designed to be judge-discriminating in **both** directions — paraphrase /
  multi-hop cases (lexically divergent but entailed) and negation / scope
  distractors (lexically overlapping but NOT entailed).
- **Judged scorer:** `provenance_bench/agent_faithfulness_judged.py` — a binary
  *certify* decision per case, scored under the **same no-overclaim gate as the
  legal pack** (≥2 judge families + Cohen's κ ≥ 0.40 + ≥3 runs + CI above chance),
  reusing `provenance_bench/aggregate` + `consensus`.
- **Measured value of the judge:** the deterministic lexical baseline scores **33%**
  on this pack (it wrongly abstains on the paraphrase cases and wrongly certifies
  the distractors); a scripted entailment **oracle** scores **100%** — a +67-point
  value-add, proving the judge is necessary and the harness measures it. (Tested
  with deterministic scripted judges; no real model is called in CI.)
- **Honest status:** the **committed** artifact is the offline **mock** run
  (`validated=False`); a real multi-family model run and a third-party-authored pack
  are OPEN (failure ledger: `agent-faithfulness-judged-not-yet-validated-2026-06-25`).
  Run it for real with:
  `tools/run_agent_faithfulness_judged.py --judges <2 vendor families> --runs 3 --write`.

## Capability proof (field requirements)

`agi-proof/field-requirements/` maps the seven DeepSeek hiring categories (and the
three new "Agent" role titles) to concrete repo artifacts, and makes the map
**machine-checkable**: `tools/verify_field_requirements.py` fails CI if any cited
module, test, or evidence file is missing. See that directory's README.

## Next

- A **third-party / held-out trajectory pack** + a second (LLM-entailment) judge to
  put a gated number behind the evaluator and lift the first-party caveat.
- Live PubMed/Crossref resolver for the medical pack; the same shape extends to
  other specialized domains (finance, non-English sources).
- A corpus-cleaning pass: run the gate over a dataset and emit accept/abstain/block
  labels as a reproducible pre-training/RL **data filter**.
