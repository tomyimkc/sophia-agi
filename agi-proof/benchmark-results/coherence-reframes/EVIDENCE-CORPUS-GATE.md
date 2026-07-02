# Evidence-corpus acceptance gate — the redirect-and-fund capstone

**Verdict: NO-GO — the NLI-entailment mechanism does not clear the frozen gate even on curated,
in-domain, fact-bearing evidence. `candidateOnly:true`, `canClaimAGI:false`.**

This is the held-out acceptance instrument the maintainer-AI asked for: *"build the evidence corpus,
not just the eval pack — and make the curated pack its held-out acceptance instrument."* It closes
the sophia-domain grounding line that D/D2/D3 (`ARC-SUMMARY.md`) left open, from the opposite side:
D showed NLI over-abstains on sophia's **terse** wiki prose; this shows it still over-abstains on an
**independent, rich, multi-sentence** evidence corpus — so terse prose was never the cause.

Harness: [`tools/nli_evidence_corpus_gate.py`](../../../tools/nli_evidence_corpus_gate.py).
Machine report: [`EVIDENCE-CORPUS-GATE.report.json`](EVIDENCE-CORPUS-GATE.report.json).
Corpus: [`data/attribution_corpus.claims.json`](data/attribution_corpus.claims.json).

## Design (per the amended pre-registration)

- **Corpus (curator ≠ gate-runner).** 99 canonically-attributed works across philosophy (45),
  political theory (17), history (12), religion (12), science (7), economics (6), assembled by a
  **separate curator agent**. Evidence = independent multi-sentence **Wikipedia lead** passages
  (198 passages fetched live, workers=4 + retry/backoff so the fetch recovered fully — the earlier
  run dropped 57/99). Evidence is independent of the claim source; no shared hostnames (each passage
  gets a distinct `ext{i}.example.org` publisher so the gate's ≥2-independent-domain rule is real).
- **Claims.** Blunt attributions `"{author} wrote {work}"` — supported (canonical author) + refuted
  (a plausible sibling author). Blunt phrasing was chosen to **match Wikipedia's blunt lead prose**
  (D3 used hedged phrasing against hedged sophia prose; here the evidence is blunt, so the claim is
  too). This is a phrasing-alignment choice fixed **before** the final run, not a post-hoc knob.
- **Two arms.**
  - **Arm 1 — evidence-as-curated (mechanism ceiling):** attach each work's own passages directly.
    Best case the mechanism can ever see.
  - **Arm 2 — evidence-through-retrieval (the deployable system):** retrieve top-k passages from the
    *whole* corpus via (a) a semantic index (all-MiniLM = deployable-if-upgraded) and (b) sophia's
    **production** hash index. **Default-on requires THIS arm to pass** — over-abstention lives here.
- **Frozen gate.** NLI (`build_nli_entailment`, `cross-encoder/nli-deberta-v3-base`) vs the **healthy
  semantic incumbent** (`build_semantic_entailment`, all-MiniLM cosine) through the real
  `fact_check_gate.external_ground`. Neither the primitive nor the gate default was changed.
- **Guards (pre-registered):** paired ΔF1 @ matched coverage ≥ +0.05, CI∌0, 3 seeds; incumbent-health
  coverage ≥ 0.10; **answerable-coverage drop ≤ 0.01** (fraction of *supported* claims admitted — the
  correct abstention-guard metric); manipulation ≥ 50% fact-bearing; MDE published before running.

## Result (n=134 claims, 67 supported; MDE 0.124)

| Arm | ΔF1 (NLI − semantic) | CI95 (seed0) | primary ΔF1 pass | answerable-cov NLI vs sem (drop) | abstention guard | incumbent healthy | manip. fact-bearing |
|---|---|---|---|---|---|---|---|
| **1 — curated (ceiling)** | **+0.233** | [0.110, 0.344] | ✅ (all 3 seeds) | 0.119 vs 0.582 (**0.463**) | ❌ | ✅ (0.46) | ✅ 0.85 |
| **2 — retrieval-semantic (deployable / default-on)** | **−0.012** | [−0.093, 0.056] | ❌ | 0.075 vs 0.836 (**0.761**) | ❌ | ✅ (0.79) | ✅ 0.91 |
| **2 — retrieval-hash (sophia production)** | **−0.058** | [−0.200, 0.062] | ❌ | 0.000 vs 0.343 (**0.343**) | ❌ | ✅ (0.28) | ✅ 0.85 |

## What the gate decided, and why it is robust

1. **The deployable arm shows no win at all.** Both retrieval arms have ΔF1 point estimates **≤ 0**
   (−0.012, −0.058) with CIs centered on / spanning zero. Default-on fails its own load-bearing clause
   before any guard is even consulted. **Default-on cannot be recommended.**

2. **Even the ceiling is disqualified — and not for lack of evidence.** Arm 1's F1 lift is real
   (+0.233, CI excludes 0, stable across 3 seeds), but it is bought **entirely by over-abstention**:
   NLI admits only **12%** of true attributions vs the incumbent's **58%** (drop 0.46 ≫ the 0.01
   guard). Crucially the **manipulation check passes on every arm (0.85–0.91 fact-bearing)** and the
   incumbent is healthy — so the evidence is genuinely rich and fact-bearing. NLI over-abstains
   *anyway*. A grounding admitter that refuses ~88% of answerable questions is not deployable; the
   abstention guard the maintainer-AI pre-registered is exactly what catches it.

3. **The NO-GO is overdetermined and power-robust.** MDE is 0.124 (> the 0.05 aspiration; I committed
   to one final run, no further tuning, and state this honestly). Power does not rescue any arm: the
   retrieval arms' effects are at/below zero (nothing positive to miss), and the curated effect
   (+0.23) is far above MDE and clearly real — just disqualified by coverage. Two guard clauses were
   not wired into this harness (two-family κ, explicit fail-closed-on-no-evidence); both are **GO
   preconditions**, so their absence can only keep the verdict at NO-GO, never manufacture a GO.

## Where this leaves the arc

The FEVER positive (NLI ≫ coherence, [`NLI-ENTAILMENT.public-report.json`](NLI-ENTAILMENT.public-report.json))
was **retrieval-bound**: it required pre-selected, single-sentence, fact-bearing evidence. Given a
realistic retrieval path — or even a curated ceiling of rich fact-bearing passages — NLI-as-grounding-
admitter over-abstains and does **not** beat sophia's healthy semantic incumbent. The default-on
grounding line is **closed**. The mechanism is retained where it is honestly earned: as a
**contradiction-only** signal (`build_hybrid_entailment` rejects on entailed contradiction while
leaving admission to the incumbent), which never triggered a coverage penalty. The NLI primitive
remains on `main` behind the `EntailmentFn` seam, injectable, fail-closed, **off by default**.
