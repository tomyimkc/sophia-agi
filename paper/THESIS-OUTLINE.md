# Thesis / Paper Outline — Sophia: Provenance-Aware, Verifier-Gated Reasoning that Abstains Instead of Fabricating

**Author of record:** Yim Kin Cheong (ORCID 0009-0005-9520-0033), Hong Kong.
**Status:** defensive-publication outline. Honestly framed: Sophia is an *AGI-candidate*
provenance system with *bounded, measured* results — **not** proven AGI (`canClaimAGI = false`).
This document is a structure for a thesis/preprint built from `paper/sophia-whitepaper.md` plus
the 2026-06-28 verification-toolkit work; it is not itself the thesis.

> **Framing rule (non-negotiable, and a selling point):** every claim states its evidence tier
> (validated / candidate / illustrative) and its independence tier. The thesis's credibility
> rests on what it declines to claim. Over-claiming AGI would forfeit that.

---

## Abstract (≈200 words)
LLMs fabricate attributions confidently. Sophia is a provenance-aware, verifier-gated reasoning
layer that, given a claim, verifies it against machine-readable sources and then **accepts,
abstains, or blocks** — failing closed rather than guessing. Contributions: (1) a measurement
discipline (a pre-registered no-overclaim gate) under which every number is reported with judge
independence, agreement, runs, and CIs; (2) a layered, **independence-labelled** verification
toolkit covering distinct fabrication modes; (3) an honest characterisation of the *limits* of
open-world verification — a precision/recall law ("no free lunch") and the discovery-and-
correction of one of the system's own over-claims. Results: a validated provenance/abstention
delta on attribution traps, and high-independence verifiers with 0% clean over-block on their
target modes. We do **not** claim general intelligence.

## 1. Introduction
- 1.1 Problem: attribution hallucination (false author/source, merged traditions, fabricated
  citations — the *Mata v. Avianca* failure mode) and why fluency makes it dangerous.
- 1.2 Thesis statement: trustworthiness comes from **fail-closed abstention + layered independent
  verification + honest measurement**, not from a single catcher or a capability claim.
- 1.3 Contributions (enumerated, each tagged with its evidence tier).
- 1.4 Explicit non-claims: not AGI; results are bounded; packs are self-authored pending
  third-party replication.

## 2. Background & related work
- Hallucination & faithfulness; selective prediction / calibration; retrieval-augmented
  generation; fact-checking corpora (ClaimReview); knowledge bases (Wikidata, Crossref);
  NLI/entailment; LLM-as-judge and its failure modes; defensive publication & provenance.

## 3. Method — the gate
- 3.1 Retrieve → reason → verify → accept/abstain/block.
- 3.2 Belief & provenance graph (min-over-chain confidence, retraction, counterfactual removal).
- 3.3 Verifier classes: deterministic (no model) vs model-based (held to the no-overclaim gate).
- 3.4 Calibrated abstention (self-consistency selective prediction).
- 3.5 Fail-closed governance contract.

## 4. Method — the independent-verification toolkit (the session's core contribution)
For each layer: mechanism · independence tier · failure mode · what it does NOT cover.
- 4.1 Debunk detection: keyword → LLM/NLI detector via a meta-labeler (detection 0→100% live).
- 4.2 Core-claim verification (verify the load-bearing claim, not every atomic side-claim).
- 4.3 Citation existence (Crossref/DOI) — fabricated studies; *high* independence.
- 4.4 Attribution-swap (Wikidata creator/author/discoverer) — wrong-creator; *high*.
- 4.5 Source faithfulness (multi-judge entailment over an independent source) — misstated
  findings; *medium*, strict-majority consensus, fail-open on insufficiency.
- 4.6 Layering & routing; **fail-open on ignorance**; per-verdict independence labelling.

## 5. Measurement discipline
- The pre-registered no-overclaim gate (≥2 judge families, κ≥0.40, ≥3 runs, CIs); answer≠judge
  separation; bootstrap CIs; hidden-eval prompts never published.
- **Self-correction as method:** the independent judge audit that revised a rate down; and the
  withdrawn 70.6%→5.9% over-block artifact. Reproducibility of negatives, not just positives.

## 6. Results
- 6.1 Provenance delta (validated): 36.1%→23.6% gated, Δ12.5% [5.6,19.4], 0% FP.
- 6.2 Calibration/abstention (validated): 0% fabrication on traps vs 16.7–25% raw.
- 6.3 Selective prediction on **external** public data (SimpleQA): cross-model validated.
- 6.4 Verification toolkit (candidate/live): high-independence verifiers at **0% over-block**;
  attribution-swap 3-run CI 10.8% [9.3,11.6] caught / 0% over-block; the coverage-bounded
  "no free lunch" matrix; the recall/independence trade.
- 6.5 What failed and why (the falsified-theory ledger): the anti-fabrication gate adds no value
  on strong models (they debunk); a learned world model and Lean proof search results.

## 7. Limitations & threats to validity
- Self-authored synthetic packs; one operator's keys; single-run candidates vs ≥3-run; oracle
  coverage is domain-bounded; model-based layers are lower-independence; entity disambiguation
  bounds recall; **no third-party clean-room replication yet** — the reason `canClaimAGI=false`.

## 8. Reproducibility
- Keyless deterministic tests; live bench commands + env vars; `agi-proof/verification-replication/`
  (REPRODUCE.md, EXPECTED-RESULTS.json, decontamination checklist, manifest checker); the
  failure ledger and capstone as first-class artifacts.

## 9. Conclusion
- Trustworthiness = labelled-independence layered verification + fail-closed abstention + honest
  measurement + a correction discipline. The remaining frontier is external (independent
  replication on an independently-authored pack), by design.

---

## Venue & format options (for the author to choose)
- **Defensive publication / priority** (recommended, already in place): tag release 0.11.0 →
  Zenodo version DOI (mints under your ORCID + date). This is the rights-protecting step.
- **Preprint:** arXiv (cs.CL / cs.AI) or SSRN — adds scholarly weight on top of the DOI.
- **Thesis:** if pursued for a degree, the structure above maps to standard chapters; keep the
  honest framing. A thesis adds academic credibility, not additional legal rights.
- **Peer-reviewed paper:** a workshop/conference on trustworthy NLP / faithfulness would suit the
  measurement-discipline + verification-toolkit framing.

> **IP note (not legal advice):** publishing establishes *priority / prior art* (defensive
> publication) and blocks others from patenting the disclosed methods — it does **not** grant
> exclusive rights, and prior public disclosure (this Apache-2.0 repo + Zenodo) generally
> forecloses your own patent on already-disclosed parts (immediate in most of the world; ~12-month
> grace in the US). If exclusive/commercial rights on any *undisclosed* novel mechanism matter,
> consult a patent attorney **before** any further public disclosure.
