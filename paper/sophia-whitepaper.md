# Sophia — the Wisdom Gate: A Provenance-Aware, Verifier-Gated Reasoning Layer That Abstains Instead of Fabricating

**Author:** Yim Kin Cheong (tomyimkc), Hong Kong — ORCID: [0009-0005-9520-0033](https://orcid.org/0009-0005-9520-0033)
**Version:** 0.9.0 · **Date:** 2026-06-22 (first public release)
**Code:** https://github.com/tomyimkc/sophia-agi · **Site:** https://tomyimkc.github.io/sophia-agi/
**Dataset:** https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus
**License:** Apache-2.0 (code, tools, benchmarks)

> **Status / scope.** This is a research program toward grounded, machine-checked
> reasoning — **not a claim of AGI**. Pre-registered thresholds are stated and, where
> not yet met, said so plainly. The deliverable is the honest machinery (verifiers,
> abstaining gate, governance contract) and the measured data, with a public failure
> ledger. This whitepaper is a defensive publication: it fixes an authorship + date
> record for the method described below.

---

## Abstract

Modern large language models (LLMs) confidently produce **attribution
hallucinations**: they credit the wrong author for an idea, merge distinct
intellectual traditions, treat legendary figures as literal authors, or cite
authorities that do not exist — and then reason on top of those errors. We present
**Sophia (the Wisdom Gate)**, a provenance-aware, verifier-gated reasoning layer that
**abstains instead of fabricating**. The core control loop is:

```
claim  →  verify against sources  →  accept · abstain · block
```

Sophia couples (i) a queryable **belief/provenance graph** with confidence
propagation, retraction, and counterfactual analysis; (ii) **deterministic and
model-based verifiers** that gate claim promotion; (iii) **calibrated abstention** that
downgrades low-confidence answers to hedge/abstain; and (iv) a **fail-closed governance
contract** enforcing that nothing fluent leaves the system without an attached,
machine-readable attribution trail. We evaluate under a **pre-registered no-overclaim
gate**: a number is reported as *validated* only with ≥2 independent judge families in
consensus, reported inter-judge agreement (Cohen's κ), ≥3 runs, and confidence
intervals. On a local 8B model, Sophia reduces hallucinated attributions from
**36.1% → 23.6%** (Δ **12.5%**, 95% CI [5.6%, 19.4%]) at **0% false-positive cost**; on
genuine "I don't know" traps a deterministic calibration scorer measures **0%
fabrication** for the full Sophia pipeline versus 16.7–25% for the raw model, corroborated
by two independent judge families (κ = 0.74). We characterize the **coverage-vs-fabrication
tradeoff** honestly: strict grounding buys trap-safety at a recall cost on thin-source
corpora, which a typed-gate + graph-neighborhood hybrid partially recovers.

---

## 1. The problem

LLMs are fluent before they are faithful. The failure mode we target is not generic
"hallucination" but **attribution hallucination**, which is especially damaging because
the error becomes a *premise* for downstream reasoning:

- Crediting *Confucius* with text from the *Dao De Jing*, or merging Confucianism and
  Daoism into one undifferentiated tradition.
- Attributing a mid-20th-century idea to an earlier figure (anachronism).
- Treating legendary or composite figures as literal, single authors.
- In law: citing authorities that do not exist (the *Mata v. Avianca* failure) or
  misstating what a real authority held (the *Ayinde* misstated-authority failure).

These are *verifiable* errors — there exists ground truth a machine can check against
sources. That is precisely what a fluency-optimized decoder does not do by default.

## 2. Design principles

1. **Everything is a claim with sources.** No paragraph leaves the system without a
   machine-readable attribution trail.
2. **Verification is first-class.** Detectors, gates, and human-reviewable ledgers
   decide what may be emitted; the model is not trusted to self-police.
3. **Fail-closed, not fail-open.** On low confidence, source conflict, or failed
   verification, Sophia abstains ("I don't know") or escalates — it never fabricates to
   fill a gap.
4. **Functional self-modeling, not consciousness.** Sophia represents its own
   knowledge, uncertainty, and limits (calibration, metacognitive monitoring, knowing
   when to defer). It makes **no** claim to subjective experience.
5. **Local-first and sovereign.** Runs on the user's own infrastructure with
   multi-backend LLM support, supporting airgapped operation.

## 3. Method

### 3.1 The gate

The control loop converts each candidate claim into one of three outcomes:

- **accept** — claim is supported by a verified source; promote it with its attribution.
- **abstain** — no sufficient source; emit a calibrated "I don't know" rather than guess.
- **block** — claim is contradicted by sources or fails a confidentiality/integrity check.

### 3.2 Belief & provenance graph

A queryable claims/justifications graph performs min-over-chain confidence propagation,
maintains a contradiction ledger, and supports **retraction** and **counterfactual
removal** (what does the conclusion become if this source is withdrawn?). This makes
every emitted claim traceable to its supporting sources and reasoning.

### 3.3 Verifiers

Two classes gate claim promotion: **deterministic verifiers** (e.g. a legal-citation
existence check against federated registers — HK e-Legislation/HKLII, UK National
Archives, US CourtListener) and **model-based verifiers** held to the multi-judge gate
for any judgment that is inherently a model call (e.g. "does this holding *support*
this proposition?").

### 3.4 Calibrated abstention

A graded answer/hedge/abstain router downgrades a gate-passing answer to hedge or
abstain when its confidence is low (downgrade-only, fail-closed). Confidence is pooled
from the routed source's author-confidence and neighbor corroboration. We report
honestly that this provenance-derived confidence is a **sound prior on source quality
but a weak predictor of answer correctness** (balanced accuracy ≈ 0.52–0.58 on a live
trap set) — it should not be over-trusted as a correctness signal.

### 3.5 Governance contract

A fail-closed gateway enforces authorization → dataflow firewall → kill-switch →
classification (Bell-LaPadula-style) → taint-labeling on side-effecting/external tools,
with optional re-verification of served output before it reaches the caller.

### 3.6 Independent-verification toolkit (layered, independence-labelled)

The gate's verification step is a **layer of independent oracles**, each routed to the
fabrication mode it covers and each verdict tagged with its **independence tier**:

- **viral/general claims** → professional fact-check verdicts (Google Fact Check ClaimReviews) — *high*;
- **authorship attribution** → Wikidata creator/author/discoverer records — *high*;
- **fabricated citations** → study/DOI existence in Crossref (the *Mata v. Avianca* mode) — *high*;
- **wrong-creator attribution swaps** → the Wikidata record vs the credited person — *high*;
- **misstated findings of a real source** → a multi-judge entailment panel over an *independent*
  retrieved source, acting only on strict-majority consensus — *medium* (model-based, flagged);
- everything else → **fail-closed abstention**: the system never vouches for what no independent
  oracle can confirm.

Two design commitments make this trustworthy rather than merely plausible. First, **fail-open on
ignorance**: a claim no oracle covers is *not* flagged — the toolkit never fabricates a
contradiction. Second, **honest coverage accounting**: each oracle's reach is bounded and
reported (e.g. professional fact-checkers review only a minority of everyday myths), so a *clean*
verdict means "no independent record contradicts this," not "true." The result is a precision/
recall trade governed by *reference quality*, not a silver bullet — there is no free lunch in
open-world verification.

## 4. Measurement discipline: the no-overclaim gate

Every public number must clear a **pre-registered** bar:

> ≥2 independent judge families in consensus (judge ≠ subject) · reported inter-judge
> agreement (Cohen's κ ≥ 0.40) · ≥3 runs · confidence intervals.

Anything below the bar is labeled *illustrative* or *candidate* and may never be quoted
as a headline. Hidden-eval prompts are never published; only aggregates are. An
independent judge audit is part of the record: on one single-judge result a Claude
panel found the original judge **over-counted** (76% agreement, 10 false positives),
so the validated rate was revised down — and the illustrative row marked accordingly.
We treat this kind of self-correction as a feature, not an embarrassment.

## 5. Results (selected)

**Provenance delta (validated).** Subject `dolphin-llama3:8b` (local), judges =
consensus of DeepSeek-Chat + Llama-3.3-70B (2 families), 3 runs:

| Hallucination (alone) | Hallucination (gated) | Δ (95% CI) | FP cost | Coverage |
|---|---|---|---|---|
| 36.1% | 23.6% | **12.5% [5.6%, 19.4%]** | **0.0%** | 34.6% |

**Calibration / abstention (validated, deterministic scorer).** Full Sophia vs raw
DeepSeek-Chat on unknown-author/quote traps: **0% fabrication** for Sophia in all 3
runs vs 16.7–25% raw; calibration Δ **22.0% [14.5%, 29.6%]**. Corroborated by two
independent judge families (GPT-4o + Claude-Sonnet), inter-judge κ = 0.74.

**Verifier accuracy (deterministic).** `legal_citation_exists` catches fabricated
HK/UK/US citations (incl. the real *Mata* fabrication *Varghese v. China Southern
Airlines*) at 100% on a small constructed set (N=14) — validating the extraction +
fail-closed gate logic, not a headline capability claim.

**The honest tradeoff (candidate).** On a recall-heavy, thin-source wiki corpus, strict
grounding scores **1.0 vs 0.0** on attribution-traps/retractions but collapses to
**0.50 vs 0.93** on plain recall, so the raw model wins *overall* — grounding buys
trap-safety at a recall cost. A typed-gate + graph-neighborhood hybrid recovers recall
to ≈0.68 while keeping all traps on the hard-abstain path.

**Independent-verification toolkit (live, candidate).** Across the toolkit's layers, the
high-independence verifiers hold **0% clean over-block** with no model judgment: the
attribution-swap verifier (Wikidata) was run with separated answer/judge models over **3 runs**
with a bootstrap CI (caught **10.8% [9.3%, 11.6%]** of a mixed pack — ≈⅔ of the swap cases it
targets — at **0.0% [0,0]** over-block); the citation-existence verifier (Crossref) and the
multi-judge source-faithfulness verifier behave the same way on their target modes. The honest
finding is a *law*, not a defect: catching open-world contamination requires **either** an oracle
that covers the claim (sparse), **or** fail-closed strictness (which over-blocks), **or** model
knowledge (low independence) — so trustworthiness comes from *composing* labelled-independence
layers and abstaining, not from any single catcher. In the course of this work one earlier
headline (a 70.6% over-block figure) was found to be a stale-report artifact and **withdrawn**
in favour of the corrected 5.9%; we record the correction as part of the method.

See [RESULTS.md](../RESULTS.md) for the complete, caveated tables and reproduction
commands, and [agi-proof/TRUSTWORTHINESS-CAPSTONE-2026-06-28.md](../agi-proof/TRUSTWORTHINESS-CAPSTONE-2026-06-28.md)
for the toolkit's full per-layer coverage/independence accounting.

## 6. Related work

*(Expanded from a structured prior-art survey — [docs/prior-art-survey.md](../docs/prior-art-survey.md).)*
Sophia sits at the intersection of several active lines of research, and we make **no
claim that the general concept of grounded abstention is novel**:

- **Selective prediction / abstention** — *Know Your Limits* (TACL 2024) surveys the
  field; R-Tuning (NAACL 2024) teaches refusal of out-of-knowledge questions; Conformal
  Abstention (Yadkori et al., 2024) abstains with finite-sample hallucination-rate
  guarantees.
- **Attribution & citation verification** — RARR and ALCE (Gao et al., 2023), Self-RAG
  (Asai et al., ICLR 2024), and AGREE (Google, NAACL 2024). Misattribution itself is
  benchmarked (CiteME, CiteAudit, MisCiteBench, AttributionBench).
- **Hallucination detection & fact-checking** — SelfCheckGPT (2023), FacTool (2023),
  FActScore (2023), RefChecker (Amazon, 2024), and ClaimReview-style pipelines.
- **Verifier-gated / guardrail systems** — NeMo Guardrails (NVIDIA, EMNLP 2023), whose
  output rails fact-check and block responses via a separate model; Guardrails AI; Llama
  Guard. Commercially: Vectara, Cleanlab TLM, Patronus/Lynx, Galileo.

We further note **granted US patents** in this space (US 12,468,899, priority 2023-05-08;
US 12,505,311) covering hallucination prevention/handling by checking output against
sources and gating it. Sophia therefore claims **no fundamental algorithmic novelty** and
no freedom-to-operate over the general method.

What is distinctive is the **specific combination and application domain**: the
attribution-hallucination framing in the **humanities** (philosophy/psychology/history/
religion) plus law — an area under-served by the STEM-centric miscitation benchmarks; a
**bilingual** evaluation corpus; the **pre-registered no-overclaim measurement protocol**
with an independent judge audit and a public failure ledger; and the **fail-closed
governance contract** binding it together. We position these as engineering, dataset, and
methodological contributions, not as a novelty claim over the pipeline.

## 7. Limitations

Benchmarks are small and several are self-authored (internally valid; third-party
replication and human semantic review remain). Provenance-derived confidence predicts
source quality, not answer correctness. Strict grounding trades recall for trap-safety.
No claim of AGI, sentience, or consciousness is made.

## 8. Reproducibility

```bash
python scripts/demo_gate.py                                   # offline gate demo, no keys
python tools/run_provenance_delta.py --models mock            # offline plumbing
python tools/run_provenance_delta.py --models <subject> \
    --judges <judgeA>,<judgeB> --runs 3                       # validated-grade run
python3 tools/verify_replication_manifest.py                  # check the verification toolkit manifest (no keys)
```

A self-contained third-party replication pack for the verification toolkit lives at
[agi-proof/verification-replication/](../agi-proof/verification-replication/) — a keyless +
live two-tier runbook (`REPRODUCE.md`), machine-readable `EXPECTED-RESULTS.json`, a
decontamination checklist, and a manifest checker. An independent run by a party other than the
author, on an independently-authored pack, remains the outstanding step.

## How to cite

See [CITATION.cff](../CITATION.cff). Once archived on Zenodo, cite the minted DOI.

---

*This whitepaper is released as a defensive publication under Apache-2.0. It is intended
to establish prior art and a citable priority date for the method described.*
