# Submission cover note & metadata — whitepaper / dataset

> Reusable text for posting the Sophia whitepaper as a **defensive publication** to
> Zenodo, OSF, or arXiv, and for the Hugging Face dataset card. The goal is a
> **timestamped, citable prior-art record** under your name. Author of record:
> **Yim Kin Cheong**, Hong Kong.

## Where to post (in order of ease)

| Venue | Endorsement needed? | What you get | Notes |
|-------|--------------------|--------------|-------|
| **Zenodo** | No | DOI + timestamp; archives a GitHub release automatically | Easiest. Also the DOI source for `CITATION.cff`. |
| **OSF Preprints** | No | DOI + timestamp | Good second mirror. |
| **arXiv** (cs.CL / cs.AI) | Yes (first submission needs endorsement) | Strong academic visibility + timestamp | Convert the markdown whitepaper to LaTeX/PDF first. |

Posting to **any** of these before further disclosure cements your priority and makes it
hard for anyone to later patent the method over you.

## Title
Sophia — the Wisdom Gate: A Provenance-Aware, Verifier-Gated Reasoning Layer That Abstains Instead of Fabricating

## Authors
Yim Kin Cheong (tomyimkc), Independent researcher, Hong Kong. *(Add ORCID once registered.)*

## arXiv primary/secondary categories
Primary: **cs.CL** (Computation and Language). Secondary: **cs.AI**, **cs.LG**.

## Abstract (≤ ~1920 chars, arXiv-ready)
Large language models routinely produce attribution hallucinations: they credit the wrong
author for an idea, merge distinct intellectual traditions, treat legendary figures as
literal authors, or cite authorities that do not exist — and then reason on top of those
errors. We present Sophia (the Wisdom Gate), a provenance-aware, verifier-gated reasoning
layer that abstains instead of fabricating, with the control loop: claim, verify against
sources, then accept, abstain, or block. Sophia couples a queryable belief/provenance
graph with confidence propagation, retraction, and counterfactual analysis; deterministic
and model-based verifiers that gate claim promotion; calibrated abstention that downgrades
low-confidence answers to hedge or abstain; and a fail-closed governance contract.
We evaluate under a pre-registered no-overclaim protocol: a result is reported as
validated only with at least two independent judge families in consensus, reported
inter-judge agreement (Cohen's kappa >= 0.40), at least three runs, and confidence
intervals. On a local 8B model, Sophia reduces hallucinated attributions from 36.1% to
23.6% (delta 12.5%, 95% CI [5.6%, 19.4%]) at zero false-positive cost; on genuine
"I don't know" traps a deterministic scorer measures 0% fabrication for the full pipeline
versus 16.7-25% for the raw model, corroborated by two independent judge families
(kappa = 0.74). We characterize the coverage-versus-fabrication tradeoff honestly: strict
grounding buys trap-safety at a recall cost on thin-source corpora, partially recovered by
a typed-gate plus graph-neighborhood hybrid. We make no claim of fundamental algorithmic
novelty over the broad space of selective prediction, attribution verification, and
guardrail systems; the contribution is a curated, rigorously measured, bilingual-humanities
application with an auditable governance contract and a public failure ledger.

## Keywords
large language models; hallucination; attribution; abstention; selective prediction;
provenance; retrieval-augmented generation; verifier-gated reasoning; trustworthy AI

## Comments field (arXiv)
Defensive publication. Code, data, and reproduction scripts:
https://github.com/tomyimkc/sophia-agi · Dataset:
https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus · Apache-2.0.

## Plain-language summary (Zenodo/OSF description)
This work releases the method and measured results for Sophia, a system that makes a
language model say "I don't know" instead of inventing a source. It checks each claim
against real documents before letting it through, and reports only numbers that pass a
strict, pre-registered honesty bar. It is published openly to establish a public,
timestamped record of the method under the author's name.

## Suggested licence on the preprint
**CC BY 4.0** for the paper text (standard for preprints; requires attribution), while the
code/data remain **Apache-2.0**. This keeps the work open while ensuring you are credited.

## Conversion note for arXiv (PDF)
arXiv wants LaTeX or PDF, not Markdown. Quickest path:
`pandoc paper/sophia-whitepaper.md -o sophia-whitepaper.pdf` (needs a TeX engine), then
submit the PDF. Zenodo/OSF accept the Markdown or PDF directly.
