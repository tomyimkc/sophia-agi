# arXiv submission metadata — Sophia (the Wisdom Gate), v0.11.0

Front-matter and form fields for an arXiv submission of `paper/arxiv/sophia-arxiv.tex`.
The Markdown whitepaper (`paper/sophia-whitepaper.md`) remains canonical; the `.tex` is a
faithful, compilable rendering. **Honest framing: AGI-candidate, not proven AGI.**

## Build
```bash
cd paper/arxiv
pdflatex sophia-arxiv.tex && pdflatex sophia-arxiv.tex   # twice for refs
```
Self-contained (standard TeX Live: geometry, booktabs, amssymb, hyperref). For arXiv, upload
`sophia-arxiv.tex` (and the generated `.bbl` if you add a bibliography).

## Form fields

- **Title:** Sophia — the Wisdom Gate: A Provenance-Aware, Verifier-Gated Reasoning Layer That Abstains Instead of Fabricating
- **Authors:** Yim Kin Cheong (ORCID 0009-0005-9520-0033), Independent researcher, Hong Kong
- **Primary category:** `cs.CL` (Computation and Language)
- **Cross-list:** `cs.AI` (Artificial Intelligence); optionally `cs.IR` (Information Retrieval)
- **ACM class:** I.2.7 (Natural Language Processing); I.2.4 (Knowledge Representation)
- **Comments:** "v0.11.0. Defensive publication; source + reproducible benchmarks and a third-party replication pack at https://github.com/tomyimkc/sophia-agi . AGI-candidate system; not a claim of AGI."
- **License:** recommend `CC BY 4.0` (consistent with the repo's Apache-2.0 openness) or the arXiv non-exclusive license. Do NOT select a no-derivatives license if you want maximum prior-art reach.
- **Report-no / DOI:** add the Zenodo version DOI once minted (also lets arXiv ↔ Zenodo cross-link).

## Abstract (plain text — paste into the arXiv abstract box; no LaTeX/markup)
Modern large language models confidently produce attribution hallucinations: they credit the wrong author for an idea, merge distinct intellectual traditions, treat legendary figures as literal authors, or cite authorities that do not exist, and then reason on top of those errors. We present Sophia (the Wisdom Gate), a provenance-aware, verifier-gated reasoning layer that abstains instead of fabricating. The control loop is: claim, verify against sources, then accept, abstain, or block. Sophia couples a queryable belief/provenance graph with confidence propagation, retraction, and counterfactual analysis; deterministic and model-based verifiers that gate claim promotion; calibrated abstention; and a fail-closed governance contract. We evaluate under a pre-registered no-overclaim gate: a number is validated only with at least two independent judge families in consensus, reported inter-judge agreement, at least three runs, and confidence intervals. On a local 8B model, Sophia reduces hallucinated attributions from 36.1% to 23.6% (delta 12.5%, 95% CI [5.6%, 19.4%]) at 0% false-positive cost; on "I don't know" traps a deterministic scorer measures 0% fabrication for the full pipeline versus 16.7-25% for the raw model (kappa 0.74). This release adds a layered, independence-labelled verification toolkit (viral-claim fact-check, authorship and wrong-creator attribution checks, fabricated-citation existence, and a multi-judge source-faithfulness check) and characterizes the limits of open-world verification: a precision/recall "no free lunch" law, plus the discovery and correction of one of the system's own over-claims. We make no claim of general intelligence.

## Honesty checklist before submitting
- [ ] No sentence implies proven AGI; "candidate" framing intact.
- [ ] Every reported number carries its evidence tier (validated/candidate) and CI where applicable.
- [ ] The withdrawn 70.6% figure is presented as a correction, not omitted.
- [ ] Limitations + "no third-party replication yet" stated plainly.
- [ ] License chosen allows prior-art reach (CC BY recommended).
