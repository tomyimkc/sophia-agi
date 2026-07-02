# Coherence → entailment → retrieval arc — complete honest summary (2026-07-02)

A single question, chased down a promotion ladder: **is there a verification/abstention signal that
beats sophia's existing calibration baselines?** Every rung pre-registered; every result adversarially
verified; `candidateOnly` / `canClaimAGI=false` throughout. The ledger rows, not this prose, are the record.

## The ladder
1. **Coherence family (O1–O5): all NEGATIVE.** Kuramoto/oscillatory readouts don't beat baselines. Root
   cause: *coherence measures confidence, not truth — a fluent hallucination is maximally coherent.*
2. **Reframes (R1–R5): 4 negative + 1 refuted.** R1's n=160 blip evaporated at n=464; R2's factcheck
   "win" was refuted on balanced MMLU (decoded verdict-structure, not correctness).
3. **Entailment (NLI): VALIDATED vs coherence.** FEVER n=400: NLI AUROC 0.962 vs coherence 0.650
   (Δ+0.31, CI excludes 0). Entailment tells *supporting* from *refuting* evidence; coherence can't.
4. **Deployment + production gate: NO-GO.** As a drop-in `fact_check_gate` backend, NLI does NOT beat
   the incumbent lexical screen (ΔF1 −0.09) and over-abstains — the abstention guard caught it.
5. **A (retrieval test): GO on FEVER, underpowered in-domain.** Sentence-level evidence flips NLI from
   −0.09 (metadata) to +0.14 (FEVER n=400, CI excludes 0). *Grounding is retrieval-bound* — the mechanism
   works given fact-bearing evidence. Qualifier: FEVER only; C1-Wikipedia arm n=18 directional but CI∋0.
6. **D + D2 (sophia domain): NOT GO — blocked, not confirmed.**
   - The non-forge evidence-groundable pack caps at **n=106** (30 supported / 76 refuted) → MDE ~0.21 ≫
     +0.05: **underpowered** (declared pre-run). Labeled evidence-groundable data is scarce in-domain.
   - The lexical incumbent **collapses** on sophia's hedged provenance evidence (coverage 0.009).
   - Fixing the incumbent (healthy semantic-similarity, coverage 0.78): NLI's F1 edge is real-but-tiny
     (+0.05 real / +0.07 forge) **but bought by catastrophic over-abstention** (NLI coverage 0.057 vs 0.78;
     drop 0.73 ≫ 0.01 — fails the abstention guard).
   - **Root cause: a claim–evidence hedge mismatch.** sophia's evidence is epistemically hedged
     ("attributed to / compiled / legendary"), so it doesn't strictly entail blunt "X wrote Y" claims;
     NLI correctly abstains — which is *consistent with sophia's own uncertainty discipline.*

## What is true, stated with its qualifier
- **Coherence ≠ truth.** ~15 instruments confirm it. First-class negative evidence for the abstention thesis.
- **Entailment > coherence, on fact-bearing evidence.** Validated on FEVER; the retrieval-bound diagnosis is
  **GO-confirmed on FEVER through the real gate only.** The sophia-domain arm is **blocked** (data scarcity +
  incumbent collapse + claim-evidence hedge mismatch), **not confirmed.**
- **The lexical token-overlap incumbent is inadequate** for realistic (paraphrastic/hedged) evidence — it
  collapses, making NLI-vs-lexical unfalsifiable. The fair baseline is semantic, not lexical.

## What shipped (net-positive, GO-independent)
`agent/nli_grounding.py` (NLI + contradiction-hybrid + healthy semantic-similarity `EntailmentFn` backends,
fail-closed, tested) — on `main` (PR #352), candidateOnly, no default-on. Self-contained gate harnesses +
all reports + this summary. The harness is the deliverable; nothing claims a capability the gate didn't grant.

## Honest open prerequisites (for anyone resuming)
1. **A larger curated evidence-groundable sophia-domain pack** (labeled data is the binding constraint).
2. **Claims that match the evidence's epistemics** (hedged claims for hedged evidence) — or accept that
   blunt-claim verification is the wrong task for a hedged-provenance corpus.
3. **A healthy incumbent** (semantic or hybrid) for any future gate — the lexical screen is not one.
