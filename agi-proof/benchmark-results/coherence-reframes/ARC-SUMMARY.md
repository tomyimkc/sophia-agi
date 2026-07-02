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

## D3 addendum (hedge-matched claims)
Matching claim epistemics to sophia's hedged evidence ('X is the traditionally attributed author of Y') RAISED NLI's answerable-coverage from D2's ~0.06 to 0.43 — the framing fix helped — but NLI still over-abstains vs the healthy semantic incumbent (0.43 vs 0.93; fails the answerable-coverage guard) because sophia's TERSE 3-sentence wiki prose often doesn't clearly state the attribution. NO-GO, robust across blunt (D2) and hedge-matched (D3) claims. This confirms retrieval-bound from the other side: entailment needs RICH evidence; sophia's terse prose is insufficient. Prereq: richer/longer sophia-domain evidence + more labeled data.

## Capstone — evidence-corpus acceptance gate (line CLOSED)
The redirect-and-fund instrument ([`EVIDENCE-CORPUS-GATE.md`](EVIDENCE-CORPUS-GATE.md)) removed D3's two escape hatches at once: an **independent, rich, multi-sentence** Wikipedia corpus (99 works, 198 passages, curator ≠ gate-runner) instead of sophia's terse prose, and **blunt claims matched to that blunt evidence**. n=134, MDE 0.124. Result: **NO-GO, overdetermined.** The **deployable retrieval arm shows no win at all** (ΔF1 −0.012 semantic / −0.058 production-hash, point estimates ≤0) — default-on fails before any guard. Even the **curated ceiling** is disqualified: a real F1 lift (+0.233, CI[0.110,0.344], 3 seeds) bought **entirely by over-abstention** (admits 12% of true attributions vs the incumbent's 58%). And it is **not a junk-evidence artifact** — the manipulation check passes on every arm (0.85–0.91 fact-bearing) and the incumbent is healthy throughout; NLI over-abstains on rich fact-bearing evidence anyway. **Terse prose was never the cause.** The FEVER positive was retrieval-bound (it needed pre-selected single-sentence fact-bearing evidence). **The default-on grounding line is closed.** The mechanism is retained only where it never triggered a coverage penalty: as the **contradiction-only** hybrid. The NLI primitive stays on `main` behind the `EntailmentFn` seam, injectable, fail-closed, **off by default**.
