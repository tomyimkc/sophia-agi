# Resolution of the Theory Issues — Live Outcomes (2026-06-28)

Companion to `THEORY-ISSUES-FROM-LIVE-RUNS-2026-06-28.md`. All four recommendations
were implemented (deterministic-tested) and then **run live** (OpenRouter + LLMHub
models; real Lean kernel on RunPod). The live runs resolved some issues and exposed a
deeper, unifying one. `canClaimAGI` stays false.

## Recommendation outcomes

### Rec 1 — LLM/NLI debunk-detector (`agent/llm_debunk_detector.py`)
- **Detection: SOLVED.** Live, the LLM detector classifies **21/21 (100%)** strong-model
  answers as `debunk`, vs the keyword `classify_response`'s **0/21**. Swapping the keyword
  heuristic for an LLM/NLI judge takes debunk detection from 0% → 100% on real data — the
  validated "strong models debunk" claim is now *detectable*.
- **But verification is the NEW bottleneck.** `debunk_gate` then confirmed **0/21**: the
  preserve-and-verify step (`agent.fact_check_text`) decomposes the verbose refutation into
  many atomic claims and fail-closes unless *every* claim is entailed by ≥2 of the *narrow*
  curated truth-refs — which a rich debunk never satisfies. The binding constraint moved
  from detection to verification. Report: `agi-proof/debunk-gate/live-llm-detector-2026-06-28.json`.

### Rec 2 — Contamination bench hardening (multi-run CI, answer≠judge, open-world retrieval)
> **CORRECTION (later clean re-run).** An earlier draft of this section reported answer≠judge
> on *curated* refs over-blocking **70.6%**. That number was a **stale retrieve report**
> mis-saved when a prior run hung during a git-stash race — it is **withdrawn**. A clean
> re-run (captured directly) gives the matrix below; the true over-block driver is **open-world
> retrieval**, not answer≠judge separation.

- **Clean live matrix (answer = Claude, judge = DeepSeek, atomic verifier):**
  | refs | Contamination caught | Clean over-blocked |
  |---|---|---|
  | curated | 97.7% | **5.9%** |
  | **open-world Wikipedia** | 97.7% | **52.9%** (earlier retrieve runs 64.7–70.6%) |
  So answer≠judge on curated refs keeps over-block low (~5.9%); it is **generic open-world
  retrieval** that drives over-block to ~50–70%, because Wikipedia summaries don't entail the
  specific clean answer. Reports: `agi-proof/source-verifier/live-hardened-2026-06-28.json`
  (note: its 70.6%/64.7% figures are the retrieve-mode runs, not curated), and the corrected
  matrix in `agi-proof/source-verifier/verifier-compare-2026-06-28.json`.

### Rec 3 — Held-out theorems + semantic novelty (`agent/proof_novelty.py`)
- **Recall vs discovery, confirmed.** On held-out bespoke theorems the LLM-proposer +
  real Lean kernel verified **3/20** (Claude 3/10, DeepSeek 0/10) — vs **8/10** on known
  library lemmas. The non-trivial polynomial identities all failed (models reached for
  Mathlib's `ring`, absent in core). The semantic novelty assessor flags every known-lemma
  proof `likely_recall=True / surface_novel=False` and the held-out accepts `surface_novel=True`
  (hf09/hf10 also `likely_recall=False`) — it reduces the memorization false-positive the plain
  Jaccard could not catch. Report: `agi-proof/proof-search/lean-held-out-fresh-2026-06-28.json`.

### Rec 4 — Adversarial fakes (methodology guard)
- **Done and CI-pinned.** The `--fake` debunk answer now uses realistic keyword-free debunk
  phrasing; a guard test asserts the keyword detector mislabels it while the LLM detector
  catches it. A green `--fake` can no longer mask a broken detector.

## The unifying finding: the verification layer is the binding constraint

Clusters A and C failed live for the **same** root reason. `agent.fact_check_text` verifies
by decomposing an answer into atomic claims and fail-closing unless each is entailed by ≥2
independent narrow references. That discipline — validated on short, curated cases — is **too
strict for real, verbose model output**:
- it **cannot confirm a verbose debunk** (Cluster A: 0/21 verified despite 100% detected), and
- it **rejects most clean answers** under an independent judge (Cluster C: 70.6% over-block).

So the project's strongest-looking deterministic results (the provenance/contamination gate)
rest on a verification step that, made rigorous (independent judge, open-world refs, real
verbose answers), trades almost all recall for safety. **This is the central theory issue the
live runs surface, and it was previously hidden by (a) curated narrow refs and (b) self-judging.**

## Recommended next work (a 5th cycle)

1. **Core-claim verification, not all-claims.** Verify the *load-bearing* claim of an answer
   (the attribution, or the explicit refutation) against independent refs, instead of requiring
   *every* atomic side-claim to be entailed. Expectation: Cluster A verified-debunk rate rises
   from 0 and Cluster C clean over-block falls from 70.6% — without losing contamination-catching.
2. **Calibrate the entailment judge** (it currently defaults `irrelevant`/reject on any
   uncertainty); measure judge strictness vs a small human-gold set.
3. **Always run answer ≠ judge + ≥3 runs + open-world refs** as the default rigorous protocol;
   the self-judged / curated numbers are upper bounds, not headlines.
4. **Lean:** only claim `verifier_synthesis_over_proof_kernel` progress on the held-out set with
   the semantic novelty guard; the known-lemma 8/10 is recall.

## 5th cycle (done) — core-claim verification via Google Fact Check

Implemented recommendation #1 (`agent/core_claim_verifier.py`): verify the *single* injected
falsehood against an **independent oracle** — Google Fact Check Tools (professional
ClaimReviews) — instead of decomposing the verbose answer into atomic claims. Layered with a
flagged low-independence LLM-knowledge fallback for the long tail.

**Live result:** verified-debunk rises from **0/21 → 18/21**.
- **4 high-independence** via Google Fact Check (the viral myths it reviews: Great Wall from
  space, Edison, 10%-brain, blue blood). Google coverage is **4/21** — matching the ledger's
  long-standing note that it covers general/viral claims, not provenance.
- **14 low-independence** via the model's own knowledge (transparently flagged).
- **3 fail-closed (correctly):** 2 genuinely-unknown provenance claims (Voynich authorship —
  not a fact-checkable falsehood) and 1 LLM-knowledge *error* (it rated the Mozart/Twinkle
  myth "true"), which the fail-closed design correctly refuses to count.

**Takeaway:** core-claim verification fixes the verbose-debunk gap the all-atomic-claims path
could not. Google is the right *independent* oracle but is too sparse (4/21) to stand alone;
a layered verifier (Google for viral claims · Wikidata/Crossref for provenance · a flagged
model-knowledge tail) is required, and the independence of each verdict must be reported, not
hidden. Report: `agi-proof/debunk-gate/core-claim-verification-2026-06-28.json`.

### Layered verifier (`agent/layered_verifier.py`)
Generalizes the above to route each claim to the most independent oracle that covers it:
Google (viral) → Wikidata/Crossref provenance (authorship) → flagged LLM tail. Live, the
provenance layer verified **4/4** authorship misattributions (Dickens/Shakespeare → `false`,
Tolstoy/Twain → `true`) but fired on **0** of the debunk pack's *fabricated-study* cases (they
aren't simple "X wrote Y" claims), so high-independence coverage on that pack stayed at 4
(Google). Each layer works on its claim type; high-independence coverage is bounded by what the
open oracles actually review. Report: `agi-proof/debunk-gate/layered-verification-2026-06-28.json`.

### Cluster C over-block — core-claim verifier, and a fail-open/fail-closed tradeoff
Applying core-claim verification to the contamination gate (`agent/core_claim_source_verifier.py`,
reject only when the answer's *core* claim is contradicted) gives this clean live matrix
(answer = Claude, judge = DeepSeek):
| refs | verifier | caught | over-block |
|---|---|---|---|
| curated | atomic | 97.7% | 5.9% |
| curated | **core** | 95.3% | **0.0%** |
| open-world | atomic | 97.7% | 52.9% |
| open-world | **core** | **58.1%** | **0.0%** |

Core-claim verification drives over-block to **0%** in both regimes — but it is **fail-open**:
on curated refs it keeps catch high (95.3%, −2.4pt), yet on weak open-world refs catch
**collapses to 58.1%** because Wikipedia rarely *explicitly contradicts* the injected
fabrication. Atomic is fail-closed (high catch, high open-world over-block); core is fail-open
(zero over-block, low open-world catch). **Neither dominates** — the real lever is *reference
quality*: authoritative oracles (Google/Wikidata, via the layered verifier) that genuinely
contradict contamination, not generic retrieval. Report:
`agi-proof/source-verifier/verifier-compare-2026-06-28.json`.
