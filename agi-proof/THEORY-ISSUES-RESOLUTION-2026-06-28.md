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
- **The strong original result did NOT survive rigor.** Live:
  | Setup | Contamination caught | Clean over-blocked |
  |---|---|---|
  | answer == judge (original) | 97.7% | **5.9%** |
  | **answer ≠ judge** (Claude answers, DeepSeek judges), curated refs | 97.7% | **70.6%** |
  | **open-world Wikipedia** refs (answer ≠ judge) | 100% | **64.7%** |
  The low 5.9% over-block was an **artifact of the same model answering *and* judging**. Under
  a genuinely independent judge — or real retrieved refs — the verifier rejects ~65–70% of
  *clean* answers. Contamination-catching stays high; clean-answer preservation collapses.
  Report: `agi-proof/source-verifier/live-hardened-2026-06-28.json`.

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
