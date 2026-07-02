# Thesis note — "coherence measures confidence, not truth" is a first-class negative result

The most valuable output of the O1–O5 + W1/W5 + R1–R5 + NLI arc is **not** any single AUROC. It is
a large, honestly-buried body of negative evidence that *defends sophia's abstention thesis*:

> **Coherence measures confidence, not truth. A fluent hallucination is maximally coherent.**
> Hidden-state / self-consistency / coherence signals decode surface structure well but factual
> support poorly, and none of ~15 instruments beat sophia's existing calibration baselines
> (stated confidence, self-consistency, provenance/lexical gates) on factual correctness.

Why this is evidence, not just a pile of failures:

- It is **measured, adversarially verified, and reproduced** — not asserted. Every negative was
  re-derived independently; the two that looked positive (R1 at n=160, R2 on factcheck) were
  *refuted* by escalation (more data / a balanced multi-domain set), which is the promotion ladder
  working as designed.
- It **rules out a whole family of "cheap internal confidence" shortcuts** that a capability-first
  lab would be tempted to ship as verification. That a signal saturates on the confident-wrong case
  is the precise reason sophia abstains on calibrated uncertainty rather than trusting fluency.
- Even the one signal that beat *coherence* (NLI entailment, FEVER 0.96 vs 0.65) **did not clear the
  production acceptance gate** vs the incumbent lexical screen (ΔF1 −0.10, over-abstains) — a second,
  sharper confirmation that "looks-supported" ≠ "is-supported," and that the honest bar is the
  incumbent, not a strawman.

**Implication for "wisdom before intelligence":** the discipline that produced these negatives —
pre-registered gates, coverage/abstention guards, ≥2 judge families, no post-hoc threshold moves —
is itself the deliverable. The catalogue of what does **not** carry a truth signal is what lets
sophia abstain honestly instead of narrating confidently. Treat these rows as citable evidence in
any thesis writing, alongside (not beneath) the positive results.

Index of the negatives (all `candidateOnly`, `canClaimAGI=false`, in `agi-proof/failure-ledger.md`):
O1–O5 (oscillatory coherence), R1 (internal-vs-stated), R2 (ensemble-disagreement, refuted on MMLU),
R3 (grounding phase-lock), R4 (perturbation stability), R5 (operational gate), and the NLI production
acceptance gate NO-GO. The one validated-vs-coherence primitive (NLI) is retained as an optional,
fail-closed backend — not a default.
