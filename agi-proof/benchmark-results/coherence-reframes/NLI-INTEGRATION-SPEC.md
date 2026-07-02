# Deploying the NLI-entailment primitive into sophia's grounding gate — spec

Status: **primitive + adapter + tests landed; production default-on NOT enabled** (gated on the
acceptance test below, on sophia's real retrieval pipeline). candidateOnly. canClaimAGI=false.

## Why

Validated result (`NLI-ENTAILMENT.public-report.json`): textual entailment decisively beats the
coherence/lexical baseline at telling **supporting** from **refuting** evidence — FEVER n=400 AUROC
**0.962 vs 0.650** (paired Δ+0.31, CI [0.26,0.37] excludes 0, 3 seeds; adversarially verified sound).
Root cause it fixes: similarity/coherence treats refuting evidence as "topical"; NLI does not.

## What landed (this PR)

- `agent/nli_grounding.py` — `build_nli_entailment(scorer=None, model_name=..., thresholds...)`
  returns a drop-in `agent.fact_check_gate.EntailmentFn`:
  `(AtomicClaim, EvidenceSource) -> "entails" | "contradicts" | "irrelevant"`.
  Premise = evidence (title + snippet), hypothesis = claim.text; label = thresholded argmax over
  [contradiction, entailment, neutral].
- `tests/test_nli_grounding.py` — 6 deterministic, download-free tests (injected scorer; no
  fact_check_gate import, so runnable on any Python).
- **Zero default change:** the gate already exposes the `EntailmentFn` seam and falls back to the
  conservative lexical screen when none is injected (`external_ground(..., entailment=None)`).

Integration smoke (py3.12): on a wrong-author source, `entailment=build_nli_entailment(...)` →
**rejected**; `entailment=None` (lexical) → **held**. The NLI backend catches the contradiction
the lexical screen cannot.

## How to turn it on

```python
from agent.nli_grounding import build_nli_entailment
try:
    entail = build_nli_entailment()          # loads cross-encoder/nli-deberta-v3-base
except Exception:
    entail = None                            # fail-closed -> gate keeps the lexical screen
result = external_ground(claim, retriever, entailment=entail)
```
Behind an opt-in flag (e.g. `SOPHIA_ENTAILMENT_BACKEND=nli`). Fail-closed by construction: a model
load failure or empty evidence yields the lexical fallback / "irrelevant", never an over-confident pass.

## Acceptance gate for production default-on (NOT yet met — measure, don't assume)

On sophia's **real retrieval pipeline** (not gold evidence), the NLI-backed `external_ground`
admission must beat the current lexical arm on **F1 at matched coverage** on the C1 fact pack
(`agent/realtime_benchmark.py`), with the deterministic/religion/history PROTECTED suites showing
no regression, reproduced across ≥2 seeds. Requires py3.11+ (the gate's possessive-quantifier regex
at `fact_check_gate.py:149`). Record the run in the failure ledger either way.

## Honest bounds

- The default model is a purpose-built NLI **specialist**; validation used **clean gold evidence**.
  Real retrieval is noisier — end-to-end quality depends on the retriever, not just the NLI head.
- This validates a **mechanism choice** (entailment, not coherence) for the grounding layer; it does
  **not** claim NLI "solves verification." Keep every artifact candidateOnly until the gate above is met.
- Latency/cost: a cross-encoder call per (claim, evidence) pair. Batch, and reserve for the
  high-risk / low-confidence claims the cheaper arms flag — an escalation tier, not a blanket pass.
