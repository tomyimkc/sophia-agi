# Training considerations — flagged factors to notice before/while training a model

Running list of methodology factors that materially affect training-data quality or the trained
model's behaviour. Each is a thing to **verify or decide before training**, not a result.
`canClaimAGI` stays false.

## TC-1 — Parallel-tool-use mining uses a HEURISTIC dependency check (verify a sample)
`okf_parallelizable_miner.py` decides two consecutive tool calls are independent (→ safe to batch
into one parallel turn) by **token overlap**: call B is treated as *dependent* on call A only if B's
input reuses a ≥5-char token that first appeared in A's *result*. This is good for mining but
**approximate**:
- **False "independent"** (the dangerous error): B truly needed A's output but reused no surface
  token (e.g. A returned a count, B used it semantically) → the miner emits a *parallel* example that
  would teach the model to issue B without A's result. A model trained on it may parallelize
  genuinely-dependent calls and act on stale/absent data.
- **False "dependent"**: coincidental token overlap (a common path/word) → a real parallelizable
  batch is conservatively split. Harmless (just fewer positive examples).
**Action before training on miner output:** spot-check a sample of the emitted `parallelizable_reads`
rows by hand; gate every row through `agent/swarm_trust_boundary.py`; consider restricting positive
examples to **read-only** tools (no writes) so a wrong batch can't cause a mutating side effect. The
asymmetry matters — bias the miner toward *false-dependent* (split when unsure), never toward
*false-independent*.

## TC-2 — Capture everything; curate offline (don't let the model self-select)
The OKF hook captures every step deterministically on purpose. Filtering "valuable" steps at capture
time bakes the model's blind spots into the corpus. Decide what is training-worthy in curation
(`okf_to_training.py` + the trust boundary), not in the prompt/skill.

## TC-3 — The NVFP4 cert measured a half-merged adapter (fix the instrument before reading the metric)
The committed v5 top1 (0.8828) was measured with 32/96 LoRA modules (the fused-MoE experts) dropped
at merge time (ledger `nvfp4-v5-cert-recovered-contaminated-2026-06-30`). Any recipe comparison
(v5→v6) is only valid once merge coverage is 96/96 (fix the Spark peft version or train with
per-expert `nn.Linear` experts). Don't tune the recipe against a contaminated measurement.

## TC-4 — Optimize the objective the gate scores (see QAT-v6-Recipe-Proposal)
v5 optimized a weight-space penalty while the cert scores output-space agreement → top1 flips. v6
trains the cert's own metrics (KD + top1-margin). General lesson: train the loss your acceptance gate
actually measures, not a proxy.
