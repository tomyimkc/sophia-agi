# Synthetic-Data Fidelity Scaling (P1 — design)

> Status: design. No new training code in this pass. This doc fixes the
> pipeline, the fidelity gate (the moat), and the exact seams it plugs into.

## The thesis, and why fidelity is the bottleneck

The advisor's point ②: *"Data is the real bottleneck, not parameters. … Low-fidelity
synthetic data collapses performance in your own pretraining studies. Focus here first."*

This is not a hypothesis in this repo — it is a **measured finding**. The
synthetic-data scaling study (`pretraining/synthetic_scaling/run_synthetic.py`)
defines two fidelities by drift (`run_synthetic.py:50`: `high_fidelity=0.15`,
`low_fidelity=0.6`) and shows:

- high-fidelity synthetic **saturates near the floor** as it grows;
- low-fidelity synthetic makes loss **rise** once it dominates the mix → model
  collapse.

The load-bearing lesson is stated at `run_synthetic.py:91-93`:

> *"High-fidelity synthetic data scales and saturates near the floor; low-fidelity
> synthetic data collapses the model once it dominates the mix. Quantity cannot
> substitute for distributional fidelity."*

(Toy study — `run_synthetic.py:85-86` — it demonstrates the lesson, not a recipe.)
Indexed at `pretraining/README.md:68,72` and `README.md:275`.

**So the bottleneck is not *generating* more data; it is *guaranteeing fidelity*
at scale.** The generation step is cheap (a model); the fidelity gate is the moat.
This doc designs that gate and the pipeline that stamps each surviving record with
a provenance passport before it touches training.

## Why this is the highest-leverage lever on a 2-node cluster

On small hardware, raw parameter scale buys little. A synthetic-data flywheel
that *only admits high-fidelity records* raises the effective quality of every
downstream training run — including the RLVR runs that the capability panel (P0)
now measures. It is the cheapest capability gain you can buy, and the fidelity
gate is what stops it from collapsing the model the study warns about.

## The pipeline

```
        generate (model)  ──►  fidelity gate  ──►  passport stamp  ──►  training pack
                               (grounded.py)        (ssil_provenance_     (contamination-
                               fail-closed:         chain.py G9C)         guarded split)
                               abstain if off-KB
```

Three stages, each reusing existing, deterministic, offline code. **No new metrics
are invented** — every stage already exists; this doc composes them.

### Stage 1 — Generate

Synthetic generation is the easy part and is *not* trusted. Any model can emit a
candidate (claimed_author, work) record. The existing packs show the shape:

- `training/self_evolve/distill.jsonl` — committed/verified self-evolve rounds as
  self-distillation SFT (`tools/export_self_evolve_distill.py:4`).
- `training/hk_advisor/` — verified bilingual HK provenance-advisor SFT traces
  (`tools/gen_hk_advisor_traces.py:4`).
- `training/sophia-math-code-curriculum/` — sympy/exec-verified synthetic math+code
  SFT (`training/sophia-math-code-curriculum/README.md:1-13`).
- `training/tool_use/` — MCP/tool-use DPO pairs (grounded vs over_call /
  ignored_error / schema_invalid).
- `training/corrections_pending/`, `training/feedback/` — the active-learning
  return paths (judge gate-misses → candidate `doNotAttributeTo` records).

The flywheel *adds* to these packs; it never edits them in place (the
contamination guard and the promotion gate depend on that).

### Stage 2 — Fidelity gate (the moat): `provenance_bench/grounded.py`

The fidelity question for a provenance record is: *"is this attribution actually
true, actually a known misattribution, or unverified?"* The **grounding-gated
detector** answers exactly this, fail-closed:

- `ground(claimed, work, kb)` (`grounded.py:58-66`) returns `TRUE` if the claimed
  author matches a KB gold author, `MISATTRIBUTION` if it is a known wrong
  attribution, else `ABSTAIN` (the KB is silent).
- `_flags(text, claimed, work, kb)` (`grounded.py:69-72`) flags **only** when the
  text asserts an attribution AND grounding finds `MISATTRIBUTION`. An `ABSTAIN`
  does **not** flag — that is the fail-closed closure of the cross-entity gap.

This is the gate that closes the honest limit proven by
`provenance_bench/cross_entity.py`: memorized provenance rules are precise
(≈0 FP) but do not transfer to unseen entities (`cross_entity.py:14-16`,
`cross_entity.py:124-158`); a structural detector transfers but FP≈1. Grounding
needs no per-entity training, so it transfers across any entity the KB covers
while keeping FP toward zero (`grounded.py:16-23, 115-119`). **That is the fidelity
property the flywheel requires.**

`run_grounded(pairs, true_controls, kb, *, seed=0)` (`grounded.py:75-120`) reports
`groundedRecall_all`, `groundedRecall_covered`, `groundedFalsePositive`,
`kbCoverage`, `abstainRate` — so the gate's own quality is measurable, not
asserted.

### Stage 3 — Passport stamp: `agent/ssil_provenance_chain.py` (gate G9C)

Every record that survives the fidelity gate is stamped into a **cryptographic
provenance chain** (SSIL gate **G9C**, "Merkle lineage"):

- `append_entry(chain, entry, *, hmac_key=None)` (`ssil_provenance_chain.py:94-104`)
  stamps `parentHash` (chain tip) + `entryHash` (SHA-256 / HMAC-SHA256).
- `verify_chain(chain, *, hmac_key=None)` (`:107-123`) recomputes every hash.
- `assert_gated_lineage(chain, leaf_id)` (`:159-168`) requires every ancestor
  **and** the leaf to carry `gateVerdict=="promote"`.
- `evaluate(bundle, *, candidate_id)` (`:189-306`) is fail-closed: `quarantine`
  on missing/unwalkable input, `reject` on tampering or an ungated ancestor.

Honest boundary (`ssil_provenance_chain.py:37-43, 59-64`): this is integrity, not
PKI — it proves lineage (every record descends from gated ancestors), not that
the gating decisions were correct. That correctness comes from stage 2.

### Stage 4 — Contamination-guarded split into the training pack

Before a stamped record reaches training, it goes through the contamination guard
the repo already enforces everywhere:

- `provenance_bench/improvement.py:60-125` — the measured self-improvement loop
  learns provenance **rules** from failures on TRAIN phrasings and scores **only**
  on disjoint HELD-OUT phrasings (`improvement.py:36-37`), with FP cost measured
  on TRUE attributions that must stay ≈0 (`:103-107`). This is the anti-cheating
  template the flywheel's train/eval split must mirror.
- `provenance_bench/heldout_split.py`, `holdout_seal.py` — construct-disjoint,
  sealed splits.

## What the flywheel adds (the work, when we build it)

1. A **generator driver** that fans synthetic (claimed, work) candidates across
   the cluster (embarrassingly parallel — one model, many seeds/domains).
2. A **fidelity-gate batch runner** over `grounded.run_grounded` that emits only
   `groundedFalsePositive≈0` survivors with their KB verdict.
3. A **passport stamper** that appends each survivor to the G9C chain with
   `gateVerdict` set by the fidelity verdict.
4. A **contamination-guarded pack writer** that adds to `training/` packs without
   editing them, mirroring `improvement.py`'s disjoint-phrasing discipline.

Each is a thin composition of existing modules. The design constraint that makes
this safe is unchanged: **quantity never substitutes for fidelity**, so the gate
is on the critical path, not the generator.

## Non-goals (this pass)

- No new training code. No live run. No change to the RLVR reward
  (`provenance_bench/rl_reward.py`) — fidelity is a data property, scored before
  training, never trained into the loss.
- Does not touch the council / test-time compute path (that is P2's comparison,
  not P1's data).

## Verification target (when built)

A flywheel run reports: records generated, records surviving the fidelity gate
(by KB verdict), `groundedFalsePositive` of the survivors, chain-verify result,
and the contamination-guard result on the emitted pack — all candidate-only,
`claimStatus: Open`, exactly like every other Sophia benchmark artifact.
