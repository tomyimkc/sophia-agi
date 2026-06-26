# Governed Scaling — a trust governor on every scaling primitive

**Status:** design thesis (no capability claim; `canClaimAGI` stays `false`).

> The path this repo bets on toward general intelligence is neither pure scale
> (the frontier-lab thesis: more compute, bigger models, better kernels) nor pure
> trust in isolation. It is **scale that carries its own proof** — efficiency
> primitives each wrapped in a *trust governor* drawn from Sophia's existing
> machinery (the gate, the OKF belief graph, calibration, the SSIL gates).

This doc is the umbrella for three feature designs that all instantiate one
pattern. It came out of building the [systems track](Systems-Track.md)
(KV cache, async RL, FlashAttention, MoE/quant) and noticing that **each
lab-efficiency primitive has a latent cognitive role Sophia already needs — and
the only thing it's missing is the governor Sophia already builds best.**

## The pattern

| Scaling primitive (systems track) | Latent cognitive role | Trust governor (existing seam) | Design doc |
|---|---|---|---|
| Async/off-policy RL with a staleness bound | Continual self-improvement | Verifier-as-reward + SSIL gates G1–G6 (`provenance_bench/rl_reward.py`, `tools/run_ssil*.py`) | [Continual-Governed-RL](Continual-Governed-RL.md) |
| Tiered KV cache, prefix sharing, eviction | Long-term memory + belief reuse | Gate-on-promote + OKF provenance (`agent/gate.py`, `okf/graph.py`, `okf/counterfactual.py`) | [Fail-Closed-Memory](Fail-Closed-Memory.md) |
| MoE top-k routing + load-balancing loss | Metacognitive resource allocation | Calibrated difficulty + monoculture alarm (`agent/calibration.py`, `agent/sector_council.py`) | [Routed-Metacognition](Routed-Metacognition.md) |
| FlashAttention exact-equivalence; quantization error bounds | Optimization that preserves correctness | The equivalence-proof bar (cross-cutting, below) |

## The four governors

1. **Promote only what verifies.** Anything a scaling primitive would *keep* — a
   cached belief, a self-generated trajectory, a routed answer — is admitted only
   after passing the gate. Fail-closed by construction; the same rule the
   reasoning loop already uses, applied to the *infrastructure's* state.

2. **Bound off-policy / off-distribution drift.** The async-RL staleness bound
   generalizes: a self-improvement step, a memory write, or a routing decision is
   accepted only within a measurable distance of trusted ground. Drift is a
   first-class, bounded quantity — not an afterthought.

3. **Make over-reliance measurable.** MoE's load-balancing loss is a
   metacognitive signal: monoculture in *which verifier / council / source*
   Sophia leans on is a number you can watch and penalize. This is pillar 4
   (functional self-modeling) given a concrete meter.

4. **Optimize only with an equivalence proof (the FlashAttention bar).** The
   kernel work's real lesson is not speed — it's *256× cheaper, provably identical
   output* (`flash == naive`, tested in `tests/test_flash_attention.py`).
   Every efficiency feature must clear that bar: an offline invariant showing the
   fast path's output is equivalent (or its error *bounded*, the quantization
   case) to the trusted reference. Quantization extends this to graceful
   degradation — bounded, **known** information loss is just calibration, and
   error bounds propagate through a reasoning chain the way `okf/graph`'s
   min-over-chain confidence already propagates.

## Why this is Sophia's program and not a lab's

A lab scales to make *more* computation possible. Governed scaling scales to make
*verifiable* computation possible — the efficiency mechanism and the trust
mechanism are the same object. It requires out-training no one (consistent with
[VISION.md](../../VISION.md): "assemble and orchestrate; innovate at the trust
layer"), and it converts "wisdom before intelligence" from a slogan into an
architecture: **every capability gain ships with the proof that it didn't cost
trust.**

## Non-goals / honesty

- No claim that any of this yields AGI, sentience, or open-ended RSI. The
  self-improvement design is explicitly *bounded* (see SSIL); `canClaimAGI`
  stays `false`.
- Each design doc carries falsifiable offline invariants and a no-overclaim
  measurement gate, exactly like the rest of the repo. A feature that can't be
  measured against a trusted reference doesn't ship.

## Build order (recommended)

1. **Continual-Governed-RL** — highest leverage; the async-RL engine plugs into
   the *existing* SSIL, which already has the gates. This is the thesis.
2. **Fail-Closed-Memory** — fills the repo's weakest pillar (memory is ⚠️ in
   VISION) with a concrete, gate-governed hierarchy.
3. **Routed-Metacognition** — a calibration/metacognition layer over the councils
   that already exist.
