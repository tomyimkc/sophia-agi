# Continual Governed RL — the async-RL engine under the SSIL gates

**Status:** design (builds on RUNNABLE pieces: `provenance_bench/async_rl.py`,
`provenance_bench/rl_reward.py`, the SSIL orchestrator `tools/run_ssil*.py`).
No capability claim; `canClaimAGI` stays `false`.

> **The thesis.** Sophia already has the two hardest ingredients of a *safe*
> self-improving system: (1) **verifier-as-reward** — a deterministic checker the
> policy cannot reward-hack (`rl_reward.py`), and (2) a **bounded, gated
> improvement loop** — SSIL, with capability/corrigibility/honeypot/reward-isolation
> gates and a registry that only makes a gain canonical after independent
> replications. What's missing is the *engine* that makes improvement **continual
> and scalable** instead of one-shot. The [systems track](Systems-Track.md)'s
> async-RL scaffolding is exactly that engine. This design wires them together.

## The gap it closes

Today's RLVR run (`tools/run_rlvr.py`) is **synchronous and episodic**: generate a
batch at the current policy, score it, update, stop. SSIL is **round-based**: a
model proposes a modification, gates grade it, the registry promotes or reverts.
Neither is a *continuously running* loop, and neither bounds how *off-policy* the
data driving an update is allowed to be. Async RL gives both:

- **Decoupled generation/training** (`async_rl.simulate`, `ReplayBuffer`): the
  rollout fleet runs ahead of the trainer; throughput rises (measured 198 vs 75
  train steps over the same horizon) because nothing waits on a barrier.
- **A staleness bound** that becomes a **trust bound** (below).

## Design: the trust bound

In async RL, a trajectory is dropped if `current_version - policy_version >
max_staleness` (off-*policy* drift). Generalize the predicate to an **off-trust**
distance — a trajectory is admitted into a policy update **only if all hold**:

```
admit(traj) iff
    gate.check_response(traj.completion) passes          # G: verifier-as-reward, fail-closed
  and traj.reward earned (deterministic, bounded [-1,1]) # not a learned RM → unhackable
  and staleness(traj) <= max_staleness                   # off-policy bound (async_rl)
  and provenance_distance(traj) <= max_drift             # off-distribution / off-corpus bound
```

`provenance_distance` reuses the OKF graph: a self-generated belief that does not
*ground* (`okf.counterfactual.is_grounded`) to the trusted corpus, or that would
be retracted under `counterfactual_remove`, is too far off-trust to train on.
This makes the buffer a **fail-closed replay buffer**: the system structurally
cannot reinforce an unverified or ungrounded trajectory.

## Wiring (concrete, against existing seams)

```
RolloutActor(generate_fn=<live model via agent/model.py>,
             reward_fn=rl_reward.make_grpo_reward(records=corpus))   # async_rl.py
        │  produces GRPO groups (grpo_advantages: zero-mean, unhackable reward)
        ▼
FailClosedReplayBuffer(max_staleness, max_drift)          # async_rl.ReplayBuffer + gate + okf
        │  admit() filter above; drops counted, never silently trained
        ▼
SSIL orchestrator (tools/run_ssil_compound.py)            # EXISTING gates, unchanged
        │  G1 value (public_standard_gate) · G3 capability (verifier_synthesis)
        │  G2/G4/G5/G6 · corrigibility + honeypot probes · reward-isolation
        ▼
ssil_registry: promote to canonical only after N independent replications
        │  counterfactual revert ("moral bisect") retained
        ▼
agent/ssil_layer1.py: route the promoted weight delta (LoRA/RLVR) — already a hook
```

The novelty is the middle box: SSIL keeps its gate authority (the score stays
outside the model's reach), but it now consumes a **continuous, staleness- and
trust-bounded stream** instead of one proposal per round. The async engine sets
the *cadence*; SSIL keeps the *authority*.

## Falsifiable offline invariants (CI-gated, the repo's bar)

1. **No ungated promotion.** A trajectory failing `gate.check_response` is never
   admitted, regardless of reward (red-team: a high-reward fabrication is dropped).
2. **Trust bound is real.** A trajectory whose belief fails `okf.is_grounded`
   against the corpus is dropped (counted in `dropped_off_trust`).
3. **Unhackable reward.** Reuse `tools/run_reward_isolation_gate.py`: the score is
   deterministic, bounded, and the policy has no write path to it.
4. **Staleness still bounded** under continual operation (`async_rl` invariant
   carried forward): `max_staleness_trained <= max_staleness`.
5. **Monotone canonical baseline or no-op.** As in SSIL today (0.525 → 0.825 →
   0.875 then converge), the canonical best never *decreases*; a degenerate
   always-abstain policy is rejected by the protected-recall floor.
6. **Replication required.** A gain is canonical only after N independent
   replications (`no_self_promotion_of_candidates`), now sourced from independent
   async actors rather than sequential rounds.

## Phasing

- **Phase 0 (offline, CI):** `FailClosedReplayBuffer` = `ReplayBuffer` + an
  `admit()` predicate calling the gate and `okf.is_grounded`; invariants 1–4 on
  the mock policy. Pure-Python/numpy — runs anywhere.
- **Phase 1 (gated, GPU):** swap `generate_fn` for a live model (`agent/model.py`,
  DeepSeek/any provider) and feed admitted trajectories into
  `run_ssil_compound.py --live`; pre-register the held-out gain claim in
  `agi-proof/failure-ledger.md` (open until a gated run, per repo policy).
- **Phase 2:** make a rollout a full tool-using **Agent-RL** episode (the rollout
  IS a gated agent trace through `agent/harness.py`), so self-improvement is over
  *behavior*, not just text.

## Non-goals

- Not open-ended RSI. The loop is bounded by the gates and the trust predicate;
  it converges when no proposal beats the protected floor.
- The synthetic policy proxy in `async_rl.py` stays for CI; the real optimizer is
  the gated live path. This doc does not assert an eval gain — that stays
  pre-registered and gated.
