# Fail-Closed Memory — a verification-gated tiered belief store

**Status:** Phase 0 **RUNNABLE** — `agent/belief_store.py` (`python -m
agent.belief_store`) implements the gate-on-promote tiered store against the real
`provenance_faithful` gate + a real OKF graph, with provenance-preserving,
audited eviction and belief reuse; 12 offline invariants in
`tests/test_belief_store.py`. Phases 1–2 (live grounded-path wiring + background
consolidation) remain design. Fills the repo's weakest pillar — memory is marked
⚠️ partial in [VISION.md](../../VISION.md).

> **The insight.** The [systems track](Systems-Track.md)'s tiered KV cache *is* a
> memory architecture. Two of its mechanisms map directly onto cognition, and the
> only thing they're missing is Sophia's gate:
> - **Prefix sharing → belief reuse.** Two reasoning chains sharing a premise
>   should share its *already-verified justification*, not re-derive it.
> - **Promotion/eviction → an auditable remember/forget policy** — where promotion
>   to the "hot/trusted" tier requires passing the verification gate.

## Where memory is today

`agent/memory.py` is a 44-line append-only JSONL log (`log_decision` /
`recent_decisions`). The OKF belief graph (`okf/graph.py`) is rich — confidence
propagation, contradiction ledger, counterfactual removal, retraction — but it is
a *curated corpus*, not a *runtime* store that grows from the agent's own verified
conclusions. There is no governed path from "Sophia concluded X and verified it"
to "X is now durable, reusable, provenance-tagged memory." This design adds it.

## Design: tiers, with a gate on promotion

Reuse the `TieredKVCache` structure (`serving/kv_cache.py`), but the unit is a
**belief block** (a claim + its justification + provenance), not opaque KV bytes:

```
HOT   (trusted)   — gate-passed, grounded beliefs; small, fast; the working set.
WARM  (provisional) — concluded but not yet re-verified; usable with a hedge.
COLD  (archival)  — on-disk, provenance-tagged; recoverable, not in the hot path.
```

Promotion and eviction are **governed**, not LRU-by-recency alone:

```
promote(belief -> HOT) iff
    gate.check_response(belief.claim) passes          # agent/gate.py, fail-closed
  and okf.counterfactual.is_grounded(graph, belief)   # grounds to trusted corpus
  and confidence(belief) >= hot_threshold             # okf.graph.propagate_confidence

demote/evict(belief) when
    contradiction_ledger flags it (okf.graph.contradiction_ledger), OR
    counterfactual_remove shows its support vanished (source retracted), OR
    capacity pressure + lowest (confidence × recency)  # the cache's LRU, trust-weighted
```

**Belief reuse (prefix sharing).** Before a reasoning chain re-derives a premise,
it looks it up by content hash (the cache's `block_hashes` over a normalized claim
key). A HOT hit returns the *verified justification*, skipping re-derivation — the
cognitive analogue of a prefix-cache hit, and it inherits the provenance so the
reuse is auditable.

**Forgetting is fail-closed.** Eviction never deletes provenance; a belief demoted
to COLD remains recoverable and its retraction is logged
(`okf.counterfactual.retract` → `audit_entry`). A contradicted belief is demoted
*out of the hot path* immediately — the system never serves from a belief it can
no longer defend.

## Falsifiable offline invariants (CI-gated)

1. **No ungated promotion.** A belief failing `gate.check_response` never reaches
   HOT (red-team: a fabricated attribution stays out of the working set).
2. **Grounding required.** A belief that fails `is_grounded` against the corpus is
   not promoted (counted in `rejected_ungrounded`).
3. **Contradiction demotes.** Injecting a contradicting source moves the belief
   out of HOT within one consolidation pass (`contradiction_ledger` fires).
4. **Reuse preserves provenance.** A prefix/belief-reuse hit returns the *same*
   provenance chain as the original derivation (no laundering — cross-check
   `okf.graph.confidence_laundering` stays empty).
5. **Forgetting is recoverable + audited.** An evicted belief is reconstructable
   from COLD and its retraction has an `audit_entry`; capacity bounds hold (carried
   from `kv_cache` invariants).
6. **Counterfactual coherence.** Removing a source retracts exactly its
   downstream HOT beliefs (`counterfactual_remove` delta == demoted set).

## Wiring

```
agent concludes X (verified)  ──▶  BeliefStore.offer(X, justification, provenance)
                                       │  promote() predicate above
        reasoning chain needs premise P                      │
                ▼                                             ▼
        BeliefStore.lookup(P)  ──hit──▶ verified justification (reuse, no re-derive)
                │ miss                                  consolidation pass (rolling):
                ▼                                       re-verify WARM, demote contradicted,
        derive + verify + offer                         evict COLD under capacity
```

`BeliefStore` is `TieredKVCache` with: block payload = a belief record; the budget
governs HOT/WARM working-set size; the disk tier = COLD (already implemented). The
graph operations are pure functions already in `okf/`. `agent/memory.log_decision`
becomes the COLD write path's audit log.

## Phasing

- **Phase 0 (offline, CI):** `BeliefStore` over `TieredKVCache` + the promote/demote
  predicates calling `agent/gate.py` and `okf/`; invariants 1–6 on a synthetic
  belief set. No new heavy deps.
- **Phase 1:** wire `BeliefStore.lookup` into `agent/grounded_agent.py` so the live
  grounded path reuses HOT beliefs; measure re-derivation avoided and any change in
  the graded-confidence discrimination already reported in VISION.
- **Phase 2:** rolling consolidation as a background pass (the "memory consolidation"
  stub `agent/memory_consolidation.py` becomes real), promoting/demoting on a
  schedule with full audit.

## Non-goals

- Not a claim of episodic/human memory or consciousness (pillar 4 stays functional
  self-modeling only). It is a *governed store*, not experience.
- No silent forgetting: every eviction is recoverable and logged.
