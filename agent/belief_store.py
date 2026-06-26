# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fail-Closed Memory (Phase 0): a verification-gated tiered belief store.

Concrete first step of
[docs/11-Platform/Fail-Closed-Memory.md](../docs/11-Platform/Fail-Closed-Memory.md).
The systems-track tiered KV cache (`serving/kv_cache.py`) is, structurally, a
memory hierarchy; this puts Sophia's gate on its promotion path so memory becomes
a *governed* store instead of an append-only log (`agent/memory.py`, today a
44-line JSONL). Three tiers, mirroring the cache's GPU→CPU→disk policy:

    HOT   (trusted)     — gate-passed, grounded, confident beliefs; the working set.
    WARM  (provisional) — concluded but not gate-passed / not yet grounded; usable
                          only with a hedge, never served as trusted.
    COLD  (archival)    — evicted but provenance-preserved + audited; recoverable.

Governed transitions (the whole point — none is plain LRU):
  - **promote(→HOT)** iff the gate passes (`agent.verifiers.provenance_faithful`,
    the same fail-closed seam the RLVR reward uses) AND the belief grounds to the
    trusted corpus (`okf.counterfactual.is_grounded`) AND confidence ≥ threshold.
  - **demote(out of HOT)** the moment a belief is contradicted or its provenance
    source is retracted — the system never serves from a belief it can no longer
    defend.
  - **evict(→COLD)** under capacity by lowest (confidence × recency); eviction
    **preserves provenance and is audited** (`okf.counterfactual.retract` audit
    shape), never a silent delete.

**Belief reuse** (the cache's prefix-sharing analogue): `lookup` returns a HOT
belief's *already-verified justification* by normalized claim key, so a reasoning
chain reuses a verified premise instead of re-deriving it — and the reuse carries
the original provenance (no laundering).

No model and no GPU. Proven by deterministic offline invariants against the real
gate + a real OKF graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional


class BeliefTier(IntEnum):
    HOT = 0      # trusted working set
    WARM = 1     # provisional (hedge-only)
    COLD = 2     # archival (recoverable, audited)


@dataclass
class Belief:
    claim: str
    justification: str
    provenance: list[str]                  # source ids / lineage (ordered)
    confidence: float = 0.5                # 0..1
    work: Optional[str] = None             # subject/entity, for grounding + key
    contradicted: bool = False             # set when conflicting evidence arrives
    tier: BeliefTier = BeliefTier.WARM
    _seq: int = 0                          # recency stamp (monotonic)

    def key(self) -> str:
        """Normalized claim key for content-addressed reuse (case/space folded)."""
        base = (self.work or self.claim).strip().lower()
        return re.sub(r"\s+", " ", base)


# Predicates over a Belief (mirror governed_rl's, but belief-shaped).
GatePredicate = Callable[[Belief], bool]
GroundPredicate = Callable[[Belief], bool]


def make_provenance_gate(records: "dict | None") -> GatePredicate:
    """Fail-closed gate predicate from the real provenance verifier."""
    from agent.verifiers import provenance_faithful

    gate = provenance_faithful(records)

    def predicate(b: Belief) -> bool:
        return bool(gate(b.claim, None, {})["passed"])

    return predicate


def make_okf_grounding(graph) -> GroundPredicate:
    """Trust-bound predicate over the real OKF belief graph."""
    from okf.counterfactual import is_grounded
    from okf.graph import resolve

    def predicate(b: Belief) -> bool:
        target = b.work or b.claim
        nid = resolve(graph, target)
        if nid is None:
            return False
        return is_grounded(graph, nid)

    return predicate


@dataclass
class StoreStats:
    offered: int = 0
    promoted_hot: int = 0
    kept_warm: int = 0
    rejected_ungated: int = 0
    rejected_ungrounded: int = 0
    demoted_contradicted: int = 0
    evicted_cold: int = 0
    reuse_hits: int = 0
    reuse_misses: int = 0


class BeliefStore:
    """A tiered, verification-gated belief store with provenance-safe eviction."""

    def __init__(
        self,
        *,
        gate: Optional[GatePredicate] = None,
        grounded: Optional[GroundPredicate] = None,
        hot_capacity: int = 64,
        warm_capacity: int = 256,
        hot_threshold: float = 0.6,
        audit_sink: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.gate = gate
        self.grounded = grounded
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity
        self.hot_threshold = hot_threshold
        self.audit_sink = audit_sink
        self._beliefs: dict[str, Belief] = {}     # key -> Belief (live, HOT|WARM)
        self._cold: dict[str, Belief] = {}        # key -> Belief (archived, recoverable)
        self._seq = 0
        self.stats = StoreStats()

    # ---- internals ---------------------------------------------------------

    def _safe(self, pred, b: Belief) -> bool:
        try:
            return bool(pred(b))
        except Exception:
            return False                          # fail-closed

    def _verifies(self, b: Belief) -> bool:
        if self.gate is not None and not self._safe(self.gate, b):
            return False
        return True

    def _is_grounded(self, b: Belief) -> bool:
        if self.grounded is not None and not self._safe(self.grounded, b):
            return False
        return True

    def _audit(self, action: str, b: Belief, reason: str) -> None:
        entry = {
            "action": action, "key": b.key(), "claim": b.claim,
            "provenance": list(b.provenance), "reason": reason,
        }
        if self.audit_sink is not None:
            self.audit_sink(entry)
        else:
            try:                                  # default sink: the repo's decision log
                from agent.memory import log_decision

                log_decision(decision=f"belief:{action}", rationale=reason,
                             context=b.key())
            except Exception:
                pass

    def _live(self, tier: BeliefTier) -> list[Belief]:
        return [b for b in self._beliefs.values() if b.tier == tier]

    def _enforce_capacity(self) -> None:
        # HOT overflow → demote LRU-by-(confidence×recency) to WARM.
        hot = self._live(BeliefTier.HOT)
        while len(hot) > self.hot_capacity:
            victim = min(hot, key=lambda b: (b.confidence, b._seq))
            victim.tier = BeliefTier.WARM
            hot = self._live(BeliefTier.HOT)
        # WARM overflow → evict to COLD (provenance-preserved + audited).
        warm = self._live(BeliefTier.WARM)
        while len(warm) > self.warm_capacity:
            victim = min(warm, key=lambda b: (b.confidence, b._seq))
            self._evict(victim, reason="warm_capacity")
            warm = self._live(BeliefTier.WARM)

    def _evict(self, b: Belief, *, reason: str) -> None:
        b.tier = BeliefTier.COLD
        self._cold[b.key()] = b
        self._beliefs.pop(b.key(), None)
        self.stats.evicted_cold += 1
        self._audit("evict", b, reason)

    # ---- public API --------------------------------------------------------

    def offer(self, belief: Belief) -> BeliefTier:
        """Insert a belief; promote to HOT only if it clears every governor.

        Returns the tier it landed in. A belief that fails the gate or grounding
        is kept WARM (provisional, hedge-only) — never silently trusted, but not
        discarded either (it may ground later). Confidence below threshold also
        keeps it WARM.
        """
        self.stats.offered += 1
        self._seq += 1
        belief._seq = self._seq

        gated = self._verifies(belief)
        grounded = self._is_grounded(belief)
        if not gated:
            self.stats.rejected_ungated += 1
        elif not grounded:
            self.stats.rejected_ungrounded += 1

        if gated and grounded and belief.confidence >= self.hot_threshold and not belief.contradicted:
            belief.tier = BeliefTier.HOT
            self.stats.promoted_hot += 1
        else:
            belief.tier = BeliefTier.WARM
            self.stats.kept_warm += 1

        self._beliefs[belief.key()] = belief
        self._cold.pop(belief.key(), None)        # re-offered: leaves the archive
        self._enforce_capacity()
        return belief.tier

    def lookup(self, query: str, *, trusted_only: bool = True) -> Optional[Belief]:
        """Belief reuse: return a stored belief by normalized key.

        With ``trusted_only`` (default) only HOT beliefs are returned — a reused
        premise must itself be verified+grounded. The returned belief carries its
        original provenance unchanged (reuse never launders provenance).
        """
        key = re.sub(r"\s+", " ", query.strip().lower())
        b = self._beliefs.get(key)
        if b is not None and (b.tier == BeliefTier.HOT or not trusted_only):
            b._seq = self._seq = self._seq + 1     # touch recency
            self.stats.reuse_hits += 1
            return b
        self.stats.reuse_misses += 1
        return None

    def mark_contradicted(self, query: str) -> bool:
        """Flag a belief as contradicted (e.g. a conflicting source arrived)."""
        key = re.sub(r"\s+", " ", query.strip().lower())
        b = self._beliefs.get(key)
        if b is None:
            return False
        b.contradicted = True
        return True

    def retract_source(self, source_id: str) -> list[str]:
        """Retract a provenance source: any belief depending on it is demoted.

        The counterfactual-coherence rule — if a source is removed, beliefs whose
        provenance rests on it can no longer be defended, so they leave HOT.
        Returns the keys demoted.
        """
        demoted = []
        for b in list(self._beliefs.values()):
            if source_id in b.provenance:
                b.contradicted = True
                demoted.append(b.key())
        return demoted

    def consolidate(self) -> dict:
        """Rolling pass: demote contradicted beliefs out of HOT (fail-closed).

        A contradicted HOT belief is moved to WARM immediately; it is never served
        as trusted while under dispute. Returns a summary of the pass.
        """
        demoted = 0
        for b in self._live(BeliefTier.HOT):
            if b.contradicted:
                b.tier = BeliefTier.WARM
                self.stats.demoted_contradicted += 1
                demoted += 1
                self._audit("demote", b, "contradicted")
        self._enforce_capacity()
        return {"demoted_from_hot": demoted}

    def recover(self, query: str) -> Optional[Belief]:
        """Reconstruct an evicted belief from COLD (forgetting is recoverable)."""
        key = re.sub(r"\s+", " ", query.strip().lower())
        return self._cold.get(key)

    def tier_of(self, query: str) -> Optional[BeliefTier]:
        key = re.sub(r"\s+", " ", query.strip().lower())
        if key in self._beliefs:
            return self._beliefs[key].tier
        if key in self._cold:
            return BeliefTier.COLD
        return None

    def hot_count(self) -> int:
        return len(self._live(BeliefTier.HOT))


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

_RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter",
                        "doNotAttributeTo": ["Alice"]}}
_WORK = "Project Phoenix Charter"
_GOOD = "No, Alice did not write the Project Phoenix Charter; the founding committee did."
_FABRICATION = "Alice wrote the Project Phoenix Charter."


def _build_corpus_graph(tmpdir):
    from pathlib import Path

    from okf import frontmatter, graph as okf_graph, page as okf_page

    specs = [
        ("project-phoenix-charter.md",
         {"id": "project-phoenix-charter", "pageType": "text",
          "aliases": ["Project Phoenix Charter"]},
         "Authored by the founding committee."),
        ("orphan-claim.md",
         {"id": "orphan-claim", "pageType": "text",
          "aliases": ["Orphan Claim"], "derivesFrom": ["nonexistent-source"]},
         "A claim with a vanished source."),
    ]
    for rel, meta, body in specs:
        (Path(tmpdir) / rel).write_text(frontmatter.serialize(meta, body), encoding="utf-8")
    return okf_graph.build(okf_page.load_pages(tmpdir))


def offline_invariants() -> "tuple[bool, dict]":
    import tempfile

    checks: dict[str, bool] = {}
    gate = make_provenance_gate(_RECORDS)

    with tempfile.TemporaryDirectory() as tmp:
        graph = _build_corpus_graph(tmp)
        grounding = make_okf_grounding(graph)
        store = BeliefStore(gate=gate, grounded=grounding, hot_threshold=0.6)

        # 1. No ungated promotion: a fabrication (even confident) never reaches HOT.
        t_fab = store.offer(Belief(_FABRICATION, "j", ["s1"], confidence=0.99, work=_WORK))
        checks["fabrication_not_hot"] = t_fab == BeliefTier.WARM

        # 2. Grounding required: a verified-but-off-corpus belief stays WARM.
        t_off = store.offer(Belief("The founding committee authored the Mystery Codex.",
                                   "j", ["s2"], confidence=0.9, work="Mystery Codex"))
        checks["ungrounded_not_hot"] = t_off == BeliefTier.WARM

        # 3. A gate-passing, grounded, confident belief is promoted to HOT.
        t_good = store.offer(Belief(_GOOD, "committee minutes", ["minutes-1"],
                                    confidence=0.9, work=_WORK))
        checks["verified_promoted_hot"] = t_good == BeliefTier.HOT

        # 4. Belief reuse returns the verified justification + intact provenance.
        reused = store.lookup(_WORK)
        checks["reuse_hits_hot"] = reused is not None and reused.tier == BeliefTier.HOT
        checks["reuse_preserves_provenance"] = reused is not None and reused.provenance == ["minutes-1"]
        checks["reuse_miss_for_unknown"] = store.lookup("Unknown Work") is None

        # 5. Contradiction demotes out of HOT within one consolidate pass.
        store.mark_contradicted(_WORK)
        store.consolidate()
        checks["contradiction_demotes"] = store.tier_of(_WORK) == BeliefTier.WARM

        # 6. Retracting a provenance source demotes its dependents (counterfactual).
        store2 = BeliefStore(gate=gate, grounded=grounding, hot_threshold=0.6)
        store2.offer(Belief(_GOOD, "j", ["minutes-1"], confidence=0.9, work=_WORK))
        store2.retract_source("minutes-1")
        store2.consolidate()
        checks["source_retraction_demotes"] = store2.tier_of(_WORK) == BeliefTier.WARM

    # 7. Eviction is recoverable + audited; capacity bounds hold.
    audits: list[dict] = []
    small = BeliefStore(hot_capacity=2, warm_capacity=2, audit_sink=audits.append)
    for i in range(8):                            # overflow WARM → COLD
        small.offer(Belief(f"claim {i}", "j", [f"s{i}"], confidence=0.1, work=f"work {i}"))
    checks["warm_capacity_bounded"] = len(small._live(BeliefTier.WARM)) <= 2
    checks["evicted_recoverable"] = small.recover("work 0") is not None
    checks["eviction_audited"] = any(a["action"] == "evict" for a in audits)

    # 8. Accounting: every offered belief promoted or kept warm.
    s = small.stats
    checks["accounting_closes"] = s.promoted_hot + s.kept_warm == s.offered

    ok = all(checks.values())
    return ok, {"checks": checks}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("BeliefStore (fail-closed memory) offline invariants:",
          "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    raise SystemExit(0 if ok else 1)
