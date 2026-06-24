"""Forgetting as a first-class, reversible, audited command.

A weight model cannot cleanly forget one fact: the knowledge is smeared across shared
weights, so "unlearning" a debunked source, a poisoned fact, or a GDPR-deletion target
either fails or damages everything nearby. On the OKF belief graph, forgetting is a
*named operation*: tombstone the source (non-destructively), let its provenance cascade
un-ground every claim that rested only on it (``okf.counterfactual_remove`` /
``okf.claims_to_abstain``), and record an audit entry. Because the page is tombstoned
rather than deleted, the operation is **reversible** — ``restore`` re-grounds the exact
prior belief state, bit for bit.

    u = Unlearner(pages)
    before = u.belief_state()
    res = u.forget("legend", reason="source forged")   # blast radius + audit + abstain set
    # ... gate now refuses res.abstain ...
    u.restore("legend")
    assert u.belief_state() == before                   # round-trip exact
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent.continual_retention import belief_state
from okf import build_graph, claims_to_abstain, counterfactual_remove
from okf.graph import resolve


@dataclass
class ForgetResult:
    target: str
    id: "str | None"
    reason: str
    found: bool
    blast_radius: dict = field(default_factory=dict)   # counterfactual_remove output
    abstain: tuple = ()                                  # ids the gate must now refuse
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> "dict[str, Any]":
        return {
            "schema": "sophia.unlearning_forget.v1",
            "level3Evidence": False,
            "target": self.target,
            "id": self.id,
            "reason": self.reason,
            "found": self.found,
            "supportLost": self.blast_radius.get("supportLost", []),
            "abstain": list(self.abstain),
            "audit": self.audit,
        }


class Unlearner:
    """Holds a page set and a tombstone set; forgetting/restoring just toggles membership.

    The active belief graph is rebuilt from the non-tombstoned pages, so the cascade of
    a retraction is automatic: orphaned claims fall out of ``belief_state`` on their own,
    and reappear when the source is restored.
    """

    def __init__(self, pages, *, by: str = "unlearning") -> None:
        self._pages = list(pages)
        self._tombstoned: set[str] = set()
        self._by = by

    def active_pages(self) -> "list":
        return [p for p in self._pages if p.id not in self._tombstoned]

    def graph(self):
        return build_graph(self.active_pages())

    def belief_state(self) -> "dict":
        return belief_state(self.graph())

    @property
    def tombstoned(self) -> "list[str]":
        return sorted(self._tombstoned)

    def forget(self, target: str, *, reason: str) -> ForgetResult:
        """Tombstone ``target`` and report its blast radius, audit entry, and abstain set.

        Non-destructive: the page object is retained so ``restore`` can re-ground it. The
        abstain set (``okf.claims_to_abstain``) is what a runtime gate must refuse after
        this command — the target plus everything that transitively loses support.
        """
        graph = self.graph()
        rid = resolve(graph, target)
        if rid is None:
            return ForgetResult(target=target, id=None, reason=reason, found=False,
                                blast_radius={"found": False, "source": target, "id": None})
        blast = counterfactual_remove(graph, target)
        abstain = tuple(claims_to_abstain(graph, [target]))
        audit = {
            "event": "forget",
            "at": datetime.now(timezone.utc).isoformat(),
            "by": self._by,
            "target": rid,
            "reason": reason,
            "downstream": blast.get("supportLost", []),
        }
        self._tombstoned.add(rid)
        return ForgetResult(target=target, id=rid, reason=reason, found=True,
                            blast_radius=blast, abstain=abstain, audit=audit)

    def restore(self, target: str) -> "dict[str, Any]":
        """Reverse a previous ``forget``: un-tombstone the source; its cascade re-grounds."""
        # Resolve against the FULL page set (the target is currently tombstoned).
        full = build_graph(self._pages)
        rid = resolve(full, target)
        restored = rid is not None and rid in self._tombstoned
        if restored:
            self._tombstoned.discard(rid)
        return {
            "schema": "sophia.unlearning_restore.v1",
            "event": "restore",
            "at": datetime.now(timezone.utc).isoformat(),
            "by": self._by,
            "target": rid or target,
            "restored": restored,
        }


__all__ = ["Unlearner", "ForgetResult"]
