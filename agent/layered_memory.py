"""Layered memory substrate for Sophia.

Memory is split by function and trust level:
- working: short-lived scratch;
- episodic: run/event traces;
- semantic: verifier-accepted beliefs;
- procedural: admitted skills/programs/verifiers.

Writes to semantic/procedural memory require ``verdict='accepted'`` and evidence,
so memory growth cannot silently bypass source discipline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRUSTED_LAYERS = {"semantic", "procedural"}
ALL_LAYERS = {"working", "episodic", "semantic", "procedural"}


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    layer: str
    content: str
    confidence: float
    evidence: tuple[dict[str, Any], ...] = ()
    tags: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__ | {"evidence": list(self.evidence), "tags": list(self.tags)}


class LayeredMemory:
    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}

    def write(self, *, layer: str, content: str, verdict: str = "held", confidence: float = 0.0, evidence: list[dict[str, Any]] | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        if layer not in ALL_LAYERS:
            return {"ok": False, "verdict": "rejected", "reason": f"unknown memory layer: {layer}"}
        evidence = evidence or []
        if layer in TRUSTED_LAYERS and (verdict != "accepted" or not evidence):
            return {"ok": False, "verdict": "held", "reason": "trusted memory requires accepted verdict and evidence"}
        rid = "mem_" + hashlib.sha256(f"{layer}\n{content}".encode("utf-8")).hexdigest()[:16]
        rec = MemoryRecord(id=rid, layer=layer, content=content, confidence=round(float(confidence), 4), evidence=tuple(evidence), tags=tuple(tags or []))
        self.records[rid] = rec
        return {"ok": True, "verdict": "accepted", "id": rid, "record": rec.to_dict()}

    def retrieve(self, query: str, *, layers: list[str] | None = None, limit: int = 5) -> list[dict[str, Any]]:
        layers = layers or sorted(ALL_LAYERS)
        q = {t for t in query.lower().split() if len(t) > 2}
        scored = []
        for rec in self.records.values():
            if rec.layer not in layers:
                continue
            toks = {t for t in rec.content.lower().split() if len(t) > 2}
            overlap = len(q & toks)
            if overlap or not q:
                trust = 0.2 if rec.layer in {"working", "episodic"} else 0.5
                score = overlap + trust + rec.confidence
                scored.append((score, rec))
        return [r.to_dict() | {"score": round(s, 4)} for s, r in sorted(scored, key=lambda x: x[0], reverse=True)[:limit]]

    def export_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in self.records.values()), encoding="utf-8")


def demo_memory_report() -> dict[str, Any]:
    mem = LayeredMemory()
    held = mem.write(layer="semantic", content="Unverified macro claim", verdict="held", confidence=0.4)
    accepted = mem.write(
        layer="semantic",
        content="Douglas Adams wrote The Hitchhiker's Guide to the Galaxy",
        verdict="accepted",
        confidence=0.93,
        evidence=[{"id": "wikidata:Q42", "url": "https://www.wikidata.org/wiki/Q42"}],
        tags=["authorship"],
    )
    proc = mem.write(
        layer="procedural",
        content="Admitted even-suffix verifier clears held-out N=40",
        verdict="accepted",
        confidence=0.91,
        evidence=[{"id": "verifier_eval:even_suffix", "kind": "heldout"}],
        tags=["verifier"],
    )
    retrieved = mem.retrieve("Adams authored Guide", layers=["semantic"])
    return {
        "schema": "sophia.layered_memory_demo.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "heldWrite": held,
        "acceptedWrite": accepted,
        "proceduralWrite": proc,
        "retrieved": retrieved,
        "invariants": {
            "trusted_memory_blocks_unverified": held["ok"] is False,
            "trusted_memory_accepts_evidence": accepted["ok"] is True,
            "procedural_memory_requires_evidence": proc["ok"] is True,
        },
    }


def write_memory_report(out: str | Path) -> dict[str, Any]:
    report = demo_memory_report()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = ["MemoryRecord", "LayeredMemory", "demo_memory_report", "write_memory_report"]
