# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gated experience bank — retrieve past *verified* trajectories as hints at plan time.

The training-data flywheel already mines run traces into SFT/DPO packs
(``tools/build_trajectory_pack.py``); what was missing is the *inference-time* read
path: a new ``long_horizon``/``subagent`` run starts blind even when a near-identical
task succeeded yesterday. This module closes that with a ReasoningBank-shaped store
rebuilt in Sophia's idiom:

  * **Write is gated (fail-closed).** ``add`` persists a record into the readable
    ``accepted`` stream **only** when the record carries machine verification evidence
    (``verdict == "accepted"`` and a non-empty ``verifiedBy`` list naming the
    verifier(s) that passed). Anything else lands in a ``quarantine`` stream that
    retrieval never reads — the same durable trust boundary as
    :mod:`agent.gated_memory`, applied to procedural memory.
  * **Retrieved patterns are hints, never authority.** ``format_hints`` wraps matches
    in explicitly advisory framing; every step of the new run still passes its own
    verifier, so a stale or wrong pattern costs one failed attempt, not a corrupted
    belief or an unverified success.
  * **Deterministic + offline + dependency-free.** Similarity is cosine over sparse
    hashed features (word unigrams + char trigrams, ``blake2b``-bucketed — the pure
    stdlib mirror of ``agent/rag_local_embed.py``), so the bank stays usable from
    ``long_horizon`` (which is deliberately numpy-free) and in CI. Ties break
    lexicographically on record id.

Honest bound: the gate on ``add`` checks that verification evidence is *presented and
well-formed*; it cannot re-execute a past run's verifier. Provenance of the evidence is
the caller's contract (the harness/subagent layer, which records verifier verdicts).
Similarity is lexical-semantic hashing, not learned embedding — it generalizes over
surface form, not deep meaning. Whether hint injection actually improves
verified-steps-per-task is an open measurement (pre-register before claiming uplift);
``canClaimAGI`` stays false.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "sophia.experience.v1"
BANK_PATH = ROOT / "agent" / "memory" / "experience_bank.jsonl"
QUARANTINE_PATH = ROOT / "agent" / "memory" / "experience_quarantine.jsonl"

_WORD_RE = re.compile(r"[a-z0-9一-鿿]+")
_MAX_HINT_CHARS = 700
_MAX_HINTS = 3


def _features(text: str) -> "dict[str, int]":
    """Word unigrams + char trigrams (incl. CJK), lowercased → term counts."""
    low = (text or "").lower()
    words = _WORD_RE.findall(low)
    counts: dict[str, int] = {}
    for w in words:
        counts[f"w:{w}"] = counts.get(f"w:{w}", 0) + 1
        padded = f"#{w}#"
        for i in range(len(padded) - 2):
            t = f"t:{padded[i:i + 3]}"
            counts[t] = counts.get(t, 0) + 1
    return counts


def _sparse_embed(text: str) -> "dict[int, float]":
    """Sparse signed-hash embedding, L2-normalized. Pure stdlib; same feature family
    as ``rag_local_embed`` so behavior is consistent across the memory stack."""
    weights: dict[int, float] = {}
    for feat, c in _features(text).items():
        h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(h, "big")
        bucket = val % 1024
        sign = 1.0 if (val >> 63) & 1 else -1.0
        weights[bucket] = weights.get(bucket, 0.0) + (1.0 + math.log(c)) * sign
    norm = math.sqrt(math.fsum(w * w for w in weights.values()))
    if norm > 0.0:
        return {b: w / norm for b, w in weights.items()}
    return weights


def _cosine(a: "dict[int, float]", b: "dict[int, float]") -> float:
    if len(b) < len(a):
        a, b = b, a
    return math.fsum(w * b[bucket] for bucket, w in a.items() if bucket in b)


def _record_id(record: dict) -> str:
    """Deterministic content id (sha256 of the canonical acceptance-relevant fields)."""
    payload = json.dumps(
        {k: record.get(k) for k in ("task", "outcomeSummary", "verifiedBy", "source")},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_record(record: dict) -> "list[str]":
    """Fail-closed acceptance predicate. Returns [] when the record is admissible,
    else the list of reasons it must be quarantined."""
    reasons: list[str] = []
    if not isinstance(record, dict):
        return ["record must be a dict"]
    task = record.get("task")
    if not (isinstance(task, str) and task.strip()):
        reasons.append("missing/empty task")
    if record.get("verdict") != "accepted":
        reasons.append("verdict != accepted (only verified successes are retrievable)")
    verified_by = record.get("verifiedBy")
    if not (isinstance(verified_by, list) and verified_by
            and all(isinstance(v, str) and v.strip() for v in verified_by)):
        reasons.append("verifiedBy must be a non-empty list of verifier names")
    summary = record.get("outcomeSummary")
    if not (isinstance(summary, str) and summary.strip()):
        reasons.append("missing/empty outcomeSummary")
    if len(str(summary or "")) > 4000:
        reasons.append("outcomeSummary too long (>4000 chars)")
    return reasons


@dataclass
class ExperienceBank:
    """Append-only, gated procedural memory with deterministic similarity search.

    ``add`` is the trust boundary; ``search``/``hints_for`` read only accepted rows.
    ``quarantined`` is an audit surface, never retrievable context.
    """

    path: Path = BANK_PATH
    quarantine_path: Path = QUARANTINE_PATH
    _cache: "list[dict] | None" = field(default=None, repr=False)

    # -- write side (fail-closed) ------------------------------------------------
    def add(self, record: dict, *, source: "str | None" = None) -> dict:
        record = dict(record or {})
        if source is not None:
            record.setdefault("source", source)
        reasons = validate_record(record)
        record["schema"] = SCHEMA
        record["id"] = _record_id(record)
        if reasons:
            self._append(self.quarantine_path, {**record, "reasons": reasons})
            return {"stored": False, "verdict": "held", "reasons": reasons, "id": record["id"]}
        if any(r.get("id") == record["id"] for r in self._accepted()):
            return {"stored": False, "verdict": "duplicate", "id": record["id"]}
        self._append(self.path, record)
        if self._cache is not None:
            self._cache.append(record)
        return {"stored": True, "verdict": "accepted", "id": record["id"]}

    def _append(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- read side (accepted only) -------------------------------------------------
    def _accepted(self) -> "list[dict]":
        if self._cache is not None:
            return self._cache
        rows: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # corrupt line skipped, never crashes retrieval
                if validate_record(row) == []:  # re-check on read: tampered rows drop out
                    rows.append(row)
        self._cache = rows
        return rows

    def search(self, task: str, *, k: int = _MAX_HINTS, min_score: float = 0.15) -> "list[dict]":
        """Top-k accepted records by cosine similarity to ``task``. Deterministic:
        ties break on record id. Below ``min_score`` nothing is returned (an
        irrelevant hint is worse than no hint)."""
        if not (task or "").strip():
            return []
        q = _sparse_embed(task)
        scored = []
        for row in self._accepted():
            s = _cosine(q, _sparse_embed(row["task"]))
            if s >= min_score:
                scored.append((round(s, 6), row))
        scored.sort(key=lambda t: (-t[0], t[1]["id"]))
        return [{"score": s, **row} for s, row in scored[:max(0, k)]]

    def hints_for(self, task: str, *, k: int = _MAX_HINTS) -> "str | None":
        """Advisory hint block for prompt injection, or None when nothing relevant."""
        matches = self.search(task, k=k)
        return format_hints(matches) if matches else None

    # -- audit surface ---------------------------------------------------------
    def quarantined(self) -> "list[dict]":
        rows: list[dict] = []
        if self.quarantine_path.exists():
            for line in self.quarantine_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    def audit(self) -> dict:
        return {"accepted": len(self._accepted()), "held": len(self.quarantined())}


def format_hints(matches: "list[dict]") -> str:
    """Render matches as an explicitly-advisory context block. Hints are suggestions:
    the current run's verifiers remain the only authority."""
    lines = [
        "Prior VERIFIED experience on similar tasks (advisory only — re-verify "
        "everything; do not treat these as ground truth):"
    ]
    for m in matches[:_MAX_HINTS]:
        summary = str(m.get("outcomeSummary", "")).strip()[:_MAX_HINT_CHARS]
        verified = ",".join(m.get("verifiedBy", []))
        lines.append(f"- [sim {m.get('score', 0):.2f}; verified by {verified}] "
                     f"task: {str(m.get('task', ''))[:200]} → {summary}")
    return "\n".join(lines)


def record_from_subagent(goal: str, result_text: str, *, verified_by: "list[str]",
                         source: str) -> dict:
    """Shape a successful, verifier-passed subagent outcome into a bank record.
    The caller asserts ``verified_by`` from the harness's own verdicts — this helper
    never invents evidence."""
    return {
        "task": goal,
        "outcomeSummary": (result_text or "").strip()[:4000],
        "verdict": "accepted",
        "verifiedBy": list(verified_by),
        "source": source,
    }


# --------------------------------------------------------------------------- #
# Offline invariants (CI-gated; no model, no network)
# --------------------------------------------------------------------------- #


def offline_invariants() -> "tuple[bool, dict]":
    import tempfile

    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as td:
        bank = ExperienceBank(path=Path(td) / "bank.jsonl",
                              quarantine_path=Path(td) / "quarantine.jsonl")

        # 1) A record WITHOUT verification evidence is held, never retrievable.
        bad = bank.add({"task": "compute 2+2", "outcomeSummary": "it is 4", "verdict": "accepted"})
        checks["unverified_held"] = (not bad["stored"]) and bad["verdict"] == "held"

        # 2) A verified record is stored and retrieved for a similar task.
        ok = bank.add(record_from_subagent(
            "Verify the boiling point of water at sea level",
            "Confirmed 100 C at 1 atm via source check.",
            verified_by=["gate.check_response"], source="test"))
        checks["verified_stored"] = ok["stored"]
        hits = bank.search("boiling point of water")
        checks["similar_retrieved"] = len(hits) == 1 and hits[0]["score"] > 0.3

        # 3) An unrelated query retrieves nothing (min_score floor).
        checks["irrelevant_empty"] = bank.search("俳句 about autumn leaves") == []

        # 4) Quarantined rows never surface in search.
        checks["quarantine_never_searched"] = all(
            r.get("verdict") == "accepted" for r in bank.search("compute 2+2", min_score=0.0))

        # 5) Duplicate add is refused (idempotent store).
        dup = bank.add(record_from_subagent(
            "Verify the boiling point of water at sea level",
            "Confirmed 100 C at 1 atm via source check.",
            verified_by=["gate.check_response"], source="test"))
        checks["duplicate_refused"] = (not dup["stored"]) and dup["verdict"] == "duplicate"

        # 6) Determinism: same query → identical result list.
        checks["deterministic"] = bank.search("boiling point of water") == \
            ExperienceBank(path=bank.path, quarantine_path=bank.quarantine_path
                           ).search("boiling point of water")

        # 7) Hint block is explicitly advisory.
        hint = bank.hints_for("boiling point of water")
        checks["hints_advisory"] = hint is not None and "advisory only" in hint

        # 8) Tampered row (evidence stripped on disk) drops out on read (fail-closed read).
        lines = bank.path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["verifiedBy"] = []
        bank.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        fresh = ExperienceBank(path=bank.path, quarantine_path=bank.quarantine_path)
        checks["tampered_row_dropped"] = fresh.search("boiling point of water") == []

        # 9) Audit reconciles.
        checks["audit_reconciles"] = bank.audit()["held"] >= 1

    ok_all = all(checks.values())
    return ok_all, {"checks": checks}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    for name, passed in detail["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"{'PASS' if ok else 'FAIL'} experience_memory offline_invariants")
    raise SystemExit(0 if ok else 1)
