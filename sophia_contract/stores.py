"""Durable, inspectable stores behind the contract.

All four are append-friendly JSONL (human-readable, greppable, hand-editable) with
an in-memory mode (``path=None``) for tests and golden vectors:

  - ClaimStore       — idempotent claim record (same key -> same claim, fail-closed
                       on a key reused with different content).
  - DecisionLog      — append-only record of every verdict the service emits.
  - PreferenceStore  — every human verdict, the feedback loop: re-verifying a claim
                       a human already ruled on returns the human's decision, so
                       review burden shrinks. Editable by hand.
  - Supersessions    — old_claim_id -> new_claim_id.
"""

from __future__ import annotations

import json
from pathlib import Path

from sophia_contract.errors import ContractError
from sophia_contract.models import content_fingerprint


def _append_jsonl(path: "Path | None", row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: "Path | None") -> "list[dict]":
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class ClaimStore:
    def __init__(self, path: "Path | None" = None):
        self.path = path
        self._by_id: dict = {}
        self._by_key: dict = {}
        for row in _read_jsonl(path):
            self._index(row)

    def _index(self, claim: dict) -> None:
        self._by_id[claim["claim_id"]] = claim
        # recover the key->id mapping from the stored fingerprint row
        if "_idempotency_key" in claim:
            self._by_key[claim["_idempotency_key"]] = claim["claim_id"]

    def get_by_id(self, claim_id: str) -> "dict | None":
        c = self._by_id.get(claim_id)
        return {k: v for k, v in c.items() if not k.startswith("_")} if c else None

    def all_claims(self) -> "list[dict]":
        """Return every stored claim (public fields only), insertion order. Used by
        lineage/PROV export; never exposes internal underscore-prefixed bookkeeping."""
        return [{k: v for k, v in c.items() if not k.startswith("_")}
                for c in self._by_id.values()]

    def record(self, claim: dict, idempotency_key: str) -> "tuple[dict, bool]":
        """Store a claim idempotently. Returns (claim, created). A repeat key with
        the SAME content returns the stored claim (created=False); a repeat key with
        DIFFERENT content fails closed with BAD_REQUEST (no silent divergence)."""
        existing_id = self._by_key.get(idempotency_key)
        if existing_id is not None:
            stored = self._by_id[existing_id]
            if stored.get("_fp") != content_fingerprint(claim["content"]):
                raise ContractError(
                    "BAD_REQUEST",
                    f"idempotency_key reused with different content (claim {existing_id})",
                )
            return self.get_by_id(existing_id), False
        row = dict(claim)
        row["_idempotency_key"] = idempotency_key
        row["_fp"] = content_fingerprint(claim["content"])
        self._index(row)
        _append_jsonl(self.path, row)
        return self.get_by_id(claim["claim_id"]), True


class DecisionLog:
    def __init__(self, path: "Path | None" = None):
        self.path = path
        self.entries: list = _read_jsonl(path)

    def append(self, entry: dict) -> None:
        self.entries.append(entry)
        _append_jsonl(self.path, entry)


class PreferenceStore:
    """Human verdicts, keyed by claim_id and by content fingerprint so a learned
    decision transfers to an identical re-submission under a new id."""

    def __init__(self, path: "Path | None" = None):
        self.path = path
        self._by_claim: dict = {}
        self._by_fp: dict = {}
        for row in _read_jsonl(path):
            self._index(row)

    def _index(self, row: dict) -> None:
        if row.get("claim_id"):
            self._by_claim[row["claim_id"]] = row
        if row.get("fingerprint"):
            self._by_fp[row["fingerprint"]] = row

    def record_human_verdict(self, *, claim_id: str, fingerprint: str, verdict: str,
                             note: str = "", reviewer: str = "founder") -> dict:
        if verdict not in ("accepted", "rejected", "superseded", "held"):
            raise ContractError("BAD_REQUEST", "human verdict must be a valid verdict value")
        row = {"claim_id": claim_id, "fingerprint": fingerprint, "verdict": verdict,
               "note": note, "reviewer": reviewer}
        self._index(row)
        _append_jsonl(self.path, row)
        return row

    def lookup(self, *, claim_id: str, fingerprint: str) -> "dict | None":
        return self._by_claim.get(claim_id) or self._by_fp.get(fingerprint)

    def all(self) -> "list[dict]":
        # de-duplicated latest view, by claim_id
        return list({r["claim_id"]: r for r in self._by_claim.values()}.values())


class Supersessions:
    def __init__(self, path: "Path | None" = None):
        self.path = path
        self._map: dict = {r["old"]: r["new"] for r in _read_jsonl(path)}

    def mark(self, old_id: str, new_id: str) -> None:
        self._map[old_id] = new_id
        _append_jsonl(self.path, {"old": old_id, "new": new_id})

    def successor(self, claim_id: str) -> "str | None":
        return self._map.get(claim_id)
