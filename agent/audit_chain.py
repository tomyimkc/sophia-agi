# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tamper-evident audit log (P5) — a hash-chained append-only JSONL.

Each entry stores the hash of the previous entry, so any edit, deletion, or
reordering of historical records is detectable: re-hashing the chain no longer
matches. This turns the gateway tracer's local log into an integrity-checkable
trail (a poor-man's transparency log) without any external service.

Pure and dependency-free. Timestamps are caller-supplied so the function is
deterministic and testable (pass ``ts`` from ``time.time()`` in production).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _entry_hash(prev_hash: str, seq: int, payload: dict[str, Any], ts: "float | None") -> str:
    body = json.dumps({"prev": prev_hash, "seq": seq, "ts": ts, "payload": payload},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_records(path: Path) -> "list[dict]":
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _last_record(path: Path) -> "dict | None":
    """Read only the final JSONL record (O(1) tail seek), not the whole log.

    append() only needs the previous record's seq/hash, so reading the entire
    file on every write would make logging O(n) per call. We seek from the end
    and scan back to the last newline.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        block = min(size, 65536)
        fh.seek(size - block)
        tail = fh.read(block).rstrip(b"\n")
        nl = tail.rfind(b"\n")
        last = tail[nl + 1:] if nl != -1 else tail
    last = last.strip()
    return json.loads(last.decode("utf-8")) if last else None


def append(path: "str | Path", payload: dict[str, Any], *, ts: "float | None" = None) -> dict:
    """Append ``payload`` as the next hash-chained record. Returns the record.

    O(1) per call: only the tail record is read to chain from (not the whole log).
    """
    path = Path(path)
    prev_rec = _last_record(path)
    seq = (prev_rec["seq"] + 1) if prev_rec else 0
    prev = prev_rec["hash"] if prev_rec else GENESIS
    rec = {"seq": seq, "ts": ts, "prev": prev, "payload": payload,
           "hash": _entry_hash(prev, seq, payload, ts)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    return rec


def verify_chain(path: "str | Path") -> dict:
    """Recompute the chain and report integrity.

    Returns ``{ok, length, broken_at, reason}``. ``broken_at`` is the seq of the
    first record whose stored hash or prev-link is inconsistent (None if intact).
    """
    records = _read_records(Path(path))
    prev = GENESIS
    for i, rec in enumerate(records):
        if rec.get("seq") != i:
            return {"ok": False, "length": len(records), "broken_at": i, "reason": "seq out of order"}
        if rec.get("prev") != prev:
            return {"ok": False, "length": len(records), "broken_at": i, "reason": "prev-hash mismatch"}
        expect = _entry_hash(prev, i, rec.get("payload", {}), rec.get("ts"))
        if rec.get("hash") != expect:
            return {"ok": False, "length": len(records), "broken_at": i, "reason": "content hash mismatch"}
        prev = rec["hash"]
    return {"ok": True, "length": len(records), "broken_at": None, "reason": "chain intact"}


__all__ = ["append", "verify_chain", "GENESIS"]
