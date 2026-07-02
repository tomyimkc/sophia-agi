# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Machine-enforced resource claims — TTL'd, heartbeat-renewed locks for shared
cluster resources (the one-GPU-job invariant as a check, not prose).

Today the 5+ concurrent sessions on this repo coordinate GPU/branch usage through
``SESSION-COORDINATION.md`` and skill prose — locks that nothing enforces. This
module makes a claim machine-readable: a launcher consults :func:`claim` before a
GPU job; a holder renews with :func:`heartbeat`; a crashed session's claim expires
by TTL instead of wedging the cluster.

Sophia discipline:

  * **Fail-closed.** An unreadable/corrupt claims file REFUSES new claims (it never
    assumes "free"); an expired claim is released only by TTL math, never by a
    non-holder's say-so; ``release`` by a non-holder is refused.
  * **Deterministic + offline + stdlib-only.** ``now`` is injectable so every branch
    is testable without wall-clock flakes. Writes are atomic (temp file +
    ``os.replace``) so a reader never sees a torn file.
  * **Advisory perimeter, honest bound.** This is cooperative locking on a shared
    file: it stops *coordinated* sessions from colliding, which is the actual
    failure mode here. It is not a distributed-consensus lock; a hostile or
    non-participating process can still ignore it. GOAP plans encode a claim as a
    precondition atom (``resource:<name>:free``) so violating plans are unreachable
    in search (see :mod:`agent.goap_planner`).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sophia.resource_claims.v1"
CLAIMS_PATH = ROOT / "agent" / "memory" / "resource_claims.json"

#: Known shared resources (open set — any name is claimable; these are the ones
#: the cluster ops skill names). Kept here so tooling can enumerate them.
KNOWN_RESOURCES = ("spark-gpu", "mac-mlx", "runpod-paid", "branch:main-merge")

DEFAULT_TTL_S = 2 * 60 * 60  # 2h — matches a long cert/train leg; heartbeat extends


def _load(path: Path) -> "dict | None":
    """Parse the claims file. Missing file → empty store; unparseable → None
    (callers must fail closed on None, never assume free)."""
    if not path.exists():
        return {"schema": SCHEMA, "claims": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("claims"), dict):
        return None
    return data


def _store(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _expired(entry: dict, now: float) -> bool:
    try:
        return now >= float(entry["ts"]) + float(entry["ttlS"])
    except (KeyError, TypeError, ValueError):
        return True  # malformed entry cannot hold a resource


def claim(resource: str, holder: str, *, ttl_s: float = DEFAULT_TTL_S,
          note: str = "", path: "Path | None" = None, now: "float | None" = None) -> dict:
    """Claim ``resource`` for ``holder``. Succeeds when the resource is free, its
    prior claim expired, or ``holder`` already holds it (re-claim = renew)."""
    if not (resource or "").strip() or not (holder or "").strip():
        return {"ok": False, "reason": "resource and holder are required"}
    path = CLAIMS_PATH if path is None else path  # call-time resolve (monkeypatch/deploy safe)
    now = time.time() if now is None else now
    data = _load(path)
    if data is None:
        return {"ok": False, "reason": "claims file unreadable — fix or remove it; "
                                       "refusing to assume the resource is free"}
    entry = data["claims"].get(resource)
    if entry and not _expired(entry, now) and entry.get("holder") != holder:
        return {"ok": False, "reason": "held by another session",
                "holder": entry.get("holder"), "note": entry.get("note", ""),
                "expiresInS": round(float(entry["ts"]) + float(entry["ttlS"]) - now, 1)}
    data["claims"][resource] = {"holder": holder, "ts": now, "ttlS": float(ttl_s),
                                "note": note}
    _store(path, data)
    return {"ok": True, "resource": resource, "holder": holder, "ttlS": float(ttl_s)}


def heartbeat(resource: str, holder: str, *, path: "Path | None" = None,
              now: "float | None" = None) -> dict:
    """Renew a live claim. Refused when the claim is missing, expired, or held by
    someone else (an expired claim must be re-claimed, not silently revived)."""
    path = CLAIMS_PATH if path is None else path  # call-time resolve (monkeypatch/deploy safe)
    now = time.time() if now is None else now
    data = _load(path)
    if data is None:
        return {"ok": False, "reason": "claims file unreadable"}
    entry = data["claims"].get(resource)
    if not entry or entry.get("holder") != holder:
        return {"ok": False, "reason": "no live claim by this holder"}
    if _expired(entry, now):
        return {"ok": False, "reason": "claim expired — re-claim explicitly"}
    entry["ts"] = now
    _store(path, data)
    return {"ok": True, "resource": resource, "holder": holder}


def release(resource: str, holder: str, *, path: "Path | None" = None,
            now: "float | None" = None) -> dict:
    """Release a claim. Only the holder may release a live claim (fail-closed);
    releasing an already-expired or absent claim is a no-op success."""
    path = CLAIMS_PATH if path is None else path  # call-time resolve (monkeypatch/deploy safe)
    now = time.time() if now is None else now
    data = _load(path)
    if data is None:
        return {"ok": False, "reason": "claims file unreadable"}
    entry = data["claims"].get(resource)
    if not entry or _expired(entry, now):
        data["claims"].pop(resource, None)
        _store(path, data)
        return {"ok": True, "resource": resource, "released": "absent-or-expired"}
    if entry.get("holder") != holder:
        return {"ok": False, "reason": "held by another session", "holder": entry.get("holder")}
    del data["claims"][resource]
    _store(path, data)
    return {"ok": True, "resource": resource, "released": "live"}


def status(*, path: "Path | None" = None, now: "float | None" = None) -> dict:
    """Read-only view: live claims with remaining TTL, expired ones flagged."""
    path = CLAIMS_PATH if path is None else path  # call-time resolve (monkeypatch/deploy safe)
    now = time.time() if now is None else now
    data = _load(path)
    if data is None:
        return {"ok": False, "reason": "claims file unreadable"}
    out = {}
    for res, entry in sorted(data["claims"].items()):
        expired = _expired(entry, now)
        out[res] = {"holder": entry.get("holder"), "note": entry.get("note", ""),
                    "expired": expired,
                    "expiresInS": (None if expired else
                                   round(float(entry["ts"]) + float(entry["ttlS"]) - now, 1))}
    return {"ok": True, "schema": SCHEMA, "claims": out,
            "knownResources": list(KNOWN_RESOURCES)}


def is_free(resource: str, *, path: "Path | None" = None, now: "float | None" = None) -> bool:
    """Launcher-side check. Fail-closed: unreadable store → NOT free."""
    path = CLAIMS_PATH if path is None else path  # call-time resolve (monkeypatch/deploy safe)
    now = time.time() if now is None else now
    data = _load(path)
    if data is None:
        return False
    entry = data["claims"].get(resource)
    return entry is None or _expired(entry, now)


# --------------------------------------------------------------------------- #
# Offline invariants (CI-gated; injectable clock, temp store)
# --------------------------------------------------------------------------- #

def offline_invariants() -> "tuple[bool, dict]":
    import tempfile

    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "claims.json"
        t0 = 1_000_000.0

        # 1) Free resource claims; second session is refused while live.
        checks["claim_free"] = claim("spark-gpu", "sessA", ttl_s=100, path=p, now=t0)["ok"]
        r = claim("spark-gpu", "sessB", path=p, now=t0 + 10)
        checks["second_claim_refused"] = (not r["ok"]) and r["holder"] == "sessA"

        # 2) Holder re-claim renews; heartbeat extends; non-holder heartbeat refused.
        checks["reclaim_renews"] = claim("spark-gpu", "sessA", ttl_s=100, path=p, now=t0 + 50)["ok"]
        checks["heartbeat_ok"] = heartbeat("spark-gpu", "sessA", path=p, now=t0 + 60)["ok"]
        checks["foreign_heartbeat_refused"] = not heartbeat("spark-gpu", "sessB", path=p, now=t0 + 60)["ok"]

        # 3) TTL expiry frees the resource for a new holder; expired heartbeat refused.
        checks["expired_heartbeat_refused"] = not heartbeat("spark-gpu", "sessA", path=p, now=t0 + 500)["ok"]
        checks["expired_claimable"] = claim("spark-gpu", "sessB", ttl_s=100, path=p, now=t0 + 500)["ok"]

        # 4) Non-holder release refused on a live claim; holder release works.
        checks["foreign_release_refused"] = not release("spark-gpu", "sessA", path=p, now=t0 + 510)["ok"]
        checks["holder_release_ok"] = release("spark-gpu", "sessB", path=p, now=t0 + 510)["ok"]
        checks["freed_after_release"] = is_free("spark-gpu", path=p, now=t0 + 511)

        # 5) Corrupt store fails closed everywhere (never assumed free).
        p.write_text("{not json", encoding="utf-8")
        checks["corrupt_claim_refused"] = not claim("spark-gpu", "sessA", path=p, now=t0)["ok"]
        checks["corrupt_not_free"] = not is_free("spark-gpu", path=p, now=t0)
        checks["corrupt_status_flagged"] = not status(path=p, now=t0)["ok"]

    ok = all(checks.values())
    return ok, {"checks": checks}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    for name, passed in detail["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"{'PASS' if ok else 'FAIL'} resource_claims offline_invariants")
    raise SystemExit(0 if ok else 1)
