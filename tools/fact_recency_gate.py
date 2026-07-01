#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fact-recency gate — a per-domain STALENESS CLOCK the other lenses missed.

The knowledge base can be internally coherent (links resolve, no contradictions)
and still be WRONG because a load-bearing empirical fact has silently gone out of
date. Coherence gates are timeless; the world is not. This gate audits a
``verifiedAsOf`` date on fact records against a per-domain staleness horizon and
ALARMS when too large a fraction of the *load-bearing* records are past horizon.

Design choices (deliberate, to stay honest):

  * ``today`` is a REQUIRED CLI argument (``--today YYYY-MM-DD``). Pure code must
    not read the wall clock — that makes the audit non-reproducible and lets a
    result drift silently between runs.
  * Records LACKING a ``verifiedAsOf`` are counted as ``unknown``, NEVER as fresh.
    A corpus must not be able to dodge staleness by simply never stamping a date;
    high unknown coverage trips a SEPARATE warning.
  * Horizons come from ``agi-proof/recency/staleness_horizons.json``. A ``null``
    horizon means the domain is timeless (text/historical record, not a moving
    front) and its records are never stale by clock alone.
  * PROTECTED domains (history, religion) are timeless here; recency NEVER licenses
    re-attribution or cross-domain merges — this gate only reads dates.

Input: a records JSON — either a list of records or ``{"records": [...]}`` — where
each record is ``{"id", "domain", "verifiedAsOf"?, "loadBearing"?}``. ``domain`` is
matched against the horizons table; unknown domains fall back to ``defaultHorizonDays``.
``loadBearing`` defaults to True (a record is assumed to matter unless flagged
otherwise) so the gate is conservative.

    python3 tools/fact_recency_gate.py --records recs.json --today 2026-07-01
    python3 tools/fact_recency_gate.py --records recs.json --today 2026-07-01 --json

Exit 0 = under alarm fraction (PASS). Exit 1 = stale fraction over threshold (ALARM).
Exit 2 = unreadable/missing input. JSON receipt to stdout; human prose to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HORIZONS = ROOT / "agi-proof" / "recency" / "staleness_horizons.json"


def _parse_date(s: str) -> date:
    """Parse a strict ``YYYY-MM-DD`` date; raise ValueError otherwise."""
    return date.fromisoformat(str(s).strip())


def load_horizons(path: Path) -> dict:
    """Load the staleness-horizons config."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records_of(data: object) -> list[dict]:
    """Normalize input into a list of record dicts."""
    if isinstance(data, dict) and "records" in data:
        recs = data["records"]
    else:
        recs = data
    if not isinstance(recs, list):
        raise ValueError("records input must be a list or {'records': [...]}")
    out: list[dict] = []
    for r in recs:
        if not isinstance(r, dict):
            raise ValueError("each record must be an object")
        out.append(r)
    return out


def audit(records: list[dict], today: date, horizons_cfg: dict) -> dict:
    """Classify records as fresh / stale / unknown against per-domain horizons.

    Only LOAD-BEARING records feed the stale-fraction that trips the alarm.
    A record with no ``verifiedAsOf`` is ``unknown`` (never counted fresh).
    A domain horizon of ``null`` means timeless (never stale by clock).
    """
    horizons = horizons_cfg.get("horizons", {})
    default_h = horizons_cfg.get("defaultHorizonDays", 365)
    alarm_frac = float(horizons_cfg.get("alarmFractionThreshold", 0.10))

    fresh: list[str] = []
    stale: list[dict] = []
    unknown: list[str] = []
    timeless: list[str] = []
    per_domain: dict[str, dict] = {}

    load_bearing_total = 0
    load_bearing_stale = 0

    for r in records:
        rid = str(r.get("id", "?"))
        domain = r.get("domain")
        load_bearing = bool(r.get("loadBearing", True))
        # horizon: explicit domain entry wins; missing domain -> default.
        if domain in horizons:
            horizon = horizons[domain]
        else:
            horizon = default_h
        d = per_domain.setdefault(
            str(domain), {"fresh": 0, "stale": 0, "unknown": 0, "timeless": 0}
        )

        va = r.get("verifiedAsOf")
        if va in (None, "", "unknown"):
            unknown.append(rid)
            d["unknown"] += 1
            continue

        if horizon is None:
            # Timeless domain: never stale by clock alone.
            timeless.append(rid)
            d["timeless"] += 1
            if load_bearing:
                load_bearing_total += 1
            continue

        try:
            vd = _parse_date(va)
        except (ValueError, TypeError):
            # Malformed date is treated as unknown (never fresh).
            unknown.append(rid)
            d["unknown"] += 1
            continue

        age_days = (today - vd).days
        if load_bearing:
            load_bearing_total += 1
        if age_days > int(horizon):
            stale.append({"id": rid, "domain": str(domain), "ageDays": age_days,
                          "horizonDays": int(horizon), "loadBearing": load_bearing})
            d["stale"] += 1
            if load_bearing:
                load_bearing_stale += 1
        else:
            fresh.append(rid)
            d["fresh"] += 1

    stale_fraction = (load_bearing_stale / load_bearing_total) if load_bearing_total else 0.0
    total = len(records)
    unknown_fraction = (len(unknown) / total) if total else 0.0

    alarm = stale_fraction > alarm_frac
    # Coverage warning: too many records have no date to audit at all.
    coverage_warn = unknown_fraction > 0.5

    return {
        "gate": "fact_recency_gate",
        "status": "preregistration_only",
        "canClaimAGI": False,
        "today": today.isoformat(),
        "totalRecords": total,
        "loadBearingRecords": load_bearing_total,
        "counts": {
            "fresh": len(fresh),
            "stale": len(stale),
            "unknown": len(unknown),
            "timeless": len(timeless),
        },
        "loadBearingStale": load_bearing_stale,
        "staleFraction": round(stale_fraction, 6),
        "alarmFractionThreshold": alarm_frac,
        "unknownFraction": round(unknown_fraction, 6),
        "coverageWarning": coverage_warn,
        "perDomain": per_domain,
        "staleRecords": stale,
        "alarm": alarm,
        "go": (not alarm),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-domain fact-recency staleness gate.")
    ap.add_argument("--records", required=True, help="path to records JSON")
    ap.add_argument("--today", required=True, help="reference date YYYY-MM-DD (no wallclock in pure code)")
    ap.add_argument("--horizons", default=str(DEFAULT_HORIZONS),
                    help="path to staleness_horizons.json")
    ap.add_argument("--json", action="store_true", help="print only the JSON receipt")
    args = ap.parse_args(argv)

    try:
        today = _parse_date(args.today)
    except (ValueError, TypeError):
        print(f"[fact_recency_gate] --today must be YYYY-MM-DD, got {args.today!r}", file=sys.stderr)
        return 2

    try:
        horizons_cfg = load_horizons(Path(args.horizons))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[fact_recency_gate] cannot read horizons: {e}", file=sys.stderr)
        return 2

    try:
        raw = json.loads(Path(args.records).read_text(encoding="utf-8"))
        records = _records_of(raw)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[fact_recency_gate] cannot read records: {e}", file=sys.stderr)
        return 2

    receipt = audit(records, today, horizons_cfg)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))

    if not args.json:
        verdict = "ALARM (stale fraction over threshold)" if receipt["alarm"] else "PASS"
        print(
            f"[fact_recency_gate] {verdict}: "
            f"{receipt['loadBearingStale']}/{receipt['loadBearingRecords']} load-bearing stale "
            f"(frac {receipt['staleFraction']:.3f} vs {receipt['alarmFractionThreshold']}); "
            f"{receipt['counts']['unknown']} unknown"
            + ("; COVERAGE WARNING (majority undated)" if receipt["coverageWarning"] else ""),
            file=sys.stderr,
        )
    return 1 if receipt["alarm"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
