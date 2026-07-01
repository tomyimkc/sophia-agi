#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Quality gate for realtime C1 dated-fact packs — enforce provenance + temporal cleanliness.

The Phase-1 powered pack must reach N>=393 with REAL dated sources and must NOT be padded with
synthesized/paraphrased items to inflate N (that is contamination — see the realtime-grounding ledger
row + measurement_spec). This validator is the discipline that keeps every growth batch honest: it
re-runs the SAME temporal machinery the seed test uses (`agent.streaming_decontam`), plus a
provenance requirement (every item carries a `source`) so labels are auditable, not asserted.

Run it on any pack file (or several) before adding a batch to the pack:
    python tools/validate_dated_pack.py eval/fact_check/phase1_dated_batch_v2.jsonl
    python tools/validate_dated_pack.py eval/fact_check/*.jsonl --require-source

Checks (fail-closed; non-zero exit on ANY violation):
  * schema: id/claim/label/sourceTimestamp/validFrom present; label in {true,false,unknowable}
  * ids unique WITHIN the file AND across all files passed (so batches never collide)
  * `source` present on every item (provenance for the coupled-verifier audit) unless --no-require-source
  * temporal_decontam(sourceTimestamp, cutoff).ok  — source predates the eval cutoff
  * valid_time(validFrom, validUntil, as_of).ok     — the fact is valid at as_of (interpretable label)

canClaimAGI stays false; this is a data-integrity gate, not a capability claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CUTOFF = "2026-07-01"
AS_OF = "2026-07-01"
LABELS = {"true", "false", "unknowable"}
POWERED_N = 393  # eval_stats.required_n_for_mde(0.10); the pack is underpowered below this


def load_rows(path: Path) -> "list[dict]":
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def validate(paths: "list[Path]", *, require_source: bool = True,
             cutoff: str = CUTOFF, as_of: str = AS_OF) -> "tuple[bool, dict]":
    import agent.streaming_decontam as sd

    errors: list[str] = []
    all_ids: dict[str, str] = {}
    mix: Counter = Counter()
    total = 0
    for path in paths:
        for r in load_rows(path):
            total += 1
            rid = r.get("id", "")
            where = f"{path.name}:{rid or '?'}"
            if not rid:
                errors.append(f"{where}: missing id")
            elif rid in all_ids:
                errors.append(f"{where}: duplicate id (also in {all_ids[rid]})")
            else:
                all_ids[rid] = path.name
            if r.get("label") not in LABELS:
                errors.append(f"{where}: label {r.get('label')!r} not in {sorted(LABELS)}")
            else:
                mix[r["label"]] += 1
            if not str(r.get("claim", "")).strip():
                errors.append(f"{where}: empty claim")
            if not r.get("sourceTimestamp"):
                errors.append(f"{where}: missing sourceTimestamp")
            if not r.get("validFrom"):
                errors.append(f"{where}: missing validFrom")
            if require_source and not str(r.get("source", "")).strip():
                errors.append(f"{where}: missing `source` provenance (use --no-require-source to skip)")
            t = sd.temporal_decontam(r.get("sourceTimestamp", ""), cutoff)
            if not t.get("ok"):
                errors.append(f"{where}: temporal_decontam failed ({t})")
            v = sd.valid_time(r.get("validFrom", ""), r.get("validUntil", ""), as_of)
            if not v.get("ok"):
                errors.append(f"{where}: valid_time failed ({v})")
    info = {
        "total": total, "mix": dict(mix), "unique_ids": len(all_ids),
        "powered_target": POWERED_N, "underpowered": total < POWERED_N,
        "shortfall_to_powered": max(0, POWERED_N - total),
    }
    return (not errors), {"errors": errors, **info}


# --------------------------------------------------------------------------- #
# Offline invariants — GPU-free, prove the gate accepts a clean item and rejects the failure modes.
# --------------------------------------------------------------------------- #
def offline_invariants() -> "tuple[bool, dict]":
    import tempfile

    import agent.streaming_decontam as sd
    checks: dict[str, bool] = {}
    clean = {"id": "x1", "claim": "c", "label": "true", "sourceTimestamp": "2020-01-01",
             "validFrom": "0001-01-01", "source": "deterministic:test"}
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "good.jsonl"
        good.write_text("# hdr\n" + json.dumps(clean) + "\n")
        ok, _ = validate([good])
        checks["accepts_clean"] = ok
        # post-cutoff source rejected
        bad = Path(d) / "bad.jsonl"
        bad.write_text(json.dumps({**clean, "id": "x2", "sourceTimestamp": "2026-08-01"}) + "\n")
        checks["rejects_post_cutoff"] = not validate([bad])[0]
        # missing source rejected (when required), accepted when not
        nosrc = Path(d) / "nosrc.jsonl"
        nosrc.write_text(json.dumps({k: v for k, v in clean.items() if k != "source"} | {"id": "x3"}) + "\n")
        checks["rejects_missing_source"] = not validate([nosrc])[0]
        checks["allows_missing_source_when_opted_out"] = validate([nosrc], require_source=False)[0]
        # duplicate id across files rejected
        dup = Path(d) / "dup.jsonl"
        dup.write_text(json.dumps(clean) + "\n")   # same id x1
        checks["rejects_dup_id_cross_file"] = not validate([good, dup])[0]
    checks["sd_module_present"] = hasattr(sd, "temporal_decontam") and hasattr(sd, "valid_time")
    ok = all(checks.values())
    return ok, {"checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate realtime C1 dated-fact pack(s) for the powered build.")
    ap.add_argument("paths", nargs="*", help="pack .jsonl file(s)")
    ap.add_argument("--no-require-source", dest="require_source", action="store_false",
                    help="do not require the `source` provenance field (the seed v1 predates it)")
    ap.add_argument("--selftest", action="store_true", help="run offline invariants and exit")
    args = ap.parse_args()
    if args.selftest:
        ok, d = offline_invariants()
        print("validate_dated_pack offline invariants:", "PASS" if ok else "FAIL")
        for k, v in d["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        return 0 if ok else 1
    if not args.paths:
        ap.error("give at least one pack file, or --selftest")
    ok, d = validate([Path(p) for p in args.paths], require_source=args.require_source)
    print(f"DATED-PACK VALIDATE: {'OK' if ok else 'FAIL'} — {d['total']} item(s), mix={d['mix']}, "
          f"unique_ids={d['unique_ids']}")
    if d["underpowered"]:
        print(f"  [note] N={d['total']} < powered target {d['powered_target']} "
              f"(shortfall {d['shortfall_to_powered']}) — pack stays not_run/underpowered; NOT a GO.")
    for e in d["errors"]:
        print(f"  [XX] {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
