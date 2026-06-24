#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Export Sophia claim lineage to W3C PROV (Stage B).

Reads claims (from a contract store directory, a claims.jsonl file, or stdin) and
emits a PROV-JSON or PROV-N document. Dependency-free, offline, deterministic.

    # From a live contract store directory
    python tools/export_prov.py --store-dir .sophia_store --format json

    # From an explicit claims.jsonl
    python tools/export_prov.py --claims path/to/claims.jsonl --format provn

    # Self-test on a built-in fixture (no external state needed; used by CI)
    python tools/export_prov.py --demo --check

Exit code is non-zero if the produced document fails structural PROV validation,
so CI gates the export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prov import claims_to_prov, to_prov_json, to_prov_n, validate_prov  # noqa: E402


def _read_jsonl(path: Path) -> "list[dict]":
    rows: list = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({k: v for k, v in row.items() if not str(k).startswith("_")})
    return rows


def _demo_claims() -> "list[dict]":
    return [
        {"claim_id": "c1", "content": "Laozi is the attributed author of the Dao De Jing.",
         "sources": ["wikidata"], "parents": [], "blp_level": "UNCLASSIFIED",
         "created_at": "2026-06-23T00:00:00Z", "signature": "sig_demo1"},
        {"claim_id": "c2", "content": "The Analects was compiled by Confucius's disciples.",
         "sources": ["wikidata", "dispute:Analects-Compiled-Not-Autograph"],
         "parents": ["c1"], "blp_level": "CONFIDENTIAL", "created_at": "2026-06-23T00:01:00Z"},
    ]


def load_claims(args) -> "list[dict]":
    if args.demo:
        return _demo_claims()
    if args.claims:
        return _read_jsonl(Path(args.claims))
    if args.store_dir:
        from sophia_contract.service import SophiaContract
        contract = SophiaContract(store_dir=Path(args.store_dir), tracing=False)
        return contract.claims.all_claims()
    data = sys.stdin.read().strip()
    if not data:
        return []
    try:
        parsed = json.loads(data)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        return [json.loads(line) for line in data.splitlines() if line.strip()]


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--store-dir", help="contract store directory (reads claims.jsonl)")
    src.add_argument("--claims", help="explicit claims.jsonl path")
    src.add_argument("--demo", action="store_true", help="use a built-in fixture")
    ap.add_argument("--format", choices=["json", "provn"], default="json")
    ap.add_argument("--recorder", default="sophia-gateway")
    ap.add_argument("--out", help="write to this path instead of stdout")
    ap.add_argument("--check", action="store_true",
                    help="validate the document structurally; non-zero exit on error")
    args = ap.parse_args(argv)

    claims = load_claims(args)
    doc = claims_to_prov(claims, recorder=args.recorder)
    errors = validate_prov(doc)

    if args.format == "json":
        text = to_prov_json(claims, recorder=args.recorder)
    else:
        text = to_prov_n(claims, recorder=args.recorder)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if errors:
        sys.stderr.write("PROV validation errors:\n" + "\n".join(f"  - {e}" for e in errors) + "\n")
        return 1
    if args.check:
        sys.stderr.write(f"PROV OK: {len(claims)} claim(s), {len(doc.get('entity', {}))} entit(y/ies)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
