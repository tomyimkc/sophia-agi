#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Scan a legal document (or raw text) for fabricated citations — build-item 4.

Extracts text from a contract / filing / memo (TXT/MD, DOCX, HTML; PDF if a parser
is installed) and runs the fail-closed citation verifier over it. Every cited
authority is checked against the trusted register (and, with --live, the federated
HKLII / e-Legislation / National Archives / CourtListener resolver). Exits non-zero
if any citation is unverified — usable as a pre-filing gate.

    python tools/scan_legal_document.py path/to/brief.docx
    python tools/scan_legal_document.py --text "See Varghese v. China Southern, 925 F.3d 1339."
    SOPHIA_LEGAL_SOURCE=live python tools/scan_legal_document.py brief.pdf --live --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.legal_docs import DocIngestError, scan_document, scan_text  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("path", nargs="?", help="document to scan (.txt/.md/.docx/.html/.pdf)")
    src.add_argument("--text", help="scan a raw text string instead of a file")
    ap.add_argument("--live", action="store_true", help="also use the live federated resolver")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    resolver = None
    if args.live:
        from agent.legal_sources import make_resolver
        resolver = make_resolver(mode="live")

    try:
        if args.text is not None:
            report = scan_text(args.text, resolver=resolver)
            report["source"] = "(--text)"
        else:
            report = scan_document(args.path, resolver=resolver)
    except DocIngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"source: {report['source']}")
        print(f"citations found: {report['citationsFound']}  |  "
              f"verified: {len(report['verified'])}  |  unverified: {len(report['unverified'])}")
        for c in report["verified"]:
            print(f"  ✓ {c}")
        for u in report["unverified"]:
            print(f"  ✗ {u['citation']}  ({u['status']})")
        print("PASS — every citation verified" if report["passed"]
              else "FAIL — unverified citation(s); verify against an official source before filing")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
