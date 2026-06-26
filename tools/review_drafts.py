#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Review queue for draft pages — the human sign-off step of the self-correction loop.

Lists `needsReview` drafts (gap stubs, librarian-filled pages) with their provenance and live
gate status, and promotes an approved one into the consolidated memory tier or rejects it.
Promotion is human-gated (an ``--approver`` is required) and re-gated (fail-closed).

  python tools/review_drafts.py                              # list pending reviews
  python tools/review_drafts.py --promote ID --approver NAME  # elevate into memory tier
  python tools/review_drafts.py --reject ID --approver NAME --reason "thin source"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from agent.canon_promote import pending_reviews, promote, reject

    ap = argparse.ArgumentParser(description="Review + promote/reject draft pages")
    ap.add_argument("--promote", metavar="ID", help="promote a draft into the memory tier")
    ap.add_argument("--reject", metavar="ID", help="reject (remove) a draft")
    ap.add_argument("--approver", help="who is signing off (required for promote/reject)")
    ap.add_argument("--reason", default="", help="reason (for --reject)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.promote:
        result = promote(args.promote, approver=args.approver or "")
    elif args.reject:
        result = reject(args.reject, approver=args.approver or "", reason=args.reason)
    else:
        pend = pending_reviews()
        if args.json:
            print(json.dumps(pend, indent=2, ensure_ascii=False))
            return 0
        if not pend:
            print("No drafts awaiting review.")
            return 0
        print(f"{len(pend)} draft(s) awaiting review:")
        for r in pend:
            flag = "OK " if r["gatePasses"] else "GATE✗"
            src = ", ".join(r["sources"]) or "(no source)"
            print(f"  [{flag}] {r['id']}  conf={r['authorConfidence']}  author={r['attributedAuthor']}  src={src}")
            if not r["gatePasses"]:
                print(f"           gate reasons: {r['gateReasons']}")
        print("\nPromote:  python tools/review_drafts.py --promote ID --approver YOU")
        return 0

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
