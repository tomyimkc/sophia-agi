#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Counterfactual + retraction probes over the OKF belief graph.

Answers the Sophia question "what would I conclude if this source were removed?"
against the live wiki (or any roots you pass), and runs a named, auditable
retraction. Non-destructive — it reports impact, it does not delete pages.

    # what loses support if this source is struck out?
    python tools/belief_counterfactual.py remove primary_source
    python tools/belief_counterfactual.py remove legend --query secondary --json

    # a named retraction with a reason (prints an audit entry)
    python tools/belief_counterfactual.py retract legend --reason "shown to be forged"

    # point at specific roots instead of the default wiki dirs
    python tools/belief_counterfactual.py remove X --root wiki --root docs/04-Disputes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from okf.counterfactual import counterfactual_remove, retract  # noqa: E402

DEFAULT_ROOTS = [ROOT / "wiki", ROOT / "docs" / "04-Disputes"]


def _build_graph(roots):
    paths = [Path(r) for r in roots] if roots else DEFAULT_ROOTS
    pages = okf.load_pages(*[p for p in paths if Path(p).exists()])
    return okf.build_graph(pages), len(pages)


def _print_remove(cf: dict) -> None:
    if not cf["found"]:
        print(f"source not found: {cf['source']}")
        return
    print(f"counterfactual: remove [{cf['id']}]")
    print(f"  affected: {cf['affectedCount']} · support lost: {cf['supportLostCount']}")
    for r in cf["affected"]:
        flag = "LOST SUPPORT" if r["supportLost"] else "rank "
        rank = "" if r["supportLost"] else f"{r['confidenceRankBefore']}->{r['confidenceRankAfter']}"
        print(f"  - {r['page']}: {flag}{rank}")
    if "query" in cf:
        q = cf["query"]
        print(f"  query[{q['entity']}]: grounded {q['groundedBefore']}->{q['groundedAfter']}"
              f" · effective {q.get('effectiveBefore')}->{q.get('effectiveAfter')}")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["remove", "retract", "revise"])
    ap.add_argument("source", help="page id, alias, or [[wikilink]] target")
    ap.add_argument("--also", action="append", dest="also", default=[],
                    help="(revise) additional target(s) to retract together (repeatable)")
    ap.add_argument("--query", help="(remove) isolate the before/after belief for this entity")
    ap.add_argument("--reason", default="(unspecified)", help="(retract/revise) why the claim(s) are retracted")
    ap.add_argument("--by", default="cli", help="(retract/revise) actor recorded in the audit entry")
    ap.add_argument("--root", action="append", dest="roots", help="page root (repeatable); default: wiki + disputes")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    graph, n = _build_graph(args.roots)

    if args.action == "revise":
        from okf.revision import revise as _revise
        rev = _revise(graph, [(t, args.reason) for t in [args.source, *args.also]], by=args.by)
        if args.json:
            print(json.dumps(rev.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"(loaded {n} pages)")
            print(f"revision: retracted {rev.retracted or '(none)'}"
                  + (f" · not found {rev.notFound}" if rev.notFound else ""))
            print(f"  cascade (lost support): {[c['page'] for c in rev.cascade] or '(none)'}")
            print(f"  abstain set: {rev.abstain}")
        return 0

    if args.action == "remove":
        cf = counterfactual_remove(graph, args.source, query=args.query)
        if args.json:
            print(json.dumps(cf, ensure_ascii=False, indent=2))
        else:
            print(f"(loaded {n} pages)")
            _print_remove(cf)
        return 0

    r = retract(graph, args.source, reason=args.reason, by=args.by)
    if args.json:
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"(loaded {n} pages)")
        if not r.found:
            print(f"target not found: {args.source}")
            return 0
        print(f"retraction: [{r.id}] — {r.reason}")
        print(f"  downstream claims losing support: {r.downstream or '(none)'}")
        print("  audit: " + json.dumps(r.audit_entry(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
