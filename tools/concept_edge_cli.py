#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Concept-edge checker CLI: classify a structured ontology edge OR check free text.

Two modes:
  - ``--edge '<json>'`` (or edge JSON on stdin): classify a structured
    ``ontology_edge`` with the symbolic Datalog gate -> ``{verdict, edgeId, detail}``
    (verdict ∈ admit | abstain | violation).
  - ``--text '<str>'``: run the surface concept gate over prose ->
    ``{passed, reasons, violations}`` (unscoped cross-tradition identity fails).

Dependency-free, offline. Exit ``0`` on a completed check (the verdict is in the
JSON); nonzero only on an internal/usage error so a caller can fail open.
See docs/11-Platform/Ontology-Claim-Boundary.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_text(text: str) -> dict:
    from agent.guarded import check_claim

    return check_claim(text or "")


def check_edge(edge: dict) -> dict:
    from agent.datalog_ontology import check_edge as _check_edge

    return _check_edge(edge)


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify a concept edge or check text.")
    parser.add_argument("--text", help="free text to check with the surface concept gate")
    parser.add_argument("--edge", help="structured ontology_edge JSON to classify (default: read stdin as edge JSON)")
    args = parser.parse_args()

    if args.text is not None:
        print(json.dumps(check_text(args.text), ensure_ascii=False))
        return 0

    raw = args.edge if args.edge is not None else sys.stdin.read()
    try:
        edge = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid edge JSON: {exc}"}, ensure_ascii=False))
        return 2
    if not isinstance(edge, dict) or not (edge.get("subject") and edge.get("object") and edge.get("edgeType")):
        print(json.dumps({"error": "edge requires subject, object, edgeType"}, ensure_ascii=False))
        return 2
    print(json.dumps(check_edge(edge), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
