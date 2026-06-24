#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia online RAG CLI (curated index + Gemini/Claude + gate).

Usage:
  python tools/build_rag_index.py
  python tools/sophia_rag.py "Did Confucius write the Dao De Jing?"
  python tools/sophia_rag.py "Did Confucius write the Dao De Jing?" --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rag_pipeline import answer_question  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sophia curated online RAG")
    parser.add_argument("question", help="User question")
    parser.add_argument("--mode", default="advisor", choices=["advisor", "repo", "life"])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = answer_question(args.question, mode=args.mode, top_k=args.top_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("[Sources]")
    for src in result["sources"][:5]:
        print(f"  - {src['path']} ({src['score']:.2f})")
    print("\n" + "=" * 60 + "\n")
    print(result["answer"])
    print("\n" + "=" * 60)
    gate = result["gate"]
    print(f"[Gate] passed={gate['passed']} violations={gate.get('violations', [])}")
    return 0 if gate.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())