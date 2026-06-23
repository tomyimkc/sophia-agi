#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.conscience import conscience_check, write_conscience_report  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "conscience" / "conscience.public-report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Sophia conscience check or benchmark")
    ap.add_argument("text", nargs="?", help="Text to check. Omit to run benchmark.")
    ap.add_argument("--mode", default="output", choices=["output", "tool", "memory"])
    ap.add_argument("--action", default=None)
    ap.add_argument("--context-json", default="{}")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    if args.text:
        try:
            context = json.loads(args.context_json or "{}")
        except json.JSONDecodeError as exc:
            print(json.dumps({"error": f"invalid context JSON: {exc}"}))
            return 2
        print(json.dumps(conscience_check(args.text, mode=args.mode, action=args.action, context=context).to_dict(), indent=2, ensure_ascii=False))
        return 0
    report = write_conscience_report(args.out)
    print(json.dumps({"ok": report["ok"], "out": args.out, "accuracy": report["accuracy"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
