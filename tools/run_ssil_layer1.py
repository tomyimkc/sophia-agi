#!/usr/bin/env python3
"""Run the Layer-1 demo: a weight delta (LoRA/RLVR adapter) through the SAME gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_layer1 import demo_layer1_report  # noqa: E402

OUT = ROOT / "agi-proof" / "self-extension" / "ssil-layer1.public-report.json"


def main() -> int:
    rep = demo_layer1_report()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = all(rep["invariants"].values())
    print(f"layer1 invariants: {'PASS' if ok else 'FAIL'} -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
