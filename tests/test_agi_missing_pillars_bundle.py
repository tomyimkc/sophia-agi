#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_agi_missing_pillars import build_report  # noqa: E402


def test_missing_pillars_bundle_ok() -> None:
    rep = build_report(seed=0)
    assert rep["ok"] is True
    assert rep["candidateOnly"] is True and rep["level3Evidence"] is False
    assert "does not prove AGI" in rep["claimBoundary"]


def main() -> int:
    test_missing_pillars_bundle_ok()
    print("test_agi_missing_pillars_bundle: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
