#!/usr/bin/env python3
"""Tests for tools/run_legal_citation_bench.py and its published numbers.

Locks the verifier's benchmark result and guards against drift between the live
runner and the curated published-results.json (RESULTS.md is generated from it).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_runner():
    spec = importlib.util.spec_from_file_location("rlcb", ROOT / "tools" / "run_legal_citation_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_clean_sweep() -> None:
    result = _load_runner().run()
    bench = json.loads((ROOT / "benchmark" / "legal_citations.json").read_text("utf-8"))
    assert result["n"] == len(bench["cases"]) >= 14  # federated HK/UK/US cases
    # every fabrication flagged, no false alarms
    assert result["confusion"]["fp"] == 0
    assert result["confusion"]["fn"] == 0
    assert result["accuracy"] == 1.0
    assert result["fabricationDetectionRecall"] == 1.0
    assert result["falseAlarmRate"] == 0.0


def test_published_results_match_runner() -> None:
    result = _load_runner().run()
    pub = json.loads((ROOT / "agi-proof" / "benchmark-results" / "published-results.json").read_text("utf-8"))
    row = next(e for e in pub["verifierEvals"] if e["verifier"] == "legal_citation_exists")
    assert row["n"] == result["n"]
    assert row["accuracy"] == result["accuracy"]
    assert row["confusion"] == result["confusion"]


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_legal_citation_bench: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
