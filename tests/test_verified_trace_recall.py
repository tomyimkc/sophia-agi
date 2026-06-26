#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the verified-trace contradiction-recall experiment.

CI-locks the four falsifiable invariants that the experiment must uphold. If any
breaks, the logger has a real defect — this test exists to fail loudly in that
case rather than let a regression ship silently.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_recall_experiment_confirms_invariants() -> None:
    """Run the experiment at small scale and assert all four invariants hold.

    The experiment compares the compiler's planted-ground-truth recall against
    the trace log's recall; on synthetic graphs both must be 1.0, fact+logic
    must agree on every step, the hash chain must survive, and every record must
    carry the no-overclaim triad.
    """
    from tools.run_verified_trace_recall import run

    # small scale (CI-fast): graphs=40, contradiction_frac=0.5 -> 20 planted.
    report = run(graphs=40, seed=2026, contradiction_frac=0.5, out=None)

    assert report["schema"] == "sophia.verified_trace_recall.v1"
    # honesty fields
    assert report["candidateOnly"] is True
    assert report["validated"] is False
    assert "not proof of AGI" in report["boundary"]

    inv = report["invariants"]
    assert inv["traceMatchesCompilerRecall"] is True, inv
    assert inv["recallIsPerfect"] is True, inv
    assert inv["chainIntactAcrossFullRun"] is True, inv
    assert inv["noOverclaimTriadOnEveryRecord"] is True, inv

    # the headline numbers themselves
    assert report["verdict"] == "CONFIRMED", report["verdict"]
    assert report["trace"]["contradictionRecall"] == 1.0
    # half the graphs have planted contradictions -> verified rate must be ~0.5
    assert report["trace"]["stepVerifiedRate"] == 0.5
    # fact and logic agree on every step (no stamp divergence)
    assert report["trace"]["factLogicAgreement"] == 1.0


def test_experiment_is_deterministic_under_seed() -> None:
    """Same seed -> identical compiler AND trace numbers (reproducibility)."""
    from tools.run_verified_trace_recall import run

    r1 = run(graphs=20, seed=7, out=None)
    r2 = run(graphs=20, seed=7, out=None)
    assert r1["compiler"] == r2["compiler"]
    assert r1["trace"]["contradictionRecall"] == r2["trace"]["contradictionRecall"]
    assert r1["trace"]["stepVerifiedRate"] == r2["trace"]["stepVerifiedRate"]


def test_experiment_writes_artifact_with_triad(tmp_path=None) -> None:
    """The written report carries the no-overclaim triad and is valid JSON."""
    import json
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp()) / "report.json"
    from tools.run_verified_trace_recall import run
    out = tmp_path if isinstance(tmp_path, Path) else Path(tmp_path)
    report = run(graphs=10, seed=2026, out=out)
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == report
    assert on_disk["candidateOnly"] is True
    assert on_disk["level3Evidence"] is False


def main() -> int:
    test_recall_experiment_confirms_invariants()
    print(f"ok {test_recall_experiment_confirms_invariants.__name__}")
    test_experiment_is_deterministic_under_seed()
    print(f"ok {test_experiment_is_deterministic_under_seed.__name__}")
    test_experiment_writes_artifact_with_triad()
    print(f"ok {test_experiment_writes_artifact_with_triad.__name__}")
    print("PASS verified-trace recall tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
