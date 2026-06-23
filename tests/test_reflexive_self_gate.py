#!/usr/bin/env python3
"""Tests for Sophia's reflexive no-overclaim self-gate."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.reflexive_self_gate import classify_self_claim, scan_paths  # noqa: E402
from tools.run_reflexive_self_gate import main as cli_main  # noqa: E402


def test_classify_self_claim_accepts_candidate_boundary() -> None:
    c = classify_self_claim("README.md", 1, "Sophia is an AGI-candidate proof package, not proven AGI.")
    assert c.verdict == "accepted"


def test_classify_self_claim_rejects_proven_agi() -> None:
    c = classify_self_claim("README.md", 1, "Sophia is proven AGI.")
    assert c.verdict == "rejected"


def test_scan_paths_rejects_manifest_can_claim_true() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "report.json").write_text(json.dumps({"canClaimAGI": True}), encoding="utf-8")
        report = scan_paths(["report.json"], repo_root=root)
        assert report["verdict"] == "rejected"


def test_cli_runs_on_default_repo_paths() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "self.json"
        rc = cli_main(["README.md", "RESULTS.md", "agi-proof/agi-verification", "--out", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["canClaimAGI"] is False
        assert data["summary"]["rejected"] == 0


def main() -> int:
    test_classify_self_claim_accepts_candidate_boundary()
    test_classify_self_claim_rejects_proven_agi()
    test_scan_paths_rejects_manifest_can_claim_true()
    test_cli_runs_on_default_repo_paths()
    print("test_reflexive_self_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
