#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the verified-trace surface in the agi-proof evidence manifest.

Locks in that build_manifest reads the recall artifact HONESTLY (reports the
verdict the experiment actually concluded, never a hardcoded value), surfaces the
verified-trace artifactIndex entries, and degrades gracefully when the artifact
is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_manifest_reads_recall_verdict_from_artifact() -> None:
    from tools.build_agi_proof_package import build_manifest
    m = build_manifest(generated="test")
    vt = m["verifiedTraces"]
    # the recall artifact exists in the repo -> its verdict must propagate
    assert vt["recallVerdict"] == "CONFIRMED", vt
    assert vt["traceRecall"] == 1.0
    assert vt["compilerRecall"] == 1.0
    assert vt["factLogicAgreement"] == 1.0
    assert vt["candidateOnly"] is True
    # the honest scope string is present
    assert "not a capability claim" in vt["scope"].lower()


def test_manifest_artifact_index_has_verified_trace_entries() -> None:
    from tools.build_agi_proof_package import build_manifest
    ai = build_manifest(generated="test")["artifactIndex"]
    required = [
        "verifiedTracePackage", "verifiedTraceRecallRunner", "verifiedTraceRecallReport",
        "verifiedTraceLogger", "verifiedTraceSchema", "verifiedTraceRlvrBridge",
        "verifiedTraceFaithfulnessProbe", "verifiedTraceCrossTraceMiner",
    ]
    for k in required:
        assert k in ai, f"missing artifactIndex entry: {k}"
        assert ai[k], f"empty artifactIndex entry: {k}"


def test_manifest_degrades_honestly_when_artifact_absent() -> None:
    """When the recall artifact is missing, the manifest must report None (not a
    fake CONFIRMED), matching the no-overclaim discipline."""
    from tools.build_agi_proof_package import build_manifest
    recall_path = ROOT / "agi-proof" / "verified-traces" / "verified-trace-recall.public-report.json"
    if not recall_path.exists():
        # artifact already absent in this checkout -> verdict must be None
        vt = build_manifest(generated="test")["verifiedTraces"]
        assert vt["recallVerdict"] is None, vt
        return
    # temporarily move the artifact aside, rebuild, then restore it
    backup = recall_path.read_bytes()
    recall_path.unlink()
    try:
        vt = build_manifest(generated="test")["verifiedTraces"]
        assert vt["recallVerdict"] is None, vt
        assert "not yet generated" in vt["scope"].lower()
    finally:
        recall_path.write_bytes(backup)


def main() -> int:
    test_manifest_reads_recall_verdict_from_artifact()
    print(f"ok {test_manifest_reads_recall_verdict_from_artifact.__name__}")
    test_manifest_artifact_index_has_verified_trace_entries()
    print(f"ok {test_manifest_artifact_index_has_verified_trace_entries.__name__}")
    test_manifest_degrades_honestly_when_artifact_absent()
    print(f"ok {test_manifest_degrades_honestly_when_artifact_absent.__name__}")
    print("PASS agi-proof manifest trace tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
