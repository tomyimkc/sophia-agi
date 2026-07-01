# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf.gap_nodes: open-item parsing + concept linking on synthetics.

Uses a small hand-written failure-ledger table and a synthetic evidence manifest
so the tests are hermetic (no dependency on the live, drifting ledger). Asserts:
open rows are parsed and Closed/Resolved rows are excluded, the manifest openItems
are merged in, gap ids are well-formed, and keyword linking to concept pages is
deterministic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name, rel_path):
    """Load a module by file path, bypassing okf/__init__ (which may drag in a
    Python-3.11+ regex sibling that fails to compile under older runtimes)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gap_nodes = _load_module("okf_gap_nodes_test", "okf/gap_nodes.py")


SYNTHETIC_LEDGER = """# Failure Ledger

| Failure ID | Status | Claim impact | Required response |
|---|---|---|---|
| confirmation-bias-recall-not-run-2026-06-25 | Open (harness built) | impact text | do X |
| calculus-priority-dispute-open-2026-06-24 | Open (candidate — instrument only) | impact | resp |
| forged-source-2026-06-20 | Closed (honest NEGATIVE — retracted) | none | none |
| some-metric-2026-06-19 | Resolved (fixed mid-sweep) | none | none |
| open-judge-endpoint-2026-06-29 | OpenAI-compatible endpoint required | impact | resp |
| unrelated-gpu-lever-2026-06-30 | Open (v5 lever built) | impact | resp |
"""


class _FakePage:
    def __init__(self, pid, meta):
        self._id = pid
        self.meta = dict(meta)

    @property
    def id(self):
        return self._id


def _concept_pages():
    return [
        _FakePage("confirmation_bias", {
            "pageType": "concept", "domain": "psychology", "subfield": "cognitive",
            "canonicalTitleEn": "confirmation bias"}),
        _FakePage("calculus", {
            "pageType": "concept", "domain": "science", "subfield": "mathematics",
            "canonicalTitleEn": "calculus"}),
        _FakePage("dna_double_helix", {
            "pageType": "concept", "domain": "science", "subfield": "biology",
            "canonicalTitleEn": "DNA double helix structure"}),
    ]


def _write_temp(text: str, suffix: str) -> Path:
    fd = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
    fd.write(text)
    fd.close()
    return Path(fd.name)


def test_open_items_parsed_and_closed_excluded():
    ledger = _write_temp(SYNTHETIC_LEDGER, ".md")
    try:
        rows = gap_nodes.parse_ledger_open_items(ledger)
    finally:
        ledger.unlink()
    ids = {r["ledgerId"] for r in rows}
    assert "confirmation-bias-recall-not-run-2026-06-25" in ids
    assert "calculus-priority-dispute-open-2026-06-24" in ids
    assert "unrelated-gpu-lever-2026-06-30" in ids
    # OpenAI-compatible bled into status but leads with 'open' -> included.
    assert "open-judge-endpoint-2026-06-29" in ids
    # Closed / Resolved excluded.
    assert "forged-source-2026-06-20" not in ids
    assert "some-metric-2026-06-19" not in ids


def test_header_and_separator_not_parsed_as_gaps():
    ledger = _write_temp(SYNTHETIC_LEDGER, ".md")
    try:
        rows = gap_nodes.parse_ledger_open_items(ledger)
    finally:
        ledger.unlink()
    ids = {r["ledgerId"] for r in rows}
    assert "Failure ID" not in ids
    assert "---" not in ids
    assert not any(set(i) <= set("-: ") for i in ids)


def test_gap_node_shape():
    ledger = _write_temp(SYNTHETIC_LEDGER, ".md")
    try:
        gaps = gap_nodes.load_gaps(ledger)
    finally:
        ledger.unlink()
    assert gaps, "expected gaps from the synthetic ledger"
    for g in gaps:
        assert g["id"].startswith("gap-")
        assert g["pageType"] == "gap"
        assert g["status"] == "open"
        assert g["title"]
        assert isinstance(g["concerns"], list)
        assert g["ledgerId"]
    # Deterministic ordering by gap id.
    assert gaps == sorted(gaps, key=lambda g: g["id"])


def test_manifest_openitems_merged():
    ledger = _write_temp(SYNTHETIC_LEDGER, ".md")
    manifest = _write_temp(json.dumps({
        "failureLedgerSummary": {
            "openItems": [
                "confirmation-bias-recall-not-run-2026-06-25",  # dup with ledger
                "dna-double-helix-probe-not-run-2026-06-26",     # manifest-only
            ]
        }
    }), ".json")
    try:
        gaps = gap_nodes.load_gaps(ledger, manifest)
    finally:
        ledger.unlink()
        manifest.unlink()
    ledger_ids = {g["ledgerId"] for g in gaps}
    assert "dna-double-helix-probe-not-run-2026-06-26" in ledger_ids
    # No duplicate for the shared id.
    dup = [g for g in gaps
           if g["ledgerId"] == "confirmation-bias-recall-not-run-2026-06-25"]
    assert len(dup) == 1


def test_link_gaps_to_concepts_deterministic():
    ledger = _write_temp(SYNTHETIC_LEDGER, ".md")
    try:
        gaps = gap_nodes.load_gaps(ledger)
    finally:
        ledger.unlink()
    pages = _concept_pages()
    linked_a = gap_nodes.link_gaps_to_concepts([dict(g) for g in gaps], pages)
    linked_b = gap_nodes.link_gaps_to_concepts([dict(g) for g in reversed(gaps)], pages)
    by_id_a = {g["id"]: g["concerns"] for g in linked_a}
    by_id_b = {g["id"]: g["concerns"] for g in linked_b}
    assert by_id_a == by_id_b, "concept linking is not deterministic"

    # 'confirmation-bias-...' must concern the confirmation_bias concept.
    cb = next(g for g in linked_a if g["ledgerId"].startswith("confirmation-bias"))
    assert "confirmation_bias" in cb["concerns"]
    # 'calculus-...' must concern the calculus concept.
    calc = next(g for g in linked_a if g["ledgerId"].startswith("calculus"))
    assert "calculus" in calc["concerns"]
    # concerns lists are sorted.
    for g in linked_a:
        assert g["concerns"] == sorted(g["concerns"])


def test_coverage_metric():
    ledger = _write_temp(SYNTHETIC_LEDGER, ".md")
    try:
        gaps = gap_nodes.load_gaps(ledger)
    finally:
        ledger.unlink()
    gap_nodes.link_gaps_to_concepts(gaps, _concept_pages())
    cov = gap_nodes.coverage(gaps)
    assert cov["gapCount"] == len(gaps)
    assert 0.0 <= cov["coverage"] <= 1.0
    assert cov["linkedGapCount"] >= 2  # confirmation-bias + calculus at least


def test_missing_files_are_safe():
    assert gap_nodes.parse_ledger_open_items("/nonexistent/ledger.md") == []
    assert gap_nodes.parse_manifest_open_items("/nonexistent/manifest.json") == []
    assert gap_nodes.load_gaps("/nonexistent/ledger.md", "/nonexistent/m.json") == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"ALL TESTS PASSED ({len(tests)} tests)")
