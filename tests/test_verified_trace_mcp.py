#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the verified-trace MCP tool impls (trace_query / trace_verify).

These exercise the impl layer in tools_impl.py (the real contract) without
requiring the FastMCP server to be importable — the @mcp.tool() wrappers in
server.py are thin dumps() shims around these. Read-only over a temp log.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _seed(log: Path) -> None:
    """Three traces: two verified, one blocked (fact+logic both fail)."""
    import agent.verified_trace as vt
    vt.TRACE_LOG = log
    from agent.verified_trace import VerifiedTrace, record
    record(VerifiedTrace(traceId="vtrace_" + "a" * 24, runId="r1", phase="benchmark",
                         stepIdx=0, claimText="ok1", claimKind="goal",
                         fact={"verdict": "allow", "source": "t"},
                         logic={"emittable": True, "contradictions": [], "laundered": [], "semanticsPreserved": True}))
    record(VerifiedTrace(traceId="vtrace_" + "b" * 24, runId="r1", phase="benchmark",
                         stepIdx=1, claimText="ok2", claimKind="goal",
                         fact={"verdict": "retrieve", "source": "t"},
                         logic={"emittable": True, "contradictions": [], "laundered": [], "semanticsPreserved": True}))
    record(VerifiedTrace(traceId="vtrace_" + "c" * 24, runId="r2", phase="conscience",
                         stepIdx=0, claimText="bad", claimKind="derived",
                         fact={"verdict": "block", "source": "t"},
                         logic={"emittable": False, "contradictions": [{"x": 1}], "laundered": [], "semanticsPreserved": False}))


def test_trace_query_aggregates() -> None:
    with tempfile.TemporaryDirectory() as td:
        _seed(Path(td) / "vt.jsonl")
        from sophia_mcp.tools_impl import trace_query
        q = trace_query()
        assert q["schema"] == "sophia.trace_query.v1"
        assert q["nTotal"] == 3
        assert q["metrics"]["stepVerifiedRate"] == round(2 / 3, 4)  # 2 of 3 verified
        assert q["chainIntact"] is True
        # fact-logic agreement: rows 1,2 both-ok agree; row3 both-fail agree -> 3/3
        assert q["metrics"]["factLogicAgreement"] == 1.0
        # boundary present (no-overclaim)
        assert "not proof of AGI" in q["boundary"]
        # _selfHash must be stripped from rows (internal bookkeeping)
        assert all("_selfHash" not in r for r in q["rows"])


def test_trace_query_filters() -> None:
    with tempfile.TemporaryDirectory() as td:
        _seed(Path(td) / "vt.jsonl")
        from sophia_mcp.tools_impl import trace_query
        # filter by run_id
        assert trace_query("r2")["nFiltered"] == 1
        # filter by verified
        assert trace_query(verified=True)["nFiltered"] == 2
        assert trace_query(verified=False)["nFiltered"] == 1
        # filter by phase
        assert trace_query(phase="conscience")["nFiltered"] == 1


def test_trace_verify_rederives_and_matches() -> None:
    with tempfile.TemporaryDirectory() as td:
        _seed(Path(td) / "vt.jsonl")
        from sophia_mcp.tools_impl import trace_verify
        # verified trace: stored True, re-derived True, match
        v_ok = trace_verify("vtrace_" + "a" * 24)
        assert v_ok["recheckMatches"] is True
        assert v_ok["rederivedVerified"] is True
        # blocked trace: stored False, re-derived False, match
        v_bad = trace_verify("vtrace_" + "c" * 24)
        assert v_bad["recheckMatches"] is True
        assert v_bad["rederivedVerified"] is False
        # chain check on whole log
        v_chain = trace_verify(check_chain=True)
        assert v_chain["chain"]["chainIntact"] is True
        # missing trace id -> structured error (not an exception)
        v_miss = trace_verify("vtrace_nope")
        assert "error" in v_miss


def test_trace_verify_detects_tamper() -> None:
    # mutate a stored line -> chain must break and recheck must surface it
    import json
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "vt.jsonl"
        _seed(log)
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        first = json.loads(lines[0])
        first["claimText"] = "TAMPERED"  # this changes _selfHash but we leave the old one
        log.write_text(json.dumps(first) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")
        from sophia_mcp.tools_impl import trace_verify
        report = trace_verify(check_chain=True)
        assert report["chain"]["chainIntact"] is False


def main() -> int:
    test_trace_query_aggregates();            print(f"ok {test_trace_query_aggregates.__name__}")
    test_trace_query_filters();               print(f"ok {test_trace_query_filters.__name__}")
    test_trace_verify_rederives_and_matches();print(f"ok {test_trace_verify_rederives_and_matches.__name__}")
    test_trace_verify_detects_tamper();       print(f"ok {test_trace_verify_detects_tamper.__name__}")
    print("PASS verified-trace MCP tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
