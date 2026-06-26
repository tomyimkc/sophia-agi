#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the verified-trace axis wired into the capability panel.

Covers: the axis aggregates correctly from a seeded trace log; the honest-empty
path (nTotal=0, no crash); the panel embeds it under axes.verifiedTraces and
exposes traceChainIntact / traceContradictionRecall checks; the panel's existing
'passed' logic is unchanged (the new axis is informational, not gating).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _seed(log: Path) -> None:
    import agent.verified_trace as vt
    vt.TRACE_LOG = log
    from agent.verified_trace import VerifiedTrace, record
    record(VerifiedTrace(traceId="vtrace_" + "a" * 24, runId="r", phase="benchmark",
                         stepIdx=0, claimText="ok", claimKind="goal",
                         fact={"verdict": "allow", "source": "t"},
                         logic={"emittable": True, "contradictions": [], "laundered": [], "semanticsPreserved": True}))
    record(VerifiedTrace(traceId="vtrace_" + "b" * 24, runId="r", phase="benchmark",
                         stepIdx=1, claimText="contra", claimKind="goal",
                         fact={"verdict": "abstain", "source": "t"},
                         logic={"emittable": False, "contradictions": [{"x": 1}], "laundered": [], "semanticsPreserved": False}))


def test_axis_honest_on_empty() -> None:
    from tools.eval_capability_panel import _verified_trace_axis
    with tempfile.TemporaryDirectory() as td:
        axis = _verified_trace_axis(trace_log=Path(td) / "none.jsonl")
        assert axis["nTotal"] == 0
        assert axis["stepVerifiedRate"] is None
        assert axis["contradictionRecall"] is None
        assert "note" in axis  # explains why (empty log), not a failure


def test_axis_aggregates_seeded() -> None:
    from tools.eval_capability_panel import _verified_trace_axis
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "vt.jsonl"
        _seed(log)
        axis = _verified_trace_axis(trace_log=log)
        assert axis["nTotal"] == 2
        assert axis["stepVerifiedRate"] == 0.5            # 1 of 2 verified
        assert axis["contradictionRecall"] == 1.0          # the one contradiction was blocked
        assert axis["chainIntact"] is True
        assert axis["phases"] == {"benchmark": 2}


def test_panel_embeds_axis_and_keeps_passed_logic() -> None:
    import agent.verified_trace as vt
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "vt.jsonl"
        vt.TRACE_LOG = log
        _seed(log)
        from tools.eval_capability_panel import run
        # limit keeps the panel fast (the axis is what we assert on, not the SEIB run)
        r = run(out=None, limit=6)
        # axis present
        assert "verifiedTraces" in r["axes"]
        assert r["axes"]["verifiedTraces"]["nTotal"] == 2
        # new checks present
        assert r["checks"]["traceChainIntact"] is True
        assert r["checks"]["traceContradictionRecall"] is True  # recall in (None, 1.0)
        # the existing capability 'passed' logic is UNCHANGED (mock mode improves accuracy)
        assert r["passed"] is True
        # no-overclaim triad intact
        assert r["candidateOnly"] is True and r["validated"] is False


def test_panel_tamper_detection_flags_chain() -> None:
    import json
    from tools.eval_capability_panel import _verified_trace_axis
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "vt.jsonl"
        _seed(log)
        # mutate the first line's content but leave its old _selfHash -> chain breaks
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        first = json.loads(lines[0])
        first["claimText"] = "TAMPERED"
        log.write_text(json.dumps(first) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")
        axis = _verified_trace_axis(trace_log=log)
        assert axis["chainIntact"] is False


def main() -> int:
    test_axis_honest_on_empty();            print(f"ok {test_axis_honest_on_empty.__name__}")
    test_axis_aggregates_seeded();          print(f"ok {test_axis_aggregates_seeded.__name__}")
    test_panel_embeds_axis_and_keeps_passed_logic()
    print(f"ok {test_panel_embeds_axis_and_keeps_passed_logic.__name__}")
    test_panel_tamper_detection_flags_chain()
    print(f"ok {test_panel_tamper_detection_flags_chain.__name__}")
    print("PASS verified-trace panel tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
