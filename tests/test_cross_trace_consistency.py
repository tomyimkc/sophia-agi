#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for cross-trace OKF contradiction mining (agent/cross_trace_consistency.py).

Verifies: a globally-consistent log reports globalConsistent=True with no
contradictions; a real cross-trace contradiction (X in one verified trace, not-X
in another verified trace) is caught even though each trace passed its own gates;
a contradiction involving an UNVERIFIED trace is correctly ignored (the gate
already caught it locally — the global miner is for the both-verified case); and
the honest coverage note is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _trace(claim: str, *, verified: bool = True, trace_id: str = "", run_id: str = "r1") -> dict:
    return {
        "traceId": trace_id or f"vtrace_{abs(hash(claim)) % 10**24:024d}",
        "runId": run_id,
        "claimText": claim,
        "verified": verified,
    }


def test_globally_consistent_log_has_no_contradictions() -> None:
    from agent.cross_trace_consistency import mine_contradictions
    traces = [
        _trace("the charter was written by the committee"),
        _trace("paris is the capital of france"),
        _trace("two plus two is four"),
    ]
    ledger = mine_contradictions(traces)
    assert ledger["globalConsistent"] is True
    assert ledger["contradictions"] == []
    assert ledger["nVerified"] == 3
    assert "not proof of global consistency" in ledger["coverageNote"]


def test_cross_trace_contradiction_caught_when_both_verified() -> None:
    from agent.cross_trace_consistency import mine_contradictions
    # the dangerous case: each run passed its own gates, but they disagree
    traces = [
        _trace("the charter was written by the committee", run_id="runA"),
        _trace("not the charter was written by the committee", run_id="runB"),
    ]
    ledger = mine_contradictions(traces)
    assert ledger["globalConsistent"] is False
    assert len(ledger["contradictions"]) == 1
    c = ledger["contradictions"][0]
    assert c["a"]["runId"] == "runA"
    assert c["b"]["runId"] == "runB"
    # the core claim is the normalized shared subject
    assert "charter" in c["claim"]


def test_contradiction_ignored_when_one_trace_unverified() -> None:
    from agent.cross_trace_consistency import mine_contradictions
    # one side is unverified -> the gate already caught it locally; the global
    # miner must NOT double-report it as a global inconsistency
    traces = [
        _trace("the charter was written by the committee", verified=True),
        _trace("not the charter was written by the committee", verified=False),
    ]
    ledger = mine_contradictions(traces)
    assert ledger["globalConsistent"] is True
    assert ledger["contradictions"] == []
    assert ledger["nVerified"] == 1  # only the verified trace was indexed


def test_negation_split_matches_compiler_convention() -> None:
    from agent.cross_trace_consistency import _split_negation
    # the compiler plants contradictions as 'not ' + base.statement
    core, negated = _split_negation("not step P")
    assert negated is True
    assert core == "step p"
    core2, negated2 = _split_negation("step P")
    assert negated2 is False
    assert core2 == "step p"
    # a positive and its negated form share the same core -> they pair
    assert core == core2


def test_mine_log_reads_default_log() -> None:
    import tempfile
    from pathlib import Path
    import agent.verified_trace as vt
    import agent.cross_trace_consistency as ctc
    log = Path(tempfile.mkdtemp()) / "vt.jsonl"
    vt.TRACE_LOG = log
    ctc_log = log  # mine_log honors the module attr
    # seed one consistent trace
    from sophia_contract.stores import _append_jsonl
    _append_jsonl(log, _trace("a true claim"))
    ledger = ctc.mine_log()
    assert ledger["nTraces"] == 1
    assert ledger["globalConsistent"] is True


def main() -> int:
    test_globally_consistent_log_has_no_contradictions()
    print(f"ok {test_globally_consistent_log_has_no_contradictions.__name__}")
    test_cross_trace_contradiction_caught_when_both_verified()
    print(f"ok {test_cross_trace_contradiction_caught_when_both_verified.__name__}")
    test_contradiction_ignored_when_one_trace_unverified()
    print(f"ok {test_contradiction_ignored_when_one_trace_unverified.__name__}")
    test_negation_split_matches_compiler_convention()
    print(f"ok {test_negation_split_matches_compiler_convention.__name__}")
    test_mine_log_reads_default_log()
    print(f"ok {test_mine_log_reads_default_log.__name__}")
    print("PASS cross-trace consistency tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
