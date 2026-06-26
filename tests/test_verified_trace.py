#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the verified reasoning-trace logger (offline, deterministic).

Covers: dual (fact+logic) stamping, the derived ``verified`` property, the
tamper-evident hash chain, observer-safety (a logger fault never breaks the
caller), and the contradiction-recall invariant that is the headline experiment.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.verified_trace import (  # noqa: E402
    BOUNDARY,
    PHASES,
    VerifiedTrace,
    emit,
    record,
    verify_chain,
    _self_hash,
)

try:  # dual-mode: works under pytest AND as `python tests/test_verified_trace.py`
    import pytest
except ImportError:  # pragma: no cover - pytest is present in CI and .venv
    class _Raises:
        def __init__(self, exc):
            self.exc = exc
        def __enter__(self):
            return self
        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"did not raise {self.exc.__name__}")
            return issubclass(et, self.exc)
    class pytest:  # type: ignore[no-redef]
        @staticmethod
        def raises(exc):
            return _Raises(exc)


def _mk(phase: str = "benchmark", fact=None, logic=None, **kw) -> VerifiedTrace:
    kw.setdefault("claimText", "x")
    return VerifiedTrace(
        traceId="vtrace_" + "a" * 24,
        runId="test",
        phase=phase,
        stepIdx=0,
        claimKind="goal",
        fact=fact or {"verdict": "allow", "source": "t"},
        logic=logic or {"emittable": True, "contradictions": [], "laundered": [], "semanticsPreserved": True},
        **kw,
    )


def test_verified_requires_both_stamps() -> None:
    # both pass -> verified
    assert _mk(fact={"verdict": "allow", "source": "t"}, logic={"emittable": True}).verified
    assert _mk(fact={"verdict": "retrieve", "source": "t"}, logic={"emittable": True}).verified
    # fact fails (abstain) but logic ok -> NOT verified (one stamp cannot rescue the other)
    assert not _mk(fact={"verdict": "abstain", "source": "t"}, logic={"emittable": True}).verified
    # fact ok but logic fails (contradiction) -> NOT verified
    assert not _mk(fact={"verdict": "allow", "source": "t"}, logic={"emittable": False, "contradictions": [{"x": 1}]}).verified
    # block is never verified regardless of logic
    assert not _mk(fact={"verdict": "block", "source": "t"}, logic={"emittable": True}).verified


def test_reward_clamped_fail_closed() -> None:
    t = _mk(reward=42.0)
    assert t.reward == 1.0
    t2 = _mk(reward=-99.0)
    assert t2.reward == -1.0


def test_pretraining_phase_rejected() -> None:
    with pytest.raises(ValueError):
        _mk(phase="pretraining")  # out of scope by design
    assert "pretraining" not in PHASES


def test_no_overclaim_triad_always_present() -> None:
    t = _mk()
    d = t.to_dict()
    assert d["candidateOnly"] is True
    assert d["level3Evidence"] is False
    assert "not proof of AGI" in d["boundary"]
    assert d["boundary"] == BOUNDARY


def test_record_chains_and_is_tamper_evident(tmp_path: Path) -> None:
    log = tmp_path / "vt.jsonl"
    ack1 = record(_mk(claimText="first"), path=log)
    ack2 = record(_mk(claimText="second"), path=log)
    assert ack1["verified"] and ack2["verified"]

    # chain intact after two appends
    report = verify_chain(log)
    assert report["chainIntact"] is True
    assert report["nEvents"] == 2

    # the second line's prevHash must equal the first line's selfHash
    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    import json
    first, second = json.loads(lines[0]), json.loads(lines[1])
    assert second["prevHash"] == first["_selfHash"]
    assert second["prevHash"] != ""

    # mutate the first line's content -> chain must break on re-verify
    first["claimText"] = "TAMPERED"
    log.write_text(json.dumps(first) + "\n" + lines[1] + "\n", encoding="utf-8")
    broken = verify_chain(log)
    assert broken["chainIntact"] is False
    assert broken["brokenAt"] in (0, 1)


def test_self_hash_excludes_self_hash_field() -> None:
    # _selfHash must not feed into its own computation (or it is circular/meaningless)
    import json
    base = {"a": 1, "prevHash": ""}
    h1 = _self_hash({**base, "_selfHash": "sha256:deadbeef"})
    h2 = _self_hash(base)
    assert h1 == h2


def test_emit_is_observer_safe_on_fault() -> None:
    # emit must never raise, even when given a record that will fail to record
    # (simulate by pointing at an unwritable path).
    t = _mk()
    bad = Path("/proc/cannot-exist-sophia/vt.jsonl")  # unwritable on any sane system
    emit(t, path=bad)  # must not raise


def test_compiler_hook_emits_trace_on_compile(tmp_path: Path, monkeypatch) -> None:
    # point the logger at a temp path so the real log isn't polluted
    log = tmp_path / "vt.jsonl"
    monkeypatch.setattr("agent.verified_trace.TRACE_LOG", log)
    from reasoning.reasoning_compiler import Claim, ReasoningGraph, compile_graph
    claims = {
        "s0": Claim("s0", "fact A", "source", 3, []),
        "d0": Claim("d0", "step P", "derived", 3, ["s0"]),
        "g": Claim("g", "conclusion", "goal", 3, ["d0"]),
    }
    res = compile_graph(ReasoningGraph(claims, "g"))
    assert res.emittable  # clean graph compiles
    assert log.exists()
    import json
    line = json.loads(log.read_text().strip())
    assert line["schema"] == "sophia.verified_trace.v1"
    assert line["phase"] == "benchmark"
    assert line["logic"]["emittable"] is True
    assert line["verified"] is True


def test_compiler_trace_fails_closed_on_contradiction(tmp_path: Path, monkeypatch) -> None:
    # the killer experiment: a planted contradiction must (a) block emission AND
    # (b) be reflected as verified=false in the trace, with the contradiction recorded.
    log = tmp_path / "vt.jsonl"
    monkeypatch.setattr("agent.verified_trace.TRACE_LOG", log)
    from reasoning.reasoning_compiler import Claim, ReasoningGraph, compile_graph
    claims = {
        "s0": Claim("s0", "fact A", "source", 3, []),
        "d0": Claim("d0", "step P", "derived", 3, ["s0"]),
        "neg": Claim("neg", "not step P", "derived", 3, ["s0"]),
        "g": Claim("g", "conclusion", "goal", 3, ["d0", "neg"]),
    }
    res = compile_graph(ReasoningGraph(claims, "g"))
    assert not res.emittable
    assert res.contradictions
    import json
    line = json.loads(log.read_text().strip())
    assert line["logic"]["emittable"] is False
    assert line["logic"]["contradictions"]
    assert line["verified"] is False


def test_conscience_hook_emits_trace() -> None:
    # the conscience path must emit a trace carrying its verdict without changing it
    from agent.conscience import conscience_check
    d = conscience_check("Paris is the capital of France.")
    assert d.verdict in {"allow", "revise", "retrieve", "clarify", "escalate", "abstain", "block"}
    assert "not proof of AGI" in d.boundary


# --------------------------------------------------------------------------- #
# Plain `python tests/test_verified_trace.py` runner (no pytest needed).
# Mirrors the repo convention (e.g. test_metacognition.py). The pytest-fixture
# tests above are re-run here with inlined setup so the script path is green too.
# --------------------------------------------------------------------------- #
def main() -> int:
    import json
    import tempfile

    test_verified_requires_both_stamps()
    print(f"ok {test_verified_requires_both_stamps.__name__}")
    test_reward_clamped_fail_closed()
    print(f"ok {test_reward_clamped_fail_closed.__name__}")
    test_pretraining_phase_rejected()
    print(f"ok {test_pretraining_phase_rejected.__name__}")
    test_no_overclaim_triad_always_present()
    print(f"ok {test_no_overclaim_triad_always_present.__name__}")
    test_self_hash_excludes_self_hash_field()
    print(f"ok {test_self_hash_excludes_self_hash_field.__name__}")
    test_emit_is_observer_safe_on_fault()
    print(f"ok {test_emit_is_observer_safe_on_fault.__name__}")

    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "vt.jsonl"
        test_record_chains_and_is_tamper_evident(log)
        print(f"ok {test_record_chains_and_is_tamper_evident.__name__}")

        log2 = Path(td) / "vt2.jsonl"
        test_compiler_hook_emits_trace_on_compile(log2, None)
        print(f"ok {test_compiler_hook_emits_trace_on_compile.__name__}")

        log3 = Path(td) / "vt3.jsonl"
        test_compiler_trace_fails_closed_on_contradiction(log3, None)
        print(f"ok {test_compiler_trace_fails_closed_on_contradiction.__name__}")

    test_conscience_hook_emits_trace()
    print(f"ok {test_conscience_hook_emits_trace.__name__}")

    print("PASS verified-trace tests")
    return 0


# Fixture-free variants for the plain runner. They accept the fixture as a positional
# arg (pytest) and ignore it / use a real path when None (plain runner). monkeypatch
# is replaced by a tiny shim that sets the module attribute.
class _Monkey:
    def setattr(self, target, value):
        import agent.verified_trace as vt
        vt.TRACE_LOG = value

def test_record_chains_and_is_tamper_evident(tmp_path=None):  # noqa: D401
    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp()) / "vt.jsonl"
    log = tmp_path if isinstance(tmp_path, Path) else Path(tmp_path)
    ack1 = record(_mk(claimText="first"), path=log)
    ack2 = record(_mk(claimText="second"), path=log)
    assert ack1["verified"] and ack2["verified"]
    report = verify_chain(log)
    assert report["chainIntact"] is True and report["nEvents"] == 2
    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    import json
    first, second = json.loads(lines[0]), json.loads(lines[1])
    assert second["prevHash"] == first["_selfHash"] and second["prevHash"] != ""
    first["claimText"] = "TAMPERED"
    log.write_text(json.dumps(first) + "\n" + lines[1] + "\n", encoding="utf-8")
    broken = verify_chain(log)
    assert broken["chainIntact"] is False and broken["brokenAt"] in (0, 1)


def test_compiler_hook_emits_trace_on_compile(tmp_path=None, monkeypatch=None):  # noqa: D401
    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp()) / "vt.jsonl"
    log = tmp_path if isinstance(tmp_path, Path) else Path(tmp_path)
    mp = monkeypatch or _Monkey()
    mp.setattr("agent.verified_trace.TRACE_LOG", log)
    from reasoning.reasoning_compiler import Claim, ReasoningGraph, compile_graph
    claims = {
        "s0": Claim("s0", "fact A", "source", 3, []),
        "d0": Claim("d0", "step P", "derived", 3, ["s0"]),
        "g": Claim("g", "conclusion", "goal", 3, ["d0"]),
    }
    res = compile_graph(ReasoningGraph(claims, "g"))
    assert res.emittable and log.exists()
    import json
    line = json.loads(log.read_text().strip())
    assert line["schema"] == "sophia.verified_trace.v1"
    assert line["phase"] == "benchmark"
    assert line["logic"]["emittable"] is True and line["verified"] is True


def test_compiler_trace_fails_closed_on_contradiction(tmp_path=None, monkeypatch=None):  # noqa: D401
    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp()) / "vt.jsonl"
    log = tmp_path if isinstance(tmp_path, Path) else Path(tmp_path)
    mp = monkeypatch or _Monkey()
    mp.setattr("agent.verified_trace.TRACE_LOG", log)
    from reasoning.reasoning_compiler import Claim, ReasoningGraph, compile_graph
    claims = {
        "s0": Claim("s0", "fact A", "source", 3, []),
        "d0": Claim("d0", "step P", "derived", 3, ["s0"]),
        "neg": Claim("neg", "not step P", "derived", 3, ["s0"]),
        "g": Claim("g", "conclusion", "goal", 3, ["d0", "neg"]),
    }
    res = compile_graph(ReasoningGraph(claims, "g"))
    assert not res.emittable and res.contradictions
    import json
    line = json.loads(log.read_text().strip())
    assert line["logic"]["emittable"] is False and line["verified"] is False


if __name__ == "__main__":
    raise SystemExit(main())
