# PLANNING/SUBSTRATE ONLY - no capability claim; canClaimAGI stays false.
"""Plain-script test for the OFFLINE OKF substrate (pytest may be absent).

Run standalone:  python tests/test_okf_trace.py
Prints 'test_okf_trace: PASS' and exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make agent/ importable regardless of cwd.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "agent"))

from okf_schema import OKFNode, content_id, from_markdown, to_markdown, validate  # noqa: E402
from okf_trace import (  # noqa: E402
    append_trace,
    build_dag,
    has_cycle,
    locate_wrong_step,
    offline_invariants,
    read_trace,
    retrace,
)


def _mk(node_type: str, title: str, body: str, **kw) -> OKFNode:
    nid = content_id(node_type, title, body)
    return OKFNode(id=nid, node_type=node_type, title=title, body=body, **kw)


def test_content_id_determinism_and_collision_resistance():
    a = content_id("fact", "Newton's laws", "F = ma")
    b = content_id("fact", "Newton's laws", "F = ma")
    assert a == b, "content_id must be deterministic"
    assert a.startswith("fact:"), "content_id must carry node_type prefix"
    # different body -> different id
    assert content_id("fact", "Newton's laws", "F=ma2") != a
    # different node_type -> different id (and prefix)
    assert content_id("step", "Newton's laws", "F = ma") != a
    # length-prefixing prevents boundary collision: ("ab","c") vs ("a","bc")
    assert content_id("fact", "ab", "c") != content_id("fact", "a", "bc")


def test_md_roundtrip_equality():
    n = _mk(
        "decision", "pick approach", "line1\nline2\nwith: colon and \"quote\"",
        sources=['data/a.json#k', 'data/b.json#k2'],
        links=["step:abc123def456"],
        verifier="deontic_verifier",
        verdict="pass",
        moral_standard="do_no_harm",
    )
    rt = from_markdown(to_markdown(n))
    assert rt == n, f"roundtrip mismatch:\n{n}\n!=\n{rt}"
    assert validate(rt) == [], f"validate failed: {validate(rt)}"

    # None scalars roundtrip
    n2 = _mk("skill", "t", "b", verifier=None, verdict=None, moral_standard=None)
    rt2 = from_markdown(to_markdown(n2))
    assert rt2 == n2
    assert rt2.verdict is None and rt2.verifier is None and rt2.moral_standard is None

    # empty lists roundtrip
    assert from_markdown(to_markdown(n2)).sources == []


def test_validate_catches_problems():
    good = _mk("fact", "t", "b", verdict="none")
    assert validate(good) == []
    bad_type = OKFNode(id="x:1", node_type="bogus", title="t", body="b")
    assert any("node_type" in e for e in validate(bad_type))
    bad_id = _mk("fact", "t", "b")
    bad_id.id = "fact:deadbeef0000"
    assert any("content_id" in e for e in validate(bad_id))
    bad_verdict = _mk("fact", "t", "b", verdict="maybe")
    assert any("verdict" in e for e in validate(bad_verdict))


def test_cycle_detection():
    x = _mk("step", "X", "bx")
    y = _mk("step", "Y", "by")
    x.links = [y.id]
    y.links = [x.id]
    assert has_cycle([x, y]) is True
    # acyclic chain
    a = _mk("step", "A", "ba")
    b = _mk("step", "B", "bb")
    a.links = [b.id]
    assert has_cycle([a, b]) is False
    # self loop
    s = _mk("step", "S", "bs")
    s.links = [s.id]
    assert has_cycle([s]) is True


def test_retrace_order_determinism():
    a = _mk("step", "A", "ba")
    b = _mk("step", "B", "bb")
    c = _mk("step", "C", "bc")
    d = _mk("step", "D", "bd")
    a.links = [b.id, c.id]
    b.links = [d.id]
    c.links = [d.id]
    nodes = [a, b, c, d]
    order1 = retrace(nodes, a.id)
    order2 = retrace(nodes, a.id)
    assert order1 == order2, "retrace must be deterministic"
    # DFS preorder: a, b, d, c  (d visited via b, not re-added under c)
    assert order1 == [a.id, b.id, d.id, c.id], order1
    assert order1[0] == a.id
    # unknown start -> empty
    assert retrace(nodes, "nope:000000000000") == []


def test_locate_wrong_step():
    trace = [
        {"node_id": "n1", "step": 0, "payload": {}, "ts": 0.0},
        {"node_id": "n2", "step": 1, "payload": {}, "ts": 1.0},
        {"node_id": "n3", "step": 2, "payload": {}, "ts": 2.0},
    ]
    # first fail wins even when a later one also fails
    verdicts = {"n1": "pass", "n2": "fail", "n3": "fail"}
    assert locate_wrong_step(trace, verdicts) == "n2"
    # all pass -> None
    assert locate_wrong_step(trace, {"n1": "pass", "n2": "pass", "n3": "pass"}) is None
    # missing verdicts treated as not-fail -> None
    assert locate_wrong_step(trace, {}) is None
    # empty trace -> None
    assert locate_wrong_step([], verdicts) is None


def test_append_read_jsonl_roundtrip_injected_ts():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sub" / "trace.jsonl"  # also exercises mkdir
        append_trace(path, "n1", 0, {"k": "v", "nested": [1, 2]}, ts=100.5)
        append_trace(path, "n2", 1, {"k": 2}, ts=200.5)
        rows = read_trace(path)
        assert len(rows) == 2
        assert rows[0] == {"node_id": "n1", "step": 0, "payload": {"k": "v", "nested": [1, 2]}, "ts": 100.5}
        assert rows[1]["ts"] == 200.5
        assert rows[1]["node_id"] == "n2"
    # reading a non-existent file -> []
    assert read_trace(Path(td) / "gone.jsonl") == []


def test_build_dag():
    a = _mk("step", "A", "ba")
    b = _mk("step", "B", "bb")
    a.links = [b.id]
    dag = build_dag([a, b])
    assert dag == {a.id: [b.id], b.id: []}


def test_offline_invariants():
    ok, results = offline_invariants()
    assert ok, f"offline_invariants failed: {results}"
    assert all(results.values())


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    if failures:
        print(f"test_okf_trace: FAIL ({failures}/{len(tests)} failed)")
        return 1
    print("test_okf_trace: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(_run())
