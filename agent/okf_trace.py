# PLANNING/SUBSTRATE ONLY - no capability claim; canClaimAGI stays false.
"""OKF traceable-memory substrate: trace log + DAG + wrong-step locator.

PURE / OFFLINE / DETERMINISTIC. stdlib only (json, argparse, pathlib, ...).
Implements the SHARED OKF API CONTRACT exactly:
  append_trace, read_trace, build_dag, has_cycle, retrace,
  locate_wrong_step, offline_invariants, and a __main__ CLI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:  # package mode: `import agent.okf_trace`
    from agent.okf_schema import (
        OKFNode,
        content_id,
        from_markdown,
        to_markdown,
        validate,
    )
except ImportError:  # script mode: `python agent/okf_trace.py` (agent/ is on sys.path)
    from okf_schema import (
        OKFNode,
        content_id,
        from_markdown,
        to_markdown,
        validate,
    )


# ---------------------------------------------------------------------------
# Trace log (jsonl). ts is INJECTABLE so tests never touch wall-clock.
# ---------------------------------------------------------------------------
def append_trace(path, node_id: str, step: int, payload: dict, ts: float | None = None) -> None:
    """Append one jsonl row to `path`. ts is injectable (no wall-clock here)."""
    row = {"node_id": node_id, "step": step, "payload": payload, "ts": ts}
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_trace(path) -> list[dict]:
    """Read a jsonl trace file into a list of dicts (order preserved)."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# DAG construction + cycle detection over node .links.
# ---------------------------------------------------------------------------
def build_dag(nodes: list[OKFNode]) -> dict:
    """Adjacency map {node_id: [linked_id, ...]} from node.links."""
    adj: dict[str, list[str]] = {}
    for node in nodes:
        adj[node.id] = list(node.links)
    return adj


def has_cycle(nodes: list[OKFNode]) -> bool:
    """True if the link graph contains a directed cycle."""
    adj = build_dag(nodes)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {nid: WHITE for nid in adj}

    def visit(u: str) -> bool:
        color[u] = GRAY
        for v in adj.get(u, []):
            if v not in color:
                # link to unknown node: treat as terminal leaf, no cycle
                continue
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and visit(v):
                return True
        color[u] = BLACK
        return False

    for nid in adj:  # dict preserves insertion order -> deterministic
        if color[nid] == WHITE:
            if visit(nid):
                return True
    return False


def retrace(nodes: list[OKFNode], start_id: str) -> list[str]:
    """Deterministic DFS preorder of reachable node ids from start_id."""
    adj = build_dag(nodes)
    order: list[str] = []
    seen: set[str] = set()

    def dfs(u: str) -> None:
        if u in seen:
            return
        seen.add(u)
        order.append(u)
        for v in adj.get(u, []):
            dfs(v)

    if start_id in adj:
        dfs(start_id)
    return order


# ---------------------------------------------------------------------------
# The core Lightman 'exact error location' primitive.
# ---------------------------------------------------------------------------
def locate_wrong_step(trace: list, verdicts: dict[str, str]) -> str | None:
    """Return FIRST node_id in trace order whose verdict == 'fail', else None.

    Accepts a trace as a list of dict rows (each carrying 'node_id') OR a bare list of node_id
    strings — the locator is tolerant of both shapes so callers can pass either a rich trace log
    or a plain node-id sequence (Postel's law; keeps the contract a superset of the harness usage).
    """
    for row in trace:
        node_id = row.get("node_id") if isinstance(row, dict) else row
        if node_id is not None and verdicts.get(node_id) == "fail":
            return node_id
    return None


# ---------------------------------------------------------------------------
# Offline invariants self-check.
# ---------------------------------------------------------------------------
def _sample_node(verdict: str | None = None, links: list[str] | None = None) -> OKFNode:
    title = "sample title"
    body = "sample body line 1\nsample body line 2"
    nid = content_id("step", title, body)
    return OKFNode(
        id=nid,
        node_type="step",
        title=title,
        body=body,
        sources=["data/x.json#a", "data/y.json#b"],
        links=links or [],
        verifier="unit",
        verdict=verdict,
        moral_standard=None,
    )


def offline_invariants() -> tuple[bool, dict]:
    """Run all offline invariants. Returns (all_ok, {check_name: bool})."""
    results: dict[str, bool] = {}

    # 1. content-id determinism
    a = content_id("fact", "t", "b")
    b = content_id("fact", "t", "b")
    c = content_id("fact", "t", "b2")
    results["content_id_determinism"] = (a == b) and (a != c) and a.startswith("fact:")

    # 2. markdown roundtrip
    n = _sample_node(verdict="pass")
    rt = from_markdown(to_markdown(n))
    results["md_roundtrip"] = (rt == n) and (validate(rt) == [])

    # 2b. roundtrip with None verdict + moral_standard set
    n2 = _sample_node(verdict=None)
    n2.moral_standard = "do_no_harm"
    n2.id = content_id(n2.node_type, n2.title, n2.body)
    rt2 = from_markdown(to_markdown(n2))
    results["md_roundtrip_none"] = rt2 == n2

    # 3. cycle detection
    x = _sample_node()
    x.title, x.body = "X", "bx"
    x.id = content_id("step", "X", "bx")
    y = _sample_node()
    y.title, y.body = "Y", "by"
    y.id = content_id("step", "Y", "by")
    x.links = [y.id]
    y.links = [x.id]
    acyclic_x = _sample_node()
    acyclic_x.title, acyclic_x.body = "A", "ba"
    acyclic_x.id = content_id("step", "A", "ba")
    acyclic_x.links = [y.id]  # y has no link back to A
    results["cycle_detect"] = has_cycle([x, y]) and not has_cycle([acyclic_x])

    # 4. locate-wrong-step correctness
    trace = [
        {"node_id": "n1", "step": 0, "payload": {}, "ts": 0.0},
        {"node_id": "n2", "step": 1, "payload": {}, "ts": 1.0},
        {"node_id": "n3", "step": 2, "payload": {}, "ts": 2.0},
    ]
    verdicts_fail = {"n1": "pass", "n2": "fail", "n3": "fail"}
    verdicts_pass = {"n1": "pass", "n2": "pass", "n3": "pass"}
    results["locate_wrong_step"] = (
        locate_wrong_step(trace, verdicts_fail) == "n2"
        and locate_wrong_step(trace, verdicts_pass) is None
    )

    # 5. append/read jsonl roundtrip (in scratch temp file)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tp = Path(td) / "trace.jsonl"
        append_trace(tp, "n1", 0, {"k": "v"}, ts=123.0)
        append_trace(tp, "n2", 1, {"k": 2}, ts=124.0)
        rows = read_trace(tp)
        results["append_read_roundtrip"] = (
            len(rows) == 2
            and rows[0]["node_id"] == "n1"
            and rows[0]["ts"] == 123.0
            and rows[1]["payload"] == {"k": 2}
        )

    all_ok = all(results.values())
    return all_ok, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_self_test() -> int:
    ok, results = offline_invariants()
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"{'PASS' if ok else 'FAIL'}  ALL")
    return 0 if ok else 1


def _cmd_report() -> int:
    ok, results = offline_invariants()
    report = {
        "module": "agent/okf_trace.py",
        "substrate_only": True,
        "canClaimAGI": False,
        "all_ok": ok,
        "checks": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


def _demo_nodes() -> list[OKFNode]:
    def mk(title: str, body: str, verdict: str | None, links: list[str]) -> OKFNode:
        nid = content_id("step", title, body)
        return OKFNode(
            id=nid, node_type="step", title=title, body=body,
            sources=[], links=links, verifier="demo", verdict=verdict,
            moral_standard=None,
        )

    # Build leaf-first so we can wire links by id.
    s3 = mk("final synthesis", "compose answer", "pass", [])
    s2 = mk("seeded WRONG step", "buggy reasoning here", "fail", [s3.id])
    s1 = mk("gather facts", "collect sources", "pass", [s2.id])
    return [s1, s2, s3]


def _cmd_demo() -> int:
    nodes = _demo_nodes()
    start = nodes[0].id
    order = retrace(nodes, start)
    verdicts = {n.id: (n.verdict or "none") for n in nodes}
    trace = [{"node_id": nid, "step": i, "payload": {}, "ts": float(i)}
             for i, nid in enumerate(order)]
    wrong = locate_wrong_step(trace, verdicts)
    print("DEMO: tiny OKF DAG (step1 -> step2[FAIL] -> step3)")
    print(f"  build_dag: {json.dumps(build_dag(nodes))}")
    print(f"  has_cycle: {has_cycle(nodes)}")
    print(f"  retrace order: {order}")
    wrong_title = next((n.title for n in nodes if n.id == wrong), None)
    print(f"  locate_wrong_step -> {wrong}  ({wrong_title})")
    ok = wrong is not None and wrong_title == "seeded WRONG step"
    print(f"  {'PASS' if ok else 'FAIL'}  located seeded fail node")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OKF traceable-memory substrate (offline).")
    parser.add_argument("--self-test", action="store_true", help="run offline invariants, print PASS/FAIL")
    parser.add_argument("--report", action="store_true", help="print JSON invariant report")
    parser.add_argument("--demo", action="store_true", help="build tiny DAG with seeded fail node, locate it")
    args = parser.parse_args(argv)

    if args.self_test:
        return _cmd_self_test()
    if args.report:
        return _cmd_report()
    if args.demo:
        return _cmd_demo()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
