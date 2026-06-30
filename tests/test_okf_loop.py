# PLANNING/HARNESS ONLY - no capability claim; canClaimAGI stays false.
"""Standalone plain-script test for agent/okf_loop.py (pytest may be absent).

Run: python tests/test_okf_loop.py
Prints 'test_okf_loop: PASS' and exits 0 on success, else 1.
PURE / OFFLINE / DETERMINISTIC. stdlib only.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))

from agent.okf_loop import LoopLog, _fixed_clock, offline_invariants  # noqa: E402
from agent.okf_schema import from_markdown, validate  # noqa: E402
from agent.okf_trace import read_trace  # noqa: E402


def main() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"  [{'ok' if cond else 'XX'}] {name}")
        if not cond:
            fails.append(name)

    # 1) module invariants
    ok, _ = offline_invariants()
    check("offline_invariants", ok)

    with tempfile.TemporaryDirectory() as td:
        log = LoopLog("t1", root=td, clock=_fixed_clock())
        e = log.event("error", "bench returned non-zero")
        o = log.observe("triage", "spawned explore agent", actor="Explore-agent")
        log.reason("why", "root cause is a missing timeout")
        log.decide("plan", "add timeout")
        log.act("fix", "edit module", tool="Edit")
        vf = log.verify("gate-1", "lint failed", verdict="fail")
        log.act("fix2", "second edit", tool="Edit")
        log.verify("gate-2", "lint passed", verdict="pass")
        r = log.resolve("done", "resolved")

        # 2) one node file per step, all valid OKF nodes that roundtrip
        node_files = sorted((log.nodes_dir).glob("*.md"))
        check("nine_node_files", len(node_files) == 9)
        all_valid = True
        for nf in node_files:
            node = from_markdown(nf.read_text(encoding="utf-8").rstrip("\n"))
            if validate(node):
                all_valid = False
        check("all_nodes_valid_roundtrip", all_valid)

        # 3) the DAG spine: observe links back to the event node
        obs = from_markdown(node_files[1].read_text(encoding="utf-8").rstrip("\n"))
        check("observe_links_event", e in obs.links)

        # 4) trace replayable in order
        trace = read_trace(log.trace_path)
        check("trace_order", [r["step"] for r in trace] == list(range(9)))

        # 5) wrong-step localization finds the FIRST failing verify
        check("locates_first_fail", log.summary()["first_failed_node"] == vf)

        # 6) resolved
        check("resolved", log.summary()["resolved"] is True)

        # 7) unknown kind rejected
        try:
            log.step("nonsense", "x", "y")
            check("rejects_unknown_kind", False)
        except ValueError:
            check("rejects_unknown_kind", True)

    if fails:
        print(f"test_okf_loop: FAIL ({', '.join(fails)})")
        return 1
    print("test_okf_loop: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
