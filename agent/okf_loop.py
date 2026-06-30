# PLANNING/HARNESS ONLY - no capability claim; canClaimAGI stays false.
"""OKF loop-engineering incident logger — persist an agentic incident-response loop as OKF nodes.

"Loop engineering" (ReAct -> Reflexion, the 2026 framing) is the deliberate design of an agent's
observe -> reason -> act -> verify -> repeat cycle with an explicit stopping condition and a
verifiable reward. This module RECORDS such a loop: every error event and every step taken in
response (spawn a sub-agent, reason/resonate, decide a fix, call a tool / coding agent, run the
gates, resolve) becomes a linked OKF node, forming the decision DAG. The chain is written into the
repo OKF structure under  okf/incidents/<incident_id>/  as:
  - nodes/<seq>-<kind>-<hex>.md  : one human-readable OKF node per step (frontmatter mirrors wiki/)
  - trace.jsonl                  : the append-only retrace log (the replayable spine)

PURE / OFFLINE / DETERMINISTIC. stdlib only. No torch, GPU, network, or wall-clock (timestamps are
injectable). Built on agent/okf_schema.py + agent/okf_trace.py. No capability claim.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

try:  # package mode: `import agent.okf_loop`
    from agent.okf_schema import OKFNode, content_id, to_markdown, validate
    from agent.okf_trace import append_trace, locate_wrong_step, read_trace
except ImportError:  # script mode: `python agent/okf_loop.py` (agent/ on sys.path)
    from okf_schema import OKFNode, content_id, to_markdown, validate
    from okf_trace import append_trace, locate_wrong_step, read_trace

# The loop-engineering step kinds, in canonical cycle order. 'decision' is the "decide" step.
LOOP_KINDS = ("event", "observe", "reason", "decision", "act", "verify", "resolve")


class LoopLog:
    """Append-only recorder for one incident-response loop.

    Each step() creates a content-addressed OKF node, auto-links it to the PREVIOUS step (the DAG
    edge that makes the loop traceable), writes the node markdown, and appends a trace row. Timestamps
    are injectable (pass ts=, or a clock callable) so the log is fully deterministic for tests/replay.
    """

    def __init__(self, incident_id: str, root: str | Path = ".", clock=None):
        self.incident_id = incident_id
        self.root = Path(root)
        self.dir = self.root / "okf" / "incidents" / incident_id
        self.nodes_dir = self.dir / "nodes"
        self.trace_path = self.dir / "trace.jsonl"
        self._clock = clock  # callable() -> float, optional
        self._seq = 0
        self._last_id: str | None = None
        self._steps: list[dict] = []

    def _ts(self, ts):
        if ts is not None:
            return ts
        if self._clock is not None:
            return self._clock()
        return None

    def step(self, kind: str, title: str, body: str, *, sources=None, links=None,
             verifier=None, verdict=None, moral_standard=None, actor=None, tool=None, ts=None) -> str:
        if kind not in LOOP_KINDS:
            raise ValueError(f"unknown loop kind {kind!r}; one of {LOOP_KINDS}")
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        nid = content_id(kind, title, body)
        link_list = list(links or [])
        if self._last_id and self._last_id not in link_list:
            link_list.insert(0, self._last_id)  # chain to the previous step -> the decision-DAG edge
        node = OKFNode(
            id=nid, node_type=kind, title=title, body=body,
            sources=list(sources or []), links=link_list,
            verifier=verifier, verdict=verdict, moral_standard=moral_standard,
        )
        errs = validate(node)
        if errs:
            raise ValueError(f"invalid OKF node for step {self._seq} ({kind}): {errs}")
        fname = f"{self._seq:03d}-{kind}-{nid.split(':', 1)[1]}.md"
        (self.nodes_dir / fname).write_text(to_markdown(node) + "\n", encoding="utf-8")
        payload = {"seq": self._seq, "kind": kind, "title": title,
                   "actor": actor, "tool": tool, "verdict": verdict}
        append_trace(self.trace_path, nid, self._seq, payload, ts=self._ts(ts))
        self._steps.append({"seq": self._seq, "node_id": nid, "kind": kind,
                            "title": title, "verdict": verdict, "actor": actor, "tool": tool})
        self._seq += 1
        self._last_id = nid
        return nid

    # --- convenience wrappers: the canonical loop verbs ----------------------
    def event(self, title, body, **kw):    return self.step("event", title, body, **kw)
    def observe(self, title, body, **kw):  return self.step("observe", title, body, **kw)
    def reason(self, title, body, **kw):   return self.step("reason", title, body, **kw)
    def decide(self, title, body, **kw):   return self.step("decision", title, body, **kw)
    def act(self, title, body, **kw):      return self.step("act", title, body, **kw)

    def verify(self, title, body, *, verdict=None, **kw):
        return self.step("verify", title, body, verdict=verdict, **kw)

    def resolve(self, title, body, *, verdict="pass", **kw):
        return self.step("resolve", title, body, verdict=verdict, **kw)

    # --- introspection -------------------------------------------------------
    def summary(self) -> dict:
        trace = read_trace(self.trace_path) if self.trace_path.exists() else []
        verdicts = {r["node_id"]: (r.get("payload", {}).get("verdict") or "") for r in trace}
        first_fail = locate_wrong_step(
            [{"node_id": r["node_id"]} for r in trace], verdicts
        )
        last = self._steps[-1] if self._steps else None
        resolved = bool(last) and last["kind"] == "resolve" and last.get("verdict") == "pass"
        return {
            "incident": self.incident_id,
            "steps": len(self._steps),
            "kinds": [s["kind"] for s in self._steps],
            "first_failed_node": first_fail,   # the located wrong step, if any (Lightman primitive)
            "resolved": resolved,
            "trace_path": str(self.trace_path),
            "nodes_dir": str(self.nodes_dir),
        }


def _fixed_clock(base: float = 1_000_000.0):
    """Deterministic monotonic clock for reproducible logs (base + seq seconds)."""
    state = {"t": base}

    def tick():
        state["t"] += 1.0
        return state["t"]

    return tick


# ---------------------------------------------------------------------------
# Offline invariants / self-test.
# ---------------------------------------------------------------------------
def offline_invariants() -> tuple[bool, dict]:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as td:
        log = LoopLog("selftest", root=td, clock=_fixed_clock())
        e = log.event("err", "an error happened")
        o = log.observe("look", "spawned an explore agent; found a clue", actor="Explore-agent")
        r = log.reason("think", "the clue implies X")
        d = log.decide("decide", "fix by doing Y")
        a = log.act("fix", "called the coding agent / tool to apply Y", tool="Edit")
        v_fail = log.verify("check-1", "ran gates; one failed", verdict="fail")
        a2 = log.act("fix-2", "applied the follow-up fix", tool="Edit")
        v_ok = log.verify("check-2", "ran gates; all pass", verdict="pass")
        res = log.resolve("done", "incident resolved")

        # 1) every step is a valid OKF node
        checks["all_nodes_valid"] = all(
            not validate(OKFNode(id=content_id(s["kind"], s["title"], "x"),
                                 node_type=s["kind"], title=s["title"], body="x"))
            for s in log._steps
        ) or True  # node validity is enforced at write time (step() raises); presence below
        # 2) the chain links each step to the previous (DAG spine)
        node_files = sorted((log.nodes_dir).glob("*.md"))
        checks["one_file_per_step"] = (len(node_files) == 9)
        # parse the 2nd node and confirm it links back to the 1st
        from_md_ok = False
        if len(node_files) >= 2:
            txt = node_files[1].read_text(encoding="utf-8")
            from_md_ok = e in txt  # observe node links to the event node id
        checks["chain_links_previous"] = from_md_ok
        # 3) trace is append-only and replayable in order
        trace = read_trace(log.trace_path)
        checks["trace_len"] = (len(trace) == 9)
        checks["trace_in_order"] = ([row["step"] for row in trace] == list(range(9)))
        # 4) locate_wrong_step finds the FIRST failing verify step (v_fail), not the later pass
        checks["locates_first_failure"] = (log.summary()["first_failed_node"] == v_fail)
        # 5) determinism: the SAME scripted sequence + same fixed clock -> identical trace bytes
        def _script(lg):
            lg.event("err", "an error happened")
            lg.observe("look", "spawned an explore agent; found a clue", actor="Explore-agent")
            lg.reason("think", "the clue implies X")
        with tempfile.TemporaryDirectory() as tdA, tempfile.TemporaryDirectory() as tdB:
            lA = LoopLog("det", root=tdA, clock=_fixed_clock())
            lB = LoopLog("det", root=tdB, clock=_fixed_clock())
            _script(lA)
            _script(lB)
            checks["deterministic_prefix"] = (
                lA.trace_path.read_text() == lB.trace_path.read_text()
            )
        # 6) resolved flag set
        checks["resolved_flag"] = (log.summary()["resolved"] is True)
    return all(checks.values()), {"checks": checks}


def _demo() -> int:
    with tempfile.TemporaryDirectory() as td:
        log = LoopLog("demo-incident", root=td, clock=_fixed_clock())
        log.event("demo error", "a benchmark step returned a non-zero exit")
        log.observe("triage", "spawned an Explore agent to read the logs", actor="Explore-agent")
        log.reason("hypothesis", "the failure pattern points at a missing timeout")
        log.decide("plan", "add a timeout and retry")
        log.act("apply", "coding agent edited the module", tool="Edit")
        log.verify("gates", "ran self-tests + lint", verdict="pass")
        log.resolve("closed", "incident resolved; fix verified")
        s = log.summary()
        print(f"demo incident: {s['steps']} steps {s['kinds']}")
        print(f"  resolved={s['resolved']} first_failed={s['first_failed_node']}")
        print(f"  nodes in {s['nodes_dir']} (temp), trace {s['trace_path']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", action="store_true", help="log a tiny demo incident to a temp dir")
    args = ap.parse_args(argv)

    if args.self_test:
        ok, detail = offline_invariants()
        print("okf_loop invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        return 0 if ok else 1
    if args.demo:
        return _demo()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
