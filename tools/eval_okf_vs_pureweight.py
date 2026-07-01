#!/usr/bin/env python3
# PLANNING/HARNESS ONLY - no capability claim; canClaimAGI stays false.
"""OFFLINE two-benchmark harness: OKF-integrated training (Arm A) vs pure-weight control (Arm B).

PURE / OFFLINE / DETERMINISTIC. stdlib only. No torch, no GPU, no network, no real training.

This harness defines 5 metrics as PURE functions over fixture traces and produces an A/B table.

Predicted result (for the record, NOT a claim of capability):
    Arm A (OKF-integrated) wins on traceability + editability + correction-cost: a wrong step is an
    exact, addressable node, and a correction is an O(1) node edit. Arm A may TIE or LOSE on raw
    accuracy / quality, which the weight-only control (Arm B) can match or exceed. The thesis being
    measured here is editability and exact-error-location, not raw accuracy.

It CONSUMES agent/okf_trace.py (locate_wrong_step) and agent/okf_schema.py per the shared OKF API
contract. If those modules are not yet present, a contract-faithful local fallback for
locate_wrong_step is used so this harness remains self-testable in isolation; the fallback mirrors
the contract exactly: FIRST node_id in trace order whose verdict == 'fail'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- CONSUME the shared OKF API contract -------------------------------------
# locate_wrong_step is the Lightman "exact error location" primitive from agent/okf_trace.py.
# Put repo-root on sys.path so `from agent.okf_trace import ...` binds the REAL module in CLI mode
# (run as `python tools/eval_okf_vs_pureweight.py`, sys.path[0] is tools/, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_OKF_TRACE_AVAILABLE = True
try:  # pragma: no cover - exercised by environment, not by the deterministic self-test
    from agent.okf_trace import locate_wrong_step  # type: ignore
except Exception:  # noqa: BLE001 - the agent module may not be written yet
    _OKF_TRACE_AVAILABLE = False

    def locate_wrong_step(trace, verdicts):  # type: ignore
        """Contract-faithful fallback (see agent/okf_trace.py contract).

        Returns the FIRST node_id in trace order whose verdict == 'fail', else None.
        Accepts a trace as a list of dict rows (each with 'node_id') OR a list of node_id strings.
        """
        for row in trace:
            node_id = row.get("node_id") if isinstance(row, dict) else row
            if node_id is not None and verdicts.get(node_id) == "fail":
                return node_id
        return None


# --- constants modelling the two arms ----------------------------------------
# Pure-weight correction means a fixed retrain pass; OKF correction is a single node edit.
RETRAIN_COST = 1000.0       # fixed, abstract "retrain a weight-only model" cost (Arm B)
OKF_NODE_EDIT_COST = 1.0    # O(1) single-node edit cost (Arm A)
QUALITY_STUB_A = 0.82       # deterministic placeholder; Arm A may tie/lose on raw accuracy
QUALITY_STUB_B = 0.84       # control can match or exceed on raw accuracy


# =============================================================================
# The 5 metrics — PURE functions over fixture traces.
# =============================================================================
def traceability_score(traces, ground_truth_wrong_steps):
    """Fraction of traces where locate_wrong_step matches the seeded wrong step.

    traces: list of dicts each with 'steps' (list[node_id]) and 'verdicts' (dict node_id->verdict).
    ground_truth_wrong_steps: list[str] aligned with traces (the seeded wrong node_id per trace).
    """
    if not traces:
        return 0.0
    hits = 0
    for tr, gt in zip(traces, ground_truth_wrong_steps):
        located = locate_wrong_step(tr["steps"], tr["verdicts"])
        if located == gt:
            hits += 1
    return hits / len(traces)


def correction_cost_ratio(arm):
    """Ratio of OKF-edit cost to pure-weight retrain cost for the given arm.

    Arm A (okf): a single O(1) node edit -> OKF_NODE_EDIT_COST / RETRAIN_COST (<< 1, cheaper).
    Arm B (pureweight): a full retrain -> RETRAIN_COST / RETRAIN_COST == 1.0 (baseline).
    """
    key = _arm_key(arm)
    if key == "A":
        return OKF_NODE_EDIT_COST / RETRAIN_COST
    return RETRAIN_COST / RETRAIN_COST


def path_efficiency(trace):
    """steps/tools count for a trace: total addressable actions (steps + tools).

    Monotonic in trace length: a longer trace (more steps and/or tools) yields a larger count.
    """
    steps = len(trace.get("steps", []))
    tools = len(trace.get("tools", []))
    return steps + tools


def forgetting_proxy(before, after):
    """Deterministic stub: fraction of UNRELATED nodes whose verdict changed.

    before/after: dict node_id -> verdict. Considers only nodes present in BOTH snapshots.
    A correction that disturbs unrelated nodes (catastrophic forgetting) raises this proxy.
    """
    shared = set(before) & set(after)
    if not shared:
        return 0.0
    changed = sum(1 for n in shared if before[n] != after[n])
    return changed / len(shared)


def quality_stub(arm):
    """Deterministic placeholder in [0,1]. NOT a capability measurement.

    Encodes the predicted result that the weight-only control may match/exceed raw accuracy.
    """
    key = _arm_key(arm)
    return QUALITY_STUB_A if key == "A" else QUALITY_STUB_B


# =============================================================================
# helpers
# =============================================================================
def _arm_key(arm):
    s = str(arm).strip().lower()
    if s in ("a", "okf", "okf-integrated", "arm_a", "arma"):
        return "A"
    if s in ("b", "pureweight", "pure-weight", "pure_weight", "control", "arm_b", "armb"):
        return "B"
    return "B"  # default to control


def load_fixture(path):
    """Load a jsonl fixture into a list of trace dicts (deterministic order = file order)."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def default_fixture_path():
    return Path(__file__).resolve().parent.parent / "eval" / "okf" / "fixture_v1.jsonl"


# =============================================================================
# compare() — the A/B table across all 5 metrics.
# =============================================================================
def compare(fixture):
    """Produce the A/B table dict across all 5 metrics.

    fixture: either a path (str/Path) to a jsonl fixture, or an already-loaded list of trace dicts.
    Returns a deterministic nested dict: {metric: {"A": value, "B": value}, ...} plus winner notes.
    """
    if isinstance(fixture, (str, Path)):
        traces = load_fixture(fixture)
    else:
        traces = list(fixture)

    ground_truth = [tr["wrong_step"] for tr in traces]

    # Arm A consumes the OKF structure (steps + verdicts) -> exact location works.
    trace_a = traceability_score(traces, ground_truth)
    # Arm B is pure-weight: no addressable node structure, so locate_wrong_step has nothing to
    # bind to -> empty verdicts -> 0.0 traceability (deterministic model of "weights are opaque").
    opaque = [{"steps": tr["steps"], "verdicts": {}} for tr in traces]
    trace_b = traceability_score(opaque, ground_truth)

    # forgetting proxy: model a single correction.
    # Arm A edits exactly one node -> only that node's verdict flips, no unrelated drift -> 0.0.
    # Arm B retrains -> a deterministic fraction of unrelated nodes drift.
    before_a, after_a = _correction_snapshots(traces, arm="A")
    before_b, after_b = _correction_snapshots(traces, arm="B")
    forget_a = forgetting_proxy(before_a, after_a)
    forget_b = forgetting_proxy(before_b, after_b)

    # path efficiency: total addressable actions across all traces (deterministic).
    pe_a = sum(path_efficiency(tr) for tr in traces)
    pe_b = pe_a  # same underlying tasks; structure differs, not the action count

    table = {
        "traceability_score": {"A": trace_a, "B": trace_b},
        "correction_cost_ratio": {
            "A": correction_cost_ratio("A"),
            "B": correction_cost_ratio("B"),
        },
        "path_efficiency": {"A": pe_a, "B": pe_b},
        "forgetting_proxy": {"A": forget_a, "B": forget_b},
        "quality_stub": {"A": quality_stub("A"), "B": quality_stub("B")},
    }
    table["_winner"] = {
        "traceability_score": "A",        # Arm A locates exact error; Arm B cannot
        "correction_cost_ratio": "A",     # O(1) node edit beats full retrain
        "path_efficiency": "tie",
        "forgetting_proxy": "A",          # node edit disturbs nothing unrelated
        "quality_stub": "B",              # control may match/exceed raw accuracy
    }
    table["_n_traces"] = len(traces)
    table["_okf_trace_available"] = _OKF_TRACE_AVAILABLE
    return table


def _correction_snapshots(traces, arm):
    """Deterministic before/after verdict snapshots modelling a single correction.

    Both arms start from the union of all trace verdicts. The fix flips the FIRST 'fail' node to
    'pass'. Arm A (node edit) touches only that node. Arm B (retrain) additionally flips a
    deterministic subset of unrelated nodes (simulated catastrophic forgetting).
    """
    before = {}
    for tr in traces:
        before.update(tr["verdicts"])
    after = dict(before)

    # the one intended fix: first failing node -> pass
    fail_nodes = [n for n, v in sorted(before.items()) if v == "fail"]
    if fail_nodes:
        after[fail_nodes[0]] = "pass"

    if _arm_key(arm) == "B":
        # retrain drifts unrelated nodes: flip every 3rd non-target node's verdict deterministically.
        unrelated = [n for n in sorted(before) if (not fail_nodes or n != fail_nodes[0])]
        for i, n in enumerate(unrelated):
            if i % 3 == 0:
                after[n] = "fail" if before[n] != "fail" else "pass"
    return before, after


# =============================================================================
# offline_invariants() — deterministic checks; no wall-clock, no network.
# =============================================================================
def offline_invariants():
    """Return (all_ok, details) over the harness's own invariants."""
    checks = {}
    path = default_fixture_path()
    traces = load_fixture(path)
    gt = [tr["wrong_step"] for tr in traces]

    # 1) traceability finds every seeded wrong step in the fixture
    checks["traceability_finds_seeded"] = (
        traceability_score(traces, gt) == 1.0
    )

    # 2) Arm A correction is cheaper than retrain (ratio < 1), Arm B == 1.0
    checks["correction_cost_arm_a_cheaper"] = (
        correction_cost_ratio("A") < 1.0 and correction_cost_ratio("B") == 1.0
    )

    # 3) path_efficiency is monotonic: a strictly longer trace yields a strictly larger value
    short = {"steps": ["s1"], "tools": ["t1"]}
    long = {"steps": ["s1", "s2", "s3"], "tools": ["t1", "t2", "t3"]}
    checks["path_efficiency_monotonic"] = (
        path_efficiency(long) > path_efficiency(short)
    )

    # 4) compare() returns all 5 metrics
    table = compare(traces)
    metrics = {
        "traceability_score",
        "correction_cost_ratio",
        "path_efficiency",
        "forgetting_proxy",
        "quality_stub",
    }
    checks["compare_has_all_5_metrics"] = metrics.issubset(set(table))

    # 5) determinism: same fixture -> identical table twice
    checks["determinism"] = compare(traces) == compare(traces)

    # 6) forgetting proxy: Arm A (node edit) never disturbs more unrelated nodes than Arm B retrain
    checks["forgetting_arm_a_le_arm_b"] = (
        table["forgetting_proxy"]["A"] <= table["forgetting_proxy"]["B"]
    )

    all_ok = all(checks.values())
    return all_ok, checks


# =============================================================================
# CLI
# =============================================================================
def _print_table(table):
    print("OKF (Arm A) vs Pure-Weight (Arm B) — OFFLINE A/B table")
    print("PLANNING/HARNESS ONLY - no capability claim; canClaimAGI stays false.")
    if not table.get("_okf_trace_available", True):
        print("NOTE: agent.okf_trace not importable; used contract-faithful local fallback.")
    print(f"traces: {table['_n_traces']}")
    print(f"{'metric':<24}{'Arm A (OKF)':>16}{'Arm B (weights)':>18}{'winner':>10}")
    print("-" * 68)
    order = [
        "traceability_score",
        "correction_cost_ratio",
        "path_efficiency",
        "forgetting_proxy",
        "quality_stub",
    ]
    for m in order:
        a = table[m]["A"]
        b = table[m]["B"]
        w = table["_winner"][m]
        a_s = f"{a:.4f}" if isinstance(a, float) else str(a)
        b_s = f"{b:.4f}" if isinstance(b, float) else str(b)
        print(f"{m:<24}{a_s:>16}{b_s:>18}{w:>10}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="OFFLINE OKF vs pure-weight harness")
    parser.add_argument("--self-test", action="store_true", help="run invariants, print PASS/FAIL")
    parser.add_argument("--report", action="store_true", help="print the A/B table")
    parser.add_argument("--fixture", default=None, help="path to a jsonl fixture")
    args = parser.parse_args(argv)

    if args.self_test:
        ok, checks = offline_invariants()
        for name, passed in checks.items():
            print(f"{'PASS' if passed else 'FAIL'}  {name}")
        print(f"{'PASS' if ok else 'FAIL'}  ALL")
        return 0 if ok else 1

    if args.report:
        path = args.fixture or default_fixture_path()
        _print_table(compare(path))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
