#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gate compute-budget auditor — the honesty machinery must not starve training.

A blind spot the correctness lenses missed: gates cost wall-clock time and, some of
them, GPU. If the per-PR (fast) lane balloons, or a GPU gate gets scheduled into the
fast lane, the verification apparatus starves the iteration/training loop it exists
to protect. This gate enforces a declared budget.

Two independent FAIL conditions, checked against ``agi-proof/gate-budget.json``:

  1. LANE CEILING: for each lane, the sum of a run's measured seconds for gates
     assigned to that lane must not exceed the lane's ``ceilingSeconds``.
  2. GPU-IN-FAST-LANE: no gate marked ``gpu: true`` (or assigned to a lane whose
     ``gpuAllowed`` is false) may appear in a fast-lane run. GPU gates belong to
     the heavy (on-merge / nightly) lane.

Input: a run-log JSON mapping ``{gate_name: measured_seconds}`` — either a bare
object or ``{"runLog": {...}}``, optionally with ``{"lane": "fast"}`` to say which
lane this run represents (default: infer each gate's own declared lane). A gate in
the run-log that is NOT in the budget config is reported as ``unbudgeted`` and, if
``--strict``, fails the gate (an unbudgeted gate has no declared cost, so it cannot
be admitted to a fast lane honestly).

    python3 tools/gate_cost_budget.py --run-log run.json
    python3 tools/gate_cost_budget.py --run-log run.json --lane fast --json

Exit 0 = within budget (PASS). Exit 1 = over ceiling or GPU-in-fast (FAIL).
Exit 2 = unreadable/missing input. JSON receipt to stdout; prose to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET = ROOT / "agi-proof" / "gate-budget.json"


def load_budget(path: Path) -> dict:
    """Load the gate-budget config."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_log_of(data: object) -> dict[str, float]:
    """Normalize the run-log input into a {gate: seconds} mapping."""
    log = data["runLog"] if isinstance(data, dict) and "runLog" in data else data
    if not isinstance(log, dict):
        raise ValueError("run-log must be an object {gate: seconds} or {'runLog': {...}}")
    out: dict[str, float] = {}
    for k, v in log.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"run-log seconds for {k!r} not a number: {v!r}")
    return out


def audit(run_log: dict[str, float], budget: dict, lane_filter: str | None = None,
          strict: bool = False) -> dict:
    """Check a run-log against declared lane ceilings and the no-GPU-in-fast rule.

    ``lane_filter`` (e.g. "fast") restricts the run to gates whose declared lane is
    that lane (this represents "the fast-lane CI job ran these gates"). When None,
    every gate is grouped under its own declared lane.
    """
    lanes_cfg = budget.get("lanes", {})
    gates_cfg = budget.get("gates", {})

    per_lane_seconds: dict[str, float] = {}
    per_lane_gates: dict[str, list[str]] = {}
    gpu_in_fast: list[str] = []
    unbudgeted: list[str] = []
    considered: dict[str, dict] = {}

    for gate, secs in run_log.items():
        g = gates_cfg.get(gate)
        if g is None:
            unbudgeted.append(gate)
            continue
        gate_lane = g.get("lane", "fast")
        if lane_filter is not None and gate_lane != lane_filter:
            # This gate is not part of the lane we are auditing; skip it.
            continue
        is_gpu = bool(g.get("gpu", False))
        lane_cfg = lanes_cfg.get(gate_lane, {})
        gpu_allowed = bool(lane_cfg.get("gpuAllowed", False))

        per_lane_seconds[gate_lane] = per_lane_seconds.get(gate_lane, 0.0) + secs
        per_lane_gates.setdefault(gate_lane, []).append(gate)
        considered[gate] = {"lane": gate_lane, "seconds": secs, "gpu": is_gpu}

        # A GPU gate in a lane that disallows GPU is the load-bearing violation.
        if is_gpu and not gpu_allowed:
            gpu_in_fast.append(gate)

    # Lane ceiling checks.
    lane_overruns: list[dict] = []
    lane_report: dict[str, dict] = {}
    for lane, total in per_lane_seconds.items():
        ceiling = lanes_cfg.get(lane, {}).get("ceilingSeconds")
        over = ceiling is not None and total > float(ceiling)
        lane_report[lane] = {
            "totalSeconds": round(total, 3),
            "ceilingSeconds": ceiling,
            "gates": per_lane_gates.get(lane, []),
            "over": over,
        }
        if over:
            lane_overruns.append({"lane": lane, "totalSeconds": round(total, 3),
                                  "ceilingSeconds": ceiling})

    fail = bool(lane_overruns) or bool(gpu_in_fast) or (strict and bool(unbudgeted))

    return {
        "gate": "gate_cost_budget",
        "status": "preregistration_only",
        "canClaimAGI": False,
        "laneFilter": lane_filter,
        "strict": strict,
        "lanes": lane_report,
        "gpuInDisallowedLane": gpu_in_fast,
        "laneOverruns": lane_overruns,
        "unbudgetedGates": unbudgeted,
        "consideredGates": considered,
        "fail": fail,
        "go": (not fail),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit gate wall-clock/GPU budget vs CI lane ceilings.")
    ap.add_argument("--run-log", required=True, help="path to run-log JSON {gate: seconds}")
    ap.add_argument("--budget", default=str(DEFAULT_BUDGET), help="path to gate-budget.json")
    ap.add_argument("--lane", default=None, help="restrict audit to gates declared for this lane (e.g. fast)")
    ap.add_argument("--strict", action="store_true", help="fail on any gate absent from the budget config")
    ap.add_argument("--json", action="store_true", help="print only the JSON receipt")
    args = ap.parse_args(argv)

    try:
        budget = load_budget(Path(args.budget))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[gate_cost_budget] cannot read budget: {e}", file=sys.stderr)
        return 2

    try:
        raw = json.loads(Path(args.run_log).read_text(encoding="utf-8"))
        run_log = _run_log_of(raw)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[gate_cost_budget] cannot read run-log: {e}", file=sys.stderr)
        return 2

    receipt = audit(run_log, budget, lane_filter=args.lane, strict=args.strict)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))

    if not args.json:
        if receipt["fail"]:
            bits = []
            if receipt["laneOverruns"]:
                bits.append("lane over ceiling: " + ", ".join(
                    f"{o['lane']} {o['totalSeconds']}s>{o['ceilingSeconds']}s" for o in receipt["laneOverruns"]))
            if receipt["gpuInDisallowedLane"]:
                bits.append("GPU gate in fast/no-GPU lane: " + ", ".join(receipt["gpuInDisallowedLane"]))
            if args.strict and receipt["unbudgetedGates"]:
                bits.append("unbudgeted gates: " + ", ".join(receipt["unbudgetedGates"]))
            print(f"[gate_cost_budget] FAIL: {'; '.join(bits)}", file=sys.stderr)
        else:
            print("[gate_cost_budget] PASS: within lane ceilings and no GPU gate in a no-GPU lane", file=sys.stderr)
    return 1 if receipt["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
