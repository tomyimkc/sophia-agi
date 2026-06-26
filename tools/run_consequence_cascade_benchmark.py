#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the Consequence-Cascade benchmark (synthetic OKF retraction pack).

This benchmark tunes the ConsequenceGate's ``flipSeverityEscalate`` threshold —
the fraction of the belief graph a retraction must orphan before the gate forces
``escalate`` (stronger process) rather than ``allow``. It does two things:

1. **Per-case verdict check.** For each synthetic OKF graph + retraction target in
   the pack, run ``agent.consequence_gate.simulate_cascade`` and compare the
   returned verdict (escalate|allow|abstain) and flip-severity to the case's
   structurally-derived ground-truth label.
2. **Threshold sweep.** Re-classify every case at each candidate threshold and
   pick the value with maximum verdict-accuracy (ties broken toward the value
   with the greatest margin from any boundary-case severity — the most robust
   separator, not just any separator). This replaces the previous hand-picked
   ``0.15`` placeholder with a data-derived value.

Deterministic, offline, pure-stdlib over synthetic graphs (no model, no network).
The ground-truth label is graph STRUCTURE (what fraction a retraction orphans),
never a model judgement, so the gate is never grading itself.

Reproduce: ``python tools/run_consequence_cascade_benchmark.py``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from agent.consequence_gate import simulate_cascade  # noqa: E402
from agent.public_sanitize import sanitize_public_artifact  # noqa: E402
from okf.page import Page  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "consequence_cascade" / "consequence_cascade_40_v1.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "consequence-cascade.public-report.json"
# Candidate thresholds swept. Span well below and above the expected discriminator
# zone (~0.15) so the optimum is interior, not at a boundary of the candidate set.
SWEEP_CANDIDATES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def graph_from_case(case: dict[str, Any]):
    """Build an OKF graph from a case's ``graph.nodes`` declaration.

    Each node is ``{"id": <slug>, "derivesFrom": [<slug>...]}``. All nodes are
    ``concept`` pages with ``consensus`` author confidence so the severity signal
    is purely structural (no confidence-rank noise). Ids MUST be lowercase slugs
    (``okf.wikilinks.normalize_target`` lowercases on resolve; mixed-case ids
    would silently fail to resolve and report ``abstain``).
    """
    pages = [
        Page(
            path=Path(f"{node['id']}.md"),
            meta={
                "id": node["id"],
                "pageType": "concept",
                "derivesFrom": list(node.get("derivesFrom", [])),
                "authorConfidence": "consensus",
            },
        )
        for node in case["graph"]["nodes"]
    ]
    return okf.build_graph(pages)


def _verdict_for(found: bool, severity: float, threshold: float) -> str:
    """Re-classify a single retraction at a given threshold (mirrors simulate_cascade)."""
    if not found:
        return "abstain"
    return "escalate" if severity >= threshold else "allow"


def _raw_severity(graph, rep) -> float:
    """Recompute the UNROUNDED flip severity exactly as the live gate does.

    ``simulate_cascade`` compares the threshold against ``len(abstain) / max(1, n)``
    (unrounded) and only rounds for reporting. The sweep must reclassify against
    that same unrounded value, not ``rep.flipSeverity`` (which is rounded to 4 dp),
    or the sweep could disagree with the gate near a boundary.
    """
    if not rep.found:
        return 0.0
    return len(rep.abstainSet) / max(1, len(graph.nodes))


def run_case(case: dict[str, Any], *, threshold: float | None = None) -> dict[str, Any]:
    """Run one case at ``threshold`` (or the live config value if None).

    Returns the per-case row. ``verdictOk`` compares the re-classified verdict at
    ``threshold`` to the case's ``expectVerdict``; ``flipSeverityBandOk`` checks
    the computed severity falls in the structurally-expected band. For abstain
    cases the target must NOT resolve (``found`` must be False) — a ghost target
    that unexpectedly resolves is pack drift and must fail the band check, not be
    silently passed.
    """
    graph = graph_from_case(case)
    rep = simulate_cascade(graph, case["move"])
    raw_sev = _raw_severity(graph, rep)
    # When a threshold is supplied (sweep mode), re-classify from the RAW severity
    # at that threshold (matching the live gate's unrounded comparison), not the
    # rounded rep.flipSeverity.
    if threshold is not None:
        verdict = _verdict_for(rep.found, raw_sev, threshold)
    else:
        verdict = rep.verdict
    expected = case["expectVerdict"]
    lo, hi = case["expectFlipSeverityBand"]
    if expected == "abstain":
        # An abstain case is well-formed only if the target genuinely does NOT
        # resolve. A ghost target that resolves is drift; fail the band check.
        band_ok = not rep.found
    else:
        band_ok = rep.found and lo <= raw_sev <= hi
    return {
        "id": case["id"],
        "caseType": case["caseType"],
        "expectVerdict": expected,
        "gotVerdict": verdict,
        "found": rep.found,
        "flipSeverity": rep.flipSeverity,  # rounded, for display
        "rawFlipSeverity": round(raw_sev, 6),  # the value reclassification actually used
        "expectFlipSeverityBand": list(case["expectFlipSeverityBand"]),
        "verdictOk": verdict == expected,
        "flipSeverityBandOk": bool(band_ok),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(bool(r[key]) for r in rows) / len(rows), 4) if rows else 0.0


def sweep_threshold(cases: list[dict[str, Any]], candidates: list[float] | None = None) -> dict[str, Any]:
    """Sweep ``flipSeverityEscalate`` over candidate values; pick the optimum.

    The optimum = max verdict-accuracy, computed by reclassifying each case at
    the candidate threshold against its RAW (unrounded) severity — matching the
    live gate's comparison exactly, so the sweep never disagrees with the gate
    near a boundary.

    Ties are broken, in order, by: (1) greater margin (min distance from the
    threshold to any case severity — a wide clean gap beats a knife-edge), then
    (2) candidate declaration order (deterministic, NO bias toward any particular
    value such as the previous placeholder). The recommendation is therefore
    purely data-derived from the pack; there is no hidden preference.
    """
    cands = candidates or SWEEP_CANDIDATES
    # Pre-compute each case's (found, raw_severity, expected) once. RAW severity
    # matches the gate's unrounded len(abstain)/n comparison.
    facts: list[tuple[bool, float, str]] = []
    for c in cases:
        graph = graph_from_case(c)
        rep = simulate_cascade(graph, c["move"])
        facts.append((rep.found, _raw_severity(graph, rep), c["expectVerdict"]))
    # All resolvable-case severities (the values a threshold sits between).
    severities = sorted({round(s, 6) for found, s, _ in facts if found})

    table = []
    best = None
    for idx, t in enumerate(cands):
        correct = sum(1 for found, s, exp in facts if _verdict_for(found, s, t) == exp)
        acc = correct / len(facts) if facts else 0.0
        # margin = distance from t to the nearest case severity on either side
        # (the half-width of the clean band around t). Larger = more robust.
        nearest = min((abs(t - s) for s in severities), default=0.0)
        entry = {"threshold": t, "verdictAccuracy": round(acc, 4), "correct": correct,
                 "n": len(facts), "marginToNearestSeverity": round(nearest, 4)}
        table.append(entry)
        # Tie-break key: (accuracy desc, margin desc, declaration-order asc).
        # Deliberately NO bias toward 0.15 or any other value.
        key = (entry["verdictAccuracy"], entry["marginToNearestSeverity"], -idx)
        if best is None or key > best[0]:
            best = (key, t, entry)
    recommended = best[1] if best else cands[len(cands) // 2]
    return {"candidates": cands, "recommended": recommended, "table": table}


def run(inp: str | Path = DEFAULT_IN, out: str | Path = DEFAULT_OUT, *,
        threshold: float | None = None, candidates: list[float] | None = None) -> dict[str, Any]:
    """Run the benchmark. ``threshold=None`` uses the live config value for the
    per-case rows; the sweep is always reported regardless."""
    cases = load_jsonl(inp)
    rows = [run_case(c, threshold=threshold) for c in cases]
    sweep = sweep_threshold(cases, candidates)
    report = {
        "schema": "sophia.consequence_cascade_benchmark.v1",
        "benchmark": "Consequence-Cascade synthetic pack (38 cases)",
        "candidateOnly": True,
        "level3Evidence": False,
        "validated": False,
        "claimBoundary": (
            "Deterministic structural benchmark over synthetic OKF graphs. The "
            "ground-truth label is graph structure (what fraction a retraction "
            "orphans), never a model judgement, so the gate is never grading "
            "itself. The recommended threshold is data-derived from SYNTHETIC "
            "graph structure; it is not validated against real belief graphs "
            "and earns level3Evidence only after a real run routes decisions "
            "through the live gate. Proves candidate tuning behavior, not AGI."
        ),
        "thresholdUsedForRows": threshold,  # None => live config value
        "n": len(rows),
        "metrics": {
            "verdictAccuracyRate": _rate(rows, "verdictOk"),
            "flipSeverityBandRate": _rate(rows, "flipSeverityBandOk"),
        },
        "thresholdSweep": sweep,
        "rows": rows,
    }
    # Headline ok: the sweep found an interior optimum (not pinned at a candidate
    # boundary) AND the recommended threshold's accuracy is >= the previous
    # hand-pick 0.15's accuracy AND every case's severity matches its band.
    rec = sweep["recommended"]
    rec_entry = next(e for e in sweep["table"] if e["threshold"] == rec)
    placeholder_entry = next((e for e in sweep["table"] if e["threshold"] == 0.15), None)
    beats_or_matches_placeholder = (
        placeholder_entry is None or rec_entry["verdictAccuracy"] >= placeholder_entry["verdictAccuracy"]
    )
    interior = min(sweep["candidates"]) < rec < max(sweep["candidates"])
    report["ok"] = (
        len(rows) == len(cases)
        and report["metrics"]["flipSeverityBandRate"] == 1.0
        and beats_or_matches_placeholder
        and rec_entry["verdictAccuracy"] >= 0.90
        and interior
    )
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the consequence-cascade benchmark + threshold sweep")
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--threshold", type=float, default=None,
                    help="override flipSeverityEscalate for the per-case rows (default: live config)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the threshold sweep")
    args = ap.parse_args()
    report = run(args.inp, args.out, threshold=args.threshold,
                 candidates=None if not args.no_sweep else [0.15])
    sweep = report["thresholdSweep"]
    print(json.dumps({
        "ok": report["ok"],
        "out": args.out,
        "metrics": report["metrics"],
        "recommendedFlipSeverityEscalate": sweep["recommended"],
        "recommendedAccuracy": next(e["verdictAccuracy"] for e in sweep["table"] if e["threshold"] == sweep["recommended"]),
    }, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
