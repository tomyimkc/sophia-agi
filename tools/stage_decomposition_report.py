#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A7 — stage-decomposition reporting + dual-baseline provenance.

Two reporting patterns adopted from Agents-A1's honest side (arXiv 2606.30616):

1. **Stage decomposition** (their Table 4): base -> intermediate -> final,
   every checkpoint through the IDENTICAL harness, with per-benchmark deltas
   and an explicit `regressions` list — a stage that helps the headline while
   silently hurting another suite (their full-domain-SFT HLE -5.8 pattern) is
   surfaced, never averaged away.
2. **Dual baseline provenance** (their tau^2-Bench 81.2-official vs
   32.5-reproduced disclosure): when a published baseline cannot be reproduced
   within tolerance, BOTH numbers are carried and the claim gate must use the
   REPRODUCED one (claims built on numbers we cannot reproduce are not claims).

candidateOnly: this tool formats evidence; it makes no claims and never
decides promotion (that stays with evaluate_update / promote_adapter).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sophia.stage_decomposition.v1"


def build_stage_decomposition(stages: "Sequence[tuple[str, dict[str, float]]]",
                              *, regression_tolerance: float = 0.0) -> "dict[str, Any]":
    """Rows of per-benchmark scores across ordered stages + flagged regressions.

    ``stages`` = [(label, {benchmark: score})] in training order (>=2). A
    regression is any stage-over-previous-stage drop beyond the tolerance on
    any benchmark — reported per (benchmark, stage) pair, fail-visible.
    """
    if len(stages) < 2:
        raise ValueError("stage decomposition needs >=2 stages (base + at least one)")
    labels = [l for l, _ in stages]
    benchmarks = sorted({b for _, m in stages for b in m})
    rows, regressions = [], []
    for b in benchmarks:
        scores = [m.get(b) for _, m in stages]
        deltas = []
        for i in range(1, len(scores)):
            if scores[i] is None or scores[i - 1] is None:
                deltas.append(None)
                continue
            d = round(scores[i] - scores[i - 1], 4)
            deltas.append(d)
            if d < -abs(regression_tolerance):
                regressions.append({"benchmark": b, "stage": labels[i], "delta": d})
        rows.append({"benchmark": b, "scores": scores, "deltas": deltas})
    return {"schema": SCHEMA, "stages": labels, "rows": rows,
            "regressions": regressions, "regressionTolerance": regression_tolerance,
            "candidateOnly": True, "level3Evidence": False,
            "note": "identical-harness stage table; regressions are fail-visible, "
                    "never averaged away. Formats evidence only — no claim."}


def baseline_provenance(*, official: "float | None", reproduced: "float | None",
                        tolerance: float = 1.0, source: str = "") -> "dict[str, Any]":
    """Dual-report a baseline; the claim gate must use the REPRODUCED number."""
    discrepant = (official is not None and reproduced is not None
                  and abs(official - reproduced) > tolerance)
    return {
        "official": official, "reproduced": reproduced, "source": source,
        "discrepant": discrepant, "tolerance": tolerance,
        "useForClaims": reproduced if reproduced is not None else None,
        "note": ("official baseline NOT reproduced within tolerance — both reported, "
                 "claims use the reproduced number" if discrepant else
                 "claims use the reproduced number (fail-closed when absent)"),
    }


def _load_metrics(path: Path, key: "str | None") -> "dict[str, float]":
    data = json.loads(path.read_text(encoding="utf-8"))
    node: Any = data
    for part in (key.split(".") if key else []):
        node = node[part]
    return {str(k): float(v) for k, v in node.items() if isinstance(v, (int, float))}


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="A7 stage-decomposition report")
    ap.add_argument("--stage", action="append", nargs=2, metavar=("LABEL", "REPORT_JSON"),
                    required=True, help="repeat in training order (>=2)")
    ap.add_argument("--metrics-key", default=None,
                    help="dot-path to the flat {benchmark: score} dict inside each report")
    ap.add_argument("--regression-tolerance", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    stages = [(label, _load_metrics(Path(p), args.metrics_key)) for label, p in args.stage]
    result = build_stage_decomposition(stages, regression_tolerance=args.regression_tolerance)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not result["regressions"] else 1  # nonzero = regressions present (visible)


if __name__ == "__main__":
    raise SystemExit(main())
