#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Intelligence-per-parameter / per-byte as a MEASURED axis, not a slogan (T5).

The "reasoning density per active param / per byte" claim is exactly the kind of vibe this
repo turns into a gated number. Given a capability score (passAt1 / judge winrate, with its
sample size n) plus the model's ACTIVE parameter count and its SERVED footprint (post-quant
GB), this computes two efficiency ratios — ``score / active-params`` and ``score / served-GB``
— each carried with a confidence interval, and marks the Pareto-optimal frontier
(maximise score, minimise params and bytes). Pure-Python (reuses tools/eval_stats); offline.

Input: a JSON list of entries, each:
    {"label": "...", "score": 0.81, "scoreMetric": "passAt1", "n": 354,
     "activeParamsB": 4.0, "servedGB": 7.0}

    python tools/build_efficiency_frontier.py --input entries.json
    python tools/build_efficiency_frontier.py --input entries.json --markdown

Claim boundary: these are EFFICIENCY ratios of an already-measured score; the CI is the
score's CI scaled by the (exact, measured) denominator. "More capable per byte" becomes a
claim only when the ratio's CI clears the comparison — candidate_only; canClaimAGI:false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_stats import fixed_n_ci_mean  # noqa: E402

# Proportion-valued metrics get a Bernoulli CI from (score, n); others report no CI here.
_PROPORTION_METRICS = {"passat1", "passAt1", "winrate", "judgewinrate", "accuracy", "acc"}


def _score_ci(score: float, n: int | None, metric: str) -> list[float] | None:
    """CI for the score. For a proportion metric with n, reconstruct the 0/1 vector
    (k=round(score*n) ones) and reuse eval_stats.fixed_n_ci_mean — no new statistics."""
    if not n or n <= 1 or metric.lower() not in {m.lower() for m in _PROPORTION_METRICS}:
        return None
    k = round(score * n)
    vec = [1.0] * k + [0.0] * (n - k)
    lo, hi = fixed_n_ci_mean(vec)
    return [round(max(0.0, lo), 4), round(min(1.0, hi), 4)]


def _ratio_with_ci(score: float, ci: list[float] | None, denom: float) -> dict:
    """score/denom with the CI scaled by the (exact) denominator."""
    if denom <= 0:
        return {"value": None, "ci": None}
    out = {"value": round(score / denom, 4)}
    out["ci"] = [round(ci[0] / denom, 4), round(ci[1] / denom, 4)] if ci else None
    return out


def _dominates(a: dict, b: dict) -> bool:
    """True if a Pareto-dominates b: a is >= on score and <= on both costs, strictly better
    on at least one axis. Missing costs are treated as +inf (never dominating on that axis)."""
    inf = float("inf")
    a_s, b_s = a["score"], b["score"]
    a_p, b_p = a.get("activeParamsB") or inf, b.get("activeParamsB") or inf
    a_g, b_g = a.get("servedGB") or inf, b.get("servedGB") or inf
    no_worse = a_s >= b_s and a_p <= b_p and a_g <= b_g
    strictly = a_s > b_s or a_p < b_p or a_g < b_g
    return no_worse and strictly


def build_frontier(entries: list[dict]) -> dict:
    """Annotate each entry with per-param / per-GB efficiency (+CI) and flag the Pareto set."""
    annotated: list[dict] = []
    for e in entries:
        score = float(e["score"])
        n = e.get("n")
        metric = str(e.get("scoreMetric", "passAt1"))
        ci = _score_ci(score, n, metric)
        params = e.get("activeParamsB")
        gb = e.get("servedGB")
        annotated.append({
            "label": e.get("label", "?"),
            "score": round(score, 4),
            "scoreMetric": metric,
            "n": n,
            "scoreCI": ci,
            "activeParamsB": params,
            "servedGB": gb,
            "perActiveParam": _ratio_with_ci(score, ci, float(params)) if params else {"value": None, "ci": None},
            "perServedGB": _ratio_with_ci(score, ci, float(gb)) if gb else {"value": None, "ci": None},
        })
    pareto = [a["label"] for a in annotated
              if not any(_dominates(b, a) for b in annotated if b is not a)]
    # rank by per-active-param efficiency (None last)
    annotated.sort(key=lambda a: (a["perActiveParam"]["value"] is None,
                                  -(a["perActiveParam"]["value"] or 0.0)))
    return {
        "schema": "sophia.efficiency_frontier.v1",
        "entries": annotated,
        "paretoOptimal": pareto,
        "claimBoundary": "Efficiency ratios of an already-measured score; CI = the score's CI "
                         "scaled by the exact measured denominator. candidate_only; canClaimAGI:false.",
    }


def _markdown(frontier: dict) -> str:
    lines = ["| model | score | score CI | active-B | served-GB | score/B | score/GB | pareto |",
             "|---|---|---|---|---|---|---|---|"]
    pset = set(frontier["paretoOptimal"])
    for a in frontier["entries"]:
        ci = f"[{a['scoreCI'][0]}, {a['scoreCI'][1]}]" if a["scoreCI"] else "—"
        pp = a["perActiveParam"]["value"]
        pg = a["perServedGB"]["value"]
        lines.append(
            f"| {a['label']} | {a['score']} | {ci} | {a['activeParamsB'] or '—'} | "
            f"{a['servedGB'] or '—'} | {pp if pp is not None else '—'} | "
            f"{pg if pg is not None else '—'} | {'★' if a['label'] in pset else ''} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="JSON list of efficiency entries")
    ap.add_argument("--out", type=Path, default=None, help="write the frontier JSON here")
    ap.add_argument("--markdown", action="store_true", help="also print a markdown table")
    args = ap.parse_args(argv)

    entries = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        print("input must be a JSON list of entries", file=sys.stderr)
        return 1
    frontier = build_frontier(entries)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(frontier, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    if args.markdown or not args.out:
        print(_markdown(frontier))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
