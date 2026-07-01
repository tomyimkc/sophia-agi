#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""W3 — provenance-weighted training + influence hooks (drop-in, fail-closed).

Thesis: the repo already computes a per-SOURCE trust rank (agent.source_ranking.rank_source
-> RankedSource.rank in [0,1]). Everyone else has to ESTIMATE data quality; Sophia already
has a trust label. So: attach each training example's source-trust to its LOSS WEIGHT and
its curriculum ORDER, and expose an influence hook so a bad output can be traced to the
training rows that caused it.

WHAT THIS DOES (runnable offline):
  * for a set of training examples each carrying a `source`, computes the real
    RankedSource.rank via agent.source_ranking and derives a normalized per-example loss
    weight (with a configurable floor so low-trust data is down-weighted, not deleted);
  * emits a curriculum ORDER (high-provenance first) and the weight vector a trainer would
    consume;
  * provides a leave-one-out INFLUENCE STUB: a deterministic, model-free proxy that ranks
    which training sources most affected a given eval item by shared-source overlap, so the
    corrective-loop plumbing is real even before a true influence-function backend is wired.

WHAT THIS DOES NOT DO (honest seam):
  * it does NOT run a real gradient influence function (TracIn / influence functions need
    the model's gradients) and does NOT itself fine-tune. It produces the weights + order +
    attribution scaffold a maintainer feeds to the MLX/LoRA trainer. candidateOnly:true.

Example schema: {"id": str, "text": str, "source": str, "domain": str}
Usage:
  python3 tools/provenance_weighted_training.py --examples ex.jsonl --floor 0.1 --out w.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from agent.source_ranking import rank_source
    _REPO_OK = True
    _IMPORT_ERR = ""
except Exception as e:  # pragma: no cover
    _REPO_OK = False
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def _env_artifact(reason: str) -> dict[str, Any]:
    return {"schema": "sophia.provenance_weighting.v1", "ok": False, "reason": reason,
            "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False}


def weight_examples(examples: list[dict[str, Any]], *, floor: float = 0.1) -> dict[str, Any]:
    """Attach a loss weight = max(floor, rank) per example, from the REAL rank_source."""
    if not (0.0 <= floor < 1.0):
        return _env_artifact(f"floor must be in [0,1); got {floor}")
    weighted = []
    for ex in examples:
        src = ex.get("source", "")
        rs = rank_source(src)              # real RankedSource(id, rank, tier, reason)
        w = max(floor, float(rs.rank))
        weighted.append({
            "id": ex.get("id"), "source": src, "domain": ex.get("domain"),
            "rank": round(rs.rank, 4), "tier": rs.tier, "weight": round(w, 4),
        })
    return {"weighted": weighted}


def influence_stub(eval_item: dict[str, Any], train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Model-free influence PROXY: rank training rows by (a) same source as the eval item's
    cited source, then (b) source trust. A real TracIn/influence backend replaces this; the
    corrective-loop plumbing (bad output -> implicated training rows -> down-weight) is real.
    """
    cited = str(eval_item.get("source", ""))
    scored = []
    for r in train_rows:
        same = 1 if str(r.get("source", "")) == cited and cited else 0
        rs = rank_source(r.get("source", ""))
        # implicated-ness: shares the eval item's (possibly low-trust) source; low trust +
        # shared source = most likely culprit for a provenance-driven error
        scored.append({"id": r.get("id"), "source": r.get("source"),
                       "sharesEvalSource": bool(same), "sourceRank": round(rs.rank, 4),
                       "suspicion": round(same * (1.0 - rs.rank), 4)})
    scored.sort(key=lambda x: x["suspicion"], reverse=True)
    return scored


def run(examples: list[dict[str, Any]], *, floor: float = 0.1,
        eval_item: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _REPO_OK:
        return _env_artifact(f"repo instruments unavailable ({_IMPORT_ERR}); run with "
                             "PYTHONPATH=. inside the sophia-agi tree")
    if not examples:
        return _env_artifact("no training examples provided (fail-closed)")

    w = weight_examples(examples, floor=floor)
    if not w.get("weighted"):
        return w  # already an env artifact
    weighted = w["weighted"]

    # curriculum: high-provenance first (stable sort by -weight, then id)
    order = sorted(range(len(weighted)),
                   key=lambda i: (-weighted[i]["weight"], str(weighted[i]["id"])))

    ranks = [x["rank"] for x in weighted]
    report = {
        "schema": "sophia.provenance_weighting.v1", "ok": True,
        "n": len(weighted), "floor": floor,
        "weights": weighted,
        "curriculumOrder": [weighted[i]["id"] for i in order],
        "rankStats": {"min": round(min(ranks), 4), "max": round(max(ranks), 4),
                      "mean": round(sum(ranks) / len(ranks), 4)},
        "tierCounts": _count(weighted, "tier"),
        "note": "Weights/order are consumed by the MLX/LoRA trainer (maintainer seam); this "
                "tool does not itself fine-tune. Influence is a model-free proxy until a real "
                "TracIn/influence-function backend is wired — validate against leave-one-out "
                "retraining on a small slice, and watch output-diversity for register collapse.",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
    }
    if eval_item is not None:
        report["influence"] = {
            "evalItem": eval_item.get("id"),
            "implicatedTrainRows": influence_stub(eval_item, examples)[:10],
        }
    return report


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W3 provenance-weighted training")
    ap.add_argument("--examples", required=True, help="JSONL {id,text,source,domain}")
    ap.add_argument("--floor", type=float, default=0.1, help="min loss weight for low-trust data")
    ap.add_argument("--eval-item", default=None, help="optional JSON file: an eval item to trace influence for")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    examples = load_jsonl(Path(args.examples))
    ev = json.loads(Path(args.eval_item).read_text()) if args.eval_item else None
    report = run(examples, floor=args.floor, eval_item=ev)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())