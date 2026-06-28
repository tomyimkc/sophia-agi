#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Score saved θ_search generations with N independent families — offline, no GPU.

The multi-seed run (training/swarm_router/multiseed_remote.sh) now exfiltrates the RAW
base/adapter generations in ``raw_generations``. That decouples expensive generation from
cheap scoring: any number of independent judge families can be applied post-hoc — for
free, repeatably — without re-renting a GPU. This is the tool that does it.

Given a run-result JSON containing ``raw_generations`` (``{base:[...], seed0:[...], …}``),
it scores every family in ``search_recall.SCORER_FAMILIES`` (extend that registry — e.g.
add an LLM-judge family — and re-run with zero GPU), reports each family's per-seed and
mean source-discipline delta with a paired bootstrap CI, and Cohen's κ between families.

  python tools/score_generations.py path/to/RESULT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.search_recall import SCORER_FAMILIES, cohens_kappa  # noqa: E402
from provenance_bench.swarm_benchmark import _paired_bootstrap_ci  # noqa: E402


def score(raw: dict, *, families: dict | None = None) -> dict:
    families = families or SCORER_FAMILIES
    base_gens = raw["base"]
    seed_keys = [k for k in raw if k.startswith("seed")]
    out: dict = {"nTraps": len(base_gens), "seeds": seed_keys, "families": {}}
    for fam, scorer in families.items():
        base_hits = [1 if scorer(g) else 0 for g in base_gens]
        per_seed = {}
        pb: list[int] = []
        pa: list[int] = []
        deltas: list[float] = []
        for sk in seed_keys:
            after = [1 if scorer(g) else 0 for g in raw[sk]]
            br = sum(base_hits) / len(base_hits)
            ar = sum(after) / len(after)
            per_seed[sk] = {"before": round(br, 3), "after": round(ar, 3), "delta": round(ar - br, 3)}
            deltas.append(ar - br)
            pb += base_hits
            pa += after
        lo, hi = _paired_bootstrap_ci(pb, pa, iters=4000, seed=0)
        out["families"][fam] = {
            "base_rate": round(sum(base_hits) / len(base_hits), 3),
            "per_seed": per_seed,
            "mean_delta": round(sum(deltas) / len(deltas), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "ci_excludes_zero": bool(lo > 0),
        }
    fams = list(families)
    if len(fams) >= 2:
        all_gens = base_gens + [g for sk in seed_keys for g in raw[sk]]
        labs = {f: [1 if families[f](g) else 0 for g in all_gens] for f in fams[:2]}
        out["kappa_families"] = fams[:2]
        out["kappa_between_families"] = cohens_kappa(labs[fams[0]], labs[fams[1]])
    out["all_families_exclude_zero"] = all(v["ci_excludes_zero"] for v in out["families"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path, help="run-result JSON with a raw_generations field")
    args = ap.parse_args()
    data = json.loads(args.result.read_text())
    raw = data.get("raw_generations")
    if not raw or "base" not in raw:
        print("No raw_generations in result (re-run with the gen-capturing harness).", file=sys.stderr)
        return 2
    report = score(raw)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
