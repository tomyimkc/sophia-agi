#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run a proposer over the difficulty ladder and report per-tier outcomes.

Each ladder item (``data/math_physics_ladder.jsonl``) is solved via
``agent.math_physics_solver`` and machine-verified. Per tier we report the
outcome mix — verified-correct / rejected / abstained — and the mean
Verified-Step Coverage (VSC), the data the dashboard renders as stacked bars.

Default ``--proposer answer-only`` is a DETERMINISTIC PLUMBING BASELINE (it feeds
the gold back as the answer), so its "verified-correct" counts validate the
harness end-to-end and are **illustrative only**, NOT a capability claim — exactly
like the GSM8K harness-validation row in RESULTS.md. The research-frontier tier
(L6, no gold) correctly ABSTAINS even under this baseline, demonstrating
abstention integrity. ``--proposer model`` runs a real backend (needs an API key)
and is the actual reasoning eval.

    python tools/run_ladder_eval.py                 # answer-only baseline, summary
    python tools/run_ladder_eval.py --json
    python tools/run_ladder_eval.py --write         # write artifact for the dashboard
    python tools/run_ladder_eval.py --proposer model
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.math_physics_solver import solve_problem  # noqa: E402

LADDER = ROOT / "data" / "math_physics_ladder.jsonl"
ARTIFACT = ROOT / "agi-proof" / "benchmark-results" / "ladder-eval.json"

_OUTCOME = {"accepted": "verifiedCorrect", "rejected": "rejected", "abstain": "abstained"}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(proposer: str = "answer-only", ladder_path: Path = LADDER) -> dict:
    items = _load(ladder_path)
    per_tier: dict[int, dict] = defaultdict(
        lambda: {"verifiedCorrect": 0, "rejected": 0, "abstained": 0, "vscSum": 0.0, "n": 0, "tierName": ""}
    )
    cases: list[dict] = []
    for it in items:
        g = solve_problem(
            it["prompt"], gold=it.get("gold"), domain=it.get("domain", "math"), proposer=proposer,
        )
        tier = int(it["tier"])
        bucket = per_tier[tier]
        bucket["tierName"] = it.get("tierName", "")
        bucket[_OUTCOME[g.verdict]] += 1
        bucket["vscSum"] += g.vsc
        bucket["n"] += 1
        cases.append({"id": it["id"], "tier": tier, "domain": it.get("domain"),
                      "verdict": g.verdict, "vsc": round(g.vsc, 4), "target": it.get("target")})
    tiers = []
    for tier in sorted(per_tier):
        b = per_tier[tier]
        tiers.append({
            "tier": tier, "tierName": b["tierName"], "n": b["n"],
            "verifiedCorrect": b["verifiedCorrect"], "rejected": b["rejected"],
            "abstained": b["abstained"], "meanVSC": round(b["vscSum"] / b["n"], 4) if b["n"] else 0.0,
        })
    return {
        "benchmark": "math_physics_ladder",
        "proposer": proposer,
        "illustrative": proposer == "answer-only",
        "n": len(items),
        "tiers": tiers,
        "cases": cases,
        "note": ("answer-only proposer is a deterministic harness-validation baseline (feeds gold back); "
                 "verified-correct counts are illustrative, NOT a capability claim. L6 abstains by design."
                 if proposer == "answer-only" else
                 "model proposer: real reasoning eval; report under the no-overclaim gate with >=3 seeds."),
        "canClaimAGI": False,
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposer", choices=["answer-only", "model"], default="answer-only")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    result = run(args.proposer)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"ladder eval — proposer={result['proposer']} N={result['n']}"
              + ("  [illustrative baseline]" if result["illustrative"] else ""))
        print(f"  {'tier':<28} {'vc':>3} {'rej':>4} {'abs':>4}  meanVSC")
        for t in result["tiers"]:
            label = f"L{t['tier']} {t['tierName']}"
            print(f"  {label:<28} {t['verifiedCorrect']:>3} {t['rejected']:>4} {t['abstained']:>4}  {t['meanVSC']:.2f}")
    if args.write:
        ARTIFACT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {ARTIFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
