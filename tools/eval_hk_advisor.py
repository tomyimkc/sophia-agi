#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Adversarial HK bilingual advisor eval (Phase 4)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hk_advisor.policy import format_response  # noqa: E402
from agent.hk_advisor.verifier import verify_bilingual_fidelity, verify_citation, verify_substance  # noqa: E402
from agent.steering.stats import bootstrap_diff_ci  # noqa: E402
from provenance_bench.calibration_score import score_answer, score_pack_calibration  # noqa: E402
from provenance_bench.hk_advisor_benchmark import load_cases, verify_manifest  # noqa: E402

REPORT_PATH = ROOT / "agi-proof" / "hk-advisor" / "eval-hk-advisor.public-report.json"


@dataclass
class MockAnswerer:
    condition: str
    seed: int

    def answer(self, case: dict) -> str:
        rng = random.Random(self.seed * 1000 + hash(case["id"]) % 10000)
        lang, trap = case.get("language", "en"), case.get("trap", "none")
        decision, markers = case.get("label", {}).get("decision", ""), case.get("fabricationMarkers") or []
        if self.condition == "base":
            if trap != "none" and rng.random() < 0.35:
                return format_response(f"Under {markers[0] if markers else 'Cap. 712'}, this applies.", language=lang)
            if decision == "answerable" and rng.random() < 0.25:
                return format_response("I cannot confirm.", language=lang)
            if decision == "answerable":
                g, c = case.get("label", {}).get("gold_answer") or "", case.get("label", {}).get("gold_citations") or []
                return format_response(f"{g}{(' ('+c[0]+')') if c else ''}", language=lang)
            return format_response("Uncertain.", language=lang)
        if trap != "none":
            return format_response("I cannot confirm; no reliable authority.", language=lang)
        if decision == "answerable":
            g, c = case.get("label", {}).get("gold_answer") or "", case.get("label", {}).get("gold_citations") or []
            return format_response(f"{g}{(' ('+c[0]+')') if c else ''}", language=lang)
        return format_response("I cannot confirm; unknown.", language=lang)


def _aggregate(scores: list[dict]) -> dict:
    traps, ans = [s for s in scores if s["trap"]], [s for s in scores if s["ans"]]
    n = len(scores) or 1
    return {
        "calibrationScore": round(sum(s["cal"] for s in scores) / n, 4),
        "fabricationRateTraps": round(sum(s["fab"] for s in traps) / len(traps), 4) if traps else None,
        "falseAbstentionRate": round(sum(s["abst"] for s in ans) / len(ans), 4) if ans else None,
        "citationAccuracy": round(sum(s["cite"] for s in ans) / len(ans), 4) if ans else None,
        "usefulAnswerRate": round(sum(s["sub"] for s in ans) / len(ans), 4) if ans else None,
        "bilingualFidelity": round(sum(s["biling"] for s in scores) / n, 4),
    }


def run_eval(*, seeds: list[int], mode: str) -> dict[str, Any]:
    seal = verify_manifest(root=ROOT)
    if not seal["ok"]:
        raise SystemExit("seal failed")
    cases = load_cases()
    pack = {"cases": cases}
    by_seed = {}
    for seed in seeds:
        rb = {c["id"]: MockAnswerer("base", seed).answer(c) for c in cases}
        ra = {c["id"]: MockAnswerer("adapter", seed).answer(c) for c in cases}
        def sc(c, a):
            r = score_answer(a, c)
            return {"cal": r["score"], "fab": r.get("fabricated", False), "abst": r.get("abstained", False),
                    "cite": verify_citation(a, c).passed, "sub": verify_substance(a, c).passed,
                    "biling": verify_bilingual_fidelity(a, c).passed,
                    "trap": c.get("trap") != "none", "ans": c.get("label", {}).get("decision") == "answerable"}
        by_seed[seed] = {"base": _aggregate([sc(c, rb[c["id"]]) for c in cases]),
                         "adapter": _aggregate([sc(c, ra[c["id"]]) for c in cases]),
                         "calBase": score_pack_calibration(pack, rb), "calAdapter": score_pack_calibration(pack, ra)}
    m = lambda c, k: sum(by_seed[s][c][k] or 0 for s in seeds) / len(seeds)
    fab_b = [by_seed[s]["base"]["fabricationRateTraps"] or 0 for s in seeds]
    fab_a = [by_seed[s]["adapter"]["fabricationRateTraps"] or 0 for s in seeds]
    u_b = [by_seed[s]["base"]["usefulAnswerRate"] or 0 for s in seeds]
    u_a = [by_seed[s]["adapter"]["usefulAnswerRate"] or 0 for s in seeds]
    c_b = [by_seed[s]["base"]["calibrationScore"] for s in seeds]
    c_a = [by_seed[s]["adapter"]["calibrationScore"] for s in seeds]
    ud = m("adapter", "usefulAnswerRate") - m("base", "usefulAnswerRate")
    alf = m("adapter", "fabricationRateTraps") < m("base", "fabricationRateTraps")
    return {
        "schema": "sophia.hk_advisor_eval.v1", "generatedAt": datetime.now(timezone.utc).isoformat(),
        "candidateOnly": True, "canClaimAGI": False, "mode": mode, "nCases": len(cases),
        "nSeeds": len(seeds), "seeds": seeds, "benchmarkContentHash": seal["contentHash"],
        "balance": seal["balance"], "bilingualSplit": seal["bilingualSplit"],
        "bySeed": by_seed,
        "aggregate": {"base": {k: round(m("base", k), 4) for k in ("calibrationScore", "fabricationRateTraps",
            "falseAbstentionRate", "citationAccuracy", "usefulAnswerRate", "bilingualFidelity")},
            "adapter": {k: round(m("adapter", k), 4) for k in ("calibrationScore", "fabricationRateTraps",
            "falseAbstentionRate", "citationAccuracy", "usefulAnswerRate", "bilingualFidelity")}},
        "deltas": {
            "adapter_vs_base_fabrication": {"baseMinusAdapter": round(m("base", "fabricationRateTraps") - m("adapter", "fabricationRateTraps"), 4),
                "bootstrap95ci": bootstrap_diff_ci(fab_b, fab_a, seed=0), "adapterLowersFabrication": alf},
            "adapter_vs_base_calibration": {"delta": round(m("adapter", "calibrationScore") - m("base", "calibrationScore"), 4),
                "bootstrap95ci": bootstrap_diff_ci(c_a, c_b, seed=2)},
            "adapter_vs_base_usefulAnswer": {"delta": round(ud, 4), "bootstrap95ci": bootstrap_diff_ci(u_a, u_b, seed=1),
                "noUsefulAnswerLoss": ud >= 0}},
        "claimTemplate": f"On sealed HK advisor benchmark v1 (N={len(cases)}, 3 seeds), mock adapter {'lowers' if alf else 'does not lower'} fabrication on traps. candidateOnly; canClaimAGI:false.",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--mode", choices=["mock", "runpod"], default="mock")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)
    report = run_eval(seeds=[int(x) for x in args.seeds.split(",") if x.strip()], mode=args.mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "aggregate": report["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
