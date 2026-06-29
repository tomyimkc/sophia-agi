#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Label the Sophrosyne external battery with >= 2 INDEPENDENT judge families (pillar 5).

Mirrors tools/label_andreia_battery.py. Two distinct model families — judge A and
judge B — each decide INDEPENDENTLY over the same raw text with the shared
tools/sophrosyne_decision prompt. We report inter-judge Cohen kappa + Gwet AC1 (CIs).
Ground truth for SCORING is the consensus QUADRANT: a case enters the scored set only
when both judges agree on the quadrant ({restrain} / {sustain} / {proportionate} /
{escalate}). Judges run at temperature 0; responses cached so re-runs are free.

    python3 tools/label_sophrosyne_battery.py \
        --judge-a 'ollama:qwen2.5:32b-instruct@http://spark-2f2d:11434/v1' --judge-a-name qwen \
        --judge-b 'vllm:mlx-community/Llama-3.3-70B-Instruct-4bit@http://tommac-studio:8081/v1' --judge-b-name llama
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.model import ModelClient, resolve_config  # noqa: E402
from tools.eval_stats import bootstrap_ci_agreement, cohen_kappa, gwet_ac1  # noqa: E402
from tools.sophrosyne_decision import build_messages, parse_verdict, quadrant_of  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "sophrosyne"
BATTERY_PATH = RESULTS_DIR / "sophrosyne_external_battery.json"
LABELED_PATH = RESULTS_DIR / "sophrosyne_external_battery.labeled.json"
CACHE_PATH = RESULTS_DIR / "sophrosyne_judge_cache.jsonl"

KAPPA_FLOOR = 0.40

# Consensus quadrant -> the canonical optimal measure used by the eval metric.
_QUADRANT_OPTIMAL = {
    "should_restrain": "restrain",
    "should_sustain": "sustain",
    "proportionate": "proportionate",
    "guard": "escalate",
}


def _load_cache() -> dict:
    cache: dict = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[(row["caseId"], row["judge"])] = row
    return cache


def _judge_one(client: ModelClient, judge_name: str, case: dict) -> dict:
    system, user = build_messages(case["text"])
    res = client.generate(system, user)
    verdict = parse_verdict(res.text) if res.ok else None
    return {
        "caseId": case["id"], "judge": judge_name,
        "ok": bool(res.ok), "verdict": verdict,
        "quadrant": quadrant_of(verdict) if verdict else None,
        "raw": (res.text or "")[:200], "error": res.error,
    }


def _run_judge(spec: str, judge_name: str, cases: list, cache: dict, workers: int) -> dict:
    cfg = resolve_config(spec)
    cfg.temperature = 0.0  # deterministic labelling
    cfg.max_tokens = 16
    client = ModelClient(cfg, retries=2)
    todo = [c for c in cases if (c["id"], judge_name) not in cache]
    n_cached = len(cases) - len(todo)
    print(f"  judge {judge_name} ({cfg.model} @ {cfg.base_url}): {n_cached} cached, {len(todo)} to run")
    results: list[dict] = []
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda c: _judge_one(client, judge_name, c), todo))
        with CACHE_PATH.open("a", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in results:
        cache[(r["caseId"], judge_name)] = r
    return {c["id"]: cache[(c["id"], judge_name)] for c in cases}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-a", required=True, help="judge A model spec (family A)")
    ap.add_argument("--judge-b", required=True, help="judge B model spec (family B, != A, != subject)")
    ap.add_argument("--judge-a-name", default="judgeA")
    ap.add_argument("--judge-b-name", default="judgeB")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: label only the first N cases")
    args = ap.parse_args(argv)
    if args.judge_a_name == args.judge_b_name:
        ap.error("judge families must have distinct names")

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    cases = battery["cases"][: args.limit] if args.limit else battery["cases"]
    cache = _load_cache()

    print(f"Labelling {len(cases)} cases with 2 judge families...")
    a = _run_judge(args.judge_a, args.judge_a_name, cases, cache, args.workers)
    b = _run_judge(args.judge_b, args.judge_b_name, cases, cache, args.workers)

    labeled: list[dict] = []
    va, vb, qa, qb = [], [], [], []
    parse_fail = 0
    for c in cases:
        ra, rb = a[c["id"]], b[c["id"]]
        agreed_q = (ra["quadrant"] is not None and ra["quadrant"] == rb["quadrant"])
        row = {
            "id": c["id"], "text": c["text"], "intendedQuadrant": c["intendedQuadrant"],
            args.judge_a_name: ra["verdict"], args.judge_b_name: rb["verdict"],
            f"{args.judge_a_name}Quadrant": ra["quadrant"], f"{args.judge_b_name}Quadrant": rb["quadrant"],
            "agreedQuadrant": bool(agreed_q),
            "consensusQuadrant": ra["quadrant"] if agreed_q else None,
            "optimal": _QUADRANT_OPTIMAL.get(ra["quadrant"]) if agreed_q else None,
        }
        labeled.append(row)
        if ra["verdict"] and rb["verdict"]:
            va.append(ra["verdict"]); vb.append(rb["verdict"])
            qa.append(ra["quadrant"]); qb.append(rb["quadrant"])
        else:
            parse_fail += 1

    n_paired = len(va)
    _r = lambda x: round(x, 4) if x is not None else None  # noqa: E731
    agreement = {
        "nPaired": n_paired,
        "parseFailures": parse_fail,
        "verdict4class": {
            "observedAgreement": round(sum(1 for x, y in zip(va, vb) if x == y) / n_paired, 4) if n_paired else None,
            "cohenKappa": _r(cohen_kappa(va, vb)),
            "cohenKappaCI95": bootstrap_ci_agreement(va, vb, cohen_kappa),
            "gwetAC1": _r(gwet_ac1(va, vb)),
            "gwetAC1CI95": bootstrap_ci_agreement(va, vb, gwet_ac1),
        },
        "quadrant4class": {
            "observedAgreement": round(sum(1 for x, y in zip(qa, qb) if x == y) / n_paired, 4) if n_paired else None,
            "cohenKappa": _r(cohen_kappa(qa, qb)),
            "cohenKappaCI95": bootstrap_ci_agreement(qa, qb, cohen_kappa),
            "gwetAC1": _r(gwet_ac1(qa, qb)),
            "gwetAC1CI95": bootstrap_ci_agreement(qa, qb, gwet_ac1),
        },
    }
    agreed = [r for r in labeled if r["agreedQuadrant"]]
    scored_quadrants: dict[str, int] = {}
    for r in agreed:
        scored_quadrants[r["consensusQuadrant"]] = scored_quadrants.get(r["consensusQuadrant"], 0) + 1

    q_kappa = agreement["quadrant4class"]["cohenKappa"]
    q_ac1 = agreement["quadrant4class"]["gwetAC1"]
    resolvable = (q_kappa is not None and q_kappa >= KAPPA_FLOOR)

    out = {
        "schema": "sophia.sophrosyne_external_battery.labeled.v1",
        "candidateOnly": True,
        "battery": battery.get("schema"),
        "n": len(labeled),
        "judges": {
            "familyA": {"name": args.judge_a_name, "spec": args.judge_a},
            "familyB": {"name": args.judge_b_name, "spec": args.judge_b},
        },
        "kappaFloor": KAPPA_FLOOR,
        "agreement": agreement,
        "scoredSet": {
            "n": len(agreed),
            "quadrantCounts": scored_quadrants,
            "note": "cases where BOTH judge families agree on quadrant; arm scoring runs on these.",
        },
        "groundTruthResolvable": bool(resolvable),
        "resolvabilityNote": (
            f"quadrant Cohen kappa={q_kappa} (floor {KAPPA_FLOOR}); Gwet AC1={q_ac1}. "
            + ("RESOLVABLE." if resolvable else "BELOW FLOOR — metric not resolvable => NO-GO (do not score for a claim).")
        ),
        "cases": labeled,
    }
    LABELED_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {LABELED_PATH.relative_to(ROOT)}")
    print(f"  paired(parse-ok)={n_paired} parseFail={parse_fail}")
    print(f"  quadrant: obsAgr={agreement['quadrant4class']['observedAgreement']} kappa={q_kappa} AC1={q_ac1}")
    print(f"  scored(agreed)={len(agreed)} quadrants={scored_quadrants}")
    print(f"  resolvable(kappa>={KAPPA_FLOOR})={resolvable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
