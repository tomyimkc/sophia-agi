#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Label the Andreia external battery with >= 2 INDEPENDENT judge families (pillar 5).

Each case's ground-truth optimal action (act|heroic|escalate|hold) is assigned by two
distinct model families — judge A and judge B — each deciding INDEPENDENTLY over the
same raw text with the shared tools/andreia_decision prompt. We report inter-judge
agreement (Cohen kappa AND Gwet AC1, both with bootstrap 95% CIs) at the 4-class verdict
level and at the 3-class quadrant level that the cowardice/recklessness metrics use.

Ground truth for SCORING is the consensus QUADRANT: a case enters the scored set only
when both judges agree on the quadrant ({act,heroic} / hold / escalate). Cases where the
families disagree are kept in the labeled file (for audit) but excluded from scoring, and
the disagreement is exactly what kappa measures. kappa < 0.40 (the spec floor) => the
metric is NOT resolvable => NO-GO; we do not paper over it.

Judges are independent of the GATE (they never see assess_courage) and should be
independent of the SUBJECT/baseline model (judge != answer model). Determinism: judges
run at temperature 0; responses are cached so re-runs are free and auditable.

    python3 tools/label_andreia_battery.py \
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
from tools.andreia_decision import build_messages, parse_verdict, quadrant_of  # noqa: E402
from tools.eval_stats import bootstrap_ci_agreement, cohen_kappa, gwet_ac1  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "andreia"
BATTERY_PATH = RESULTS_DIR / "andreia_external_battery.json"
LABELED_PATH = RESULTS_DIR / "andreia_external_battery.labeled.json"
CACHE_PATH = RESULTS_DIR / "andreia_judge_cache.jsonl"

KAPPA_FLOOR = 0.40


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

    # Per-case labels + consensus.
    labeled: list[dict] = []
    va, vb, qa, qb = [], [], [], []   # aligned, parse-success only (for kappa)
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
        }
        if agreed_q:
            # canonical optimal for the metric: heroic only if BOTH said heroic, else act
            if ra["quadrant"] == "should_act":
                row["optimal"] = "heroic" if (ra["verdict"] == "heroic" and rb["verdict"] == "heroic") else "act"
            elif ra["quadrant"] == "should_hold":
                row["optimal"] = "hold"
            else:
                row["optimal"] = "escalate"
        else:
            row["optimal"] = None
        labeled.append(row)
        if ra["verdict"] and rb["verdict"]:
            va.append(ra["verdict"]); vb.append(rb["verdict"])
            qa.append(ra["quadrant"]); qb.append(rb["quadrant"])
        else:
            parse_fail += 1

    n_paired = len(va)
    agreement = {
        "nPaired": n_paired,
        "parseFailures": parse_fail,
        "verdict4class": {
            "observedAgreement": round(sum(1 for x, y in zip(va, vb) if x == y) / n_paired, 4) if n_paired else None,
            "cohenKappa": cohen_kappa(va, vb),
            "cohenKappaCI95": bootstrap_ci_agreement(va, vb, cohen_kappa),
            "gwetAC1": gwet_ac1(va, vb),
            "gwetAC1CI95": bootstrap_ci_agreement(va, vb, gwet_ac1),
        },
        "quadrant3class": {
            "observedAgreement": round(sum(1 for x, y in zip(qa, qb) if x == y) / n_paired, 4) if n_paired else None,
            "cohenKappa": cohen_kappa(qa, qb),
            "cohenKappaCI95": bootstrap_ci_agreement(qa, qb, cohen_kappa),
            "gwetAC1": gwet_ac1(qa, qb),
            "gwetAC1CI95": bootstrap_ci_agreement(qa, qb, gwet_ac1),
        },
    }
    agreed = [r for r in labeled if r["agreedQuadrant"]]
    scored_quadrants: dict[str, int] = {}
    for r in agreed:
        scored_quadrants[r["consensusQuadrant"]] = scored_quadrants.get(r["consensusQuadrant"], 0) + 1

    q_kappa = agreement["quadrant3class"]["cohenKappa"]
    q_ac1 = agreement["quadrant3class"]["gwetAC1"]
    # Resolvability uses kappa, with AC1 as the prevalence-robust companion (per the spec:
    # kappa>=0.40 OR Gwet AC1 with CI). We require the kappa floor for GO and report both.
    resolvable = (q_kappa is not None and q_kappa >= KAPPA_FLOOR)

    out = {
        "schema": "sophia.andreia_external_battery.labeled.v1",
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
    print(f"  quadrant: obsAgr={agreement['quadrant3class']['observedAgreement']} "
          f"kappa={q_kappa} CI={agreement['quadrant3class']['cohenKappaCI95']} "
          f"AC1={q_ac1} CI={agreement['quadrant3class']['gwetAC1CI95']}")
    print(f"  4-class: obsAgr={agreement['verdict4class']['observedAgreement']} "
          f"kappa={agreement['verdict4class']['cohenKappa']} AC1={agreement['verdict4class']['gwetAC1']}")
    print(f"  scored(agreed)={len(agreed)} quadrants={scored_quadrants}")
    print(f"  resolvable(kappa>={KAPPA_FLOOR})={resolvable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
