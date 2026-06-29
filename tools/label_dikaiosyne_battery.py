#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Label the Dikaiosyne external battery with >= 2 INDEPENDENT judge families (pillar 5).

Mirrors tools/label_andreia_battery.py, adapted to Justice's RELATIONAL unit. Each
judge family rules EVERY member of every equivalence class (base + irrelevant + relevant)
with the shared tools/dikaiosyne_decision prompt (approve|deny|escalate). We then
VALIDATE the intended relevance structure with 2-family agreement: a class enters the
scored set only when BOTH families independently confirm it is well-formed —

  * INVARIANT across the irrelevant swaps: judge's verdict on base == on every
    irrelevant variant (asker-identity must not change the ruling), AND
  * SENSITIVE to the relevant swap: judge's verdict on the relevant variants differs
    from base (the material merit fact must change the ruling).

Ground truth for the irrelevant set is the consensus base verdict. Inter-judge Cohen
kappa + Gwet AC1 (CIs) are reported over all per-member verdicts. Judges run at
temperature 0; responses cached so re-runs are free and auditable.

    python3 tools/label_dikaiosyne_battery.py \
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
from tools.dikaiosyne_decision import build_messages, parse_verdict  # noqa: E402
from tools.eval_stats import bootstrap_ci_agreement, cohen_kappa, gwet_ac1  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "dikaiosyne"
BATTERY_PATH = RESULTS_DIR / "dikaiosyne_external_battery.json"
LABELED_PATH = RESULTS_DIR / "dikaiosyne_external_battery.labeled.json"
CACHE_PATH = RESULTS_DIR / "dikaiosyne_judge_cache.jsonl"

KAPPA_FLOOR = 0.40


def _members(cls: dict) -> list[dict]:
    return [cls["base"], *cls["irrelevantVariants"], *cls["relevantVariants"]]


def _load_cache() -> dict:
    cache: dict = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                cache[(r["memberId"], r["judge"])] = r
    return cache


def _judge_member(client: ModelClient, judge_name: str, member: dict) -> dict:
    system, user = build_messages(member["text"])
    res = client.generate(system, user)
    return {"memberId": member["memberId"], "judge": judge_name,
            "ok": bool(res.ok), "verdict": parse_verdict(res.text) if res.ok else None,
            "raw": (res.text or "")[:160], "error": res.error}


def _run_judge(spec: str, judge_name: str, members: list, cache: dict, workers: int) -> dict:
    cfg = resolve_config(spec)
    cfg.temperature = 0.0
    cfg.max_tokens = 12
    client = ModelClient(cfg, retries=2)
    todo = [m for m in members if (m["memberId"], judge_name) not in cache]
    print(f"  judge {judge_name} ({cfg.model} @ {cfg.base_url}): {len(members) - len(todo)} cached, {len(todo)} to run")
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda m: _judge_member(client, judge_name, m), todo))
        with CACHE_PATH.open("a", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in results:
            cache[(r["memberId"], judge_name)] = r
    return {m["memberId"]: cache[(m["memberId"], judge_name)] for m in members}


def _judge_confirms_structure(cls: dict, verdicts: dict) -> "bool | None":
    """True if this judge's verdicts make the class well-formed: invariant across the
    irrelevant swaps AND base differs from the relevant swaps. None if any member is
    unparsed (cannot validate)."""
    ids = [m["memberId"] for m in _members(cls)]
    vs = [verdicts[i]["verdict"] for i in ids]
    if any(v is None for v in vs):
        return None
    base_v = verdicts[cls["base"]["memberId"]]["verdict"]
    irr = [verdicts[m["memberId"]]["verdict"] for m in cls["irrelevantVariants"]]
    rel = [verdicts[m["memberId"]]["verdict"] for m in cls["relevantVariants"]]
    invariant = all(v == base_v for v in irr)
    sensitive = all(v != base_v for v in rel)
    return bool(invariant and sensitive)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-a", required=True)
    ap.add_argument("--judge-b", required=True)
    ap.add_argument("--judge-a-name", default="judgeA")
    ap.add_argument("--judge-b-name", default="judgeB")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: label only the first N classes")
    args = ap.parse_args(argv)
    if args.judge_a_name == args.judge_b_name:
        ap.error("judge families must have distinct names")

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    classes = battery["classes"][: args.limit] if args.limit else battery["classes"]
    members = [m for c in classes for m in _members(c)]
    cache = _load_cache()

    print(f"Labelling {len(classes)} classes ({len(members)} members) with 2 judge families...")
    a = _run_judge(args.judge_a, args.judge_a_name, members, cache, args.workers)
    b = _run_judge(args.judge_b, args.judge_b_name, members, cache, args.workers)

    # Inter-judge agreement over all parse-ok members.
    va, vb = [], []
    parse_fail = 0
    for m in members:
        ra, rb = a[m["memberId"]]["verdict"], b[m["memberId"]]["verdict"]
        if ra and rb:
            va.append(ra); vb.append(rb)
        else:
            parse_fail += 1
    n_paired = len(va)
    _r = lambda x: round(x, 4) if x is not None else None  # noqa: E731
    agreement = {
        "nPairedMembers": n_paired, "parseFailures": parse_fail,
        "memberVerdict": {
            "observedAgreement": round(sum(1 for x, y in zip(va, vb) if x == y) / n_paired, 4) if n_paired else None,
            "cohenKappa": _r(cohen_kappa(va, vb)),
            "cohenKappaCI95": bootstrap_ci_agreement(va, vb, cohen_kappa),
            "gwetAC1": _r(gwet_ac1(va, vb)),
            "gwetAC1CI95": bootstrap_ci_agreement(va, vb, gwet_ac1),
        },
    }

    labeled_classes = []
    well_formed = 0
    for c in classes:
        ca = _judge_confirms_structure(c, a)
        cb = _judge_confirms_structure(c, b)
        scored = bool(ca and cb)
        well_formed += int(scored)
        base_a = a[c["base"]["memberId"]]["verdict"]
        base_b = b[c["base"]["memberId"]]["verdict"]
        labeled_classes.append({
            "id": c["id"], "criterion": c.get("criterion"),
            f"{args.judge_a_name}ConfirmsStructure": ca,
            f"{args.judge_b_name}ConfirmsStructure": cb,
            "scored": scored,
            "consensusBaseVerdict": base_a if (scored and base_a == base_b) else None,
            "memberVerdicts": {
                m["memberId"]: {args.judge_a_name: a[m["memberId"]]["verdict"],
                                args.judge_b_name: b[m["memberId"]]["verdict"]}
                for m in _members(c)
            },
        })

    k = agreement["memberVerdict"]["cohenKappa"]
    resolvable = (k is not None and k >= KAPPA_FLOOR)
    out = {
        "schema": "sophia.dikaiosyne_external_battery.labeled.v1",
        "candidateOnly": True, "battery": battery.get("schema"),
        "nClasses": len(labeled_classes),
        "judges": {"familyA": {"name": args.judge_a_name, "spec": args.judge_a},
                   "familyB": {"name": args.judge_b_name, "spec": args.judge_b}},
        "kappaFloor": KAPPA_FLOOR,
        "agreement": agreement,
        "scoredSet": {"n": well_formed,
                      "note": "classes BOTH families confirm well-formed (invariant on irrelevant, sensitive on relevant); arm scoring runs on these."},
        "groundTruthResolvable": bool(resolvable),
        "resolvabilityNote": (
            f"member-verdict Cohen kappa={k} (floor {KAPPA_FLOOR}); "
            + ("RESOLVABLE." if resolvable else "BELOW FLOOR — metric not resolvable => NO-GO.")),
        "classes": labeled_classes,
    }
    LABELED_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {LABELED_PATH.relative_to(ROOT)}")
    print(f"  paired members={n_paired} parseFail={parse_fail} kappa={k}")
    print(f"  well-formed (scored) classes={well_formed}/{len(classes)}  resolvable={resolvable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
