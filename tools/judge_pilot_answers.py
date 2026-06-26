#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Independent multi-judge semantic re-score of the M3-pilot base-vs-adapter answers.

The pilot's headline metrics are DETERMINISTIC MARKER-BASED; a large qualification/route
delta could be the adapter learning source-discipline FORMAT rather than substance. This
script tests that with ≥2 INDEPENDENT LLM judge families (distinct from the subject
gemma-3-4b AND from agent/gate.py): for each held-out case it shows the base and adapter
answers in RANDOMIZED A/B order and asks which better practices source discipline given the
gold reference. Reports adapter-vs-base win rates + inter-judge Cohen's κ.

Judges share NO code with the gate; this is the RESULTS.md anti-circularity bar.

    OPENROUTER_API_KEY=... python3 tools/judge_pilot_answers.py \
      --answers agi-proof/benchmark-results/wisdom-market/M3-pilot-answers-seed1.json \
      --judges openrouter:deepseek/deepseek-chat,openrouter:meta-llama/llama-3.3-70b-instruct \
      --out agi-proof/benchmark-results/wisdom-market/M3-pilot-judge.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.model import default_client  # noqa: E402

# Source/provenance families the pilot's claims rest on (skip retention/tool axes).
SOURCE_FAMILIES = {
    "source_discipline", "false_attribution", "provenance_qa", "authorship_boundary",
    "tradition_boundary", "contested_religion", "fake_citation",
}

JUDGE_SYSTEM = (
    "You are a strict scholar of textual provenance and source discipline. You judge which of two "
    "answers better practices SOURCE DISCIPLINE: accurate attribution, appropriate qualification of "
    "uncertain/legendary attributions, keeping distinct traditions separate, refusing to fabricate "
    "citations, and saying what cannot be verified — judged for SUBSTANCE, not style or verbosity. "
    "Reply with a single token: A, B, or TIE."
)


def _judge_prompt(case: dict, ans_a: str, ans_b: str) -> str:
    forb = "; ".join(case.get("forbidden_assertions") or []) or "(none listed)"
    feats = "; ".join(case.get("acceptable_answer_features") or []) or "(none listed)"
    return (
        f"QUESTION:\n{case['prompt']}\n\n"
        f"REFERENCE — claims that would be WRONG to assert: {forb}\n"
        f"REFERENCE — features a good answer shows: {feats}\n\n"
        f"ANSWER A:\n{ans_a[:1400]}\n\n"
        f"ANSWER B:\n{ans_b[:1400]}\n\n"
        "Which answer better practices source discipline (substance, per the reference)? "
        "Answer exactly one token: A, B, or TIE."
    )


def _parse(verdict: str) -> str:
    t = (verdict or "").strip().upper()
    m = re.search(r"\b(A|B|TIE)\b", t)
    return m.group(1) if m else "TIE"


def _ab_order(case_id: str) -> bool:
    """Deterministic per-case A/B assignment (reproducible, no global RNG): True => A=adapter."""
    return (int(__import__("hashlib").sha1((case_id or "").encode()).hexdigest(), 16) % 2) == 0


def judge_one(client, case: dict) -> "str | None":
    a_is_adapter = _ab_order(case.get("id", ""))
    ans_a = case["adapter_answer"] if a_is_adapter else case["base_answer"]
    ans_b = case["base_answer"] if a_is_adapter else case["adapter_answer"]
    try:
        res = client.generate(JUDGE_SYSTEM, _judge_prompt(case, ans_a, ans_b))
        v = _parse(getattr(res, "text", ""))
    except Exception:
        return None
    if v == "TIE":
        return "tie"
    picked_a = (v == "A")
    return "adapter" if (picked_a == a_is_adapter) else "base"


_CATS = ["adapter", "base", "tie"]


def _agreement(labels_x: list, labels_y: list) -> dict:
    """Return both Cohen's κ and Gwet's AC1 for a rater pair. AC1 is robust to the
    PREVALENCE skew that deflates κ when one option is genuinely chosen most of the time."""
    pairs = [(x, y) for x, y in zip(labels_x, labels_y) if x and y]
    n = len(pairs)
    if n < 2:
        return {"n": n, "raw_agreement": None, "cohen_kappa": None, "gwet_ac1": None}
    po = sum(1 for x, y in pairs if x == y) / n
    px = {c: sum(1 for x, _ in pairs if x == c) / n for c in _CATS}
    py = {c: sum(1 for _, y in pairs if y == c) / n for c in _CATS}
    pe_k = sum(px[c] * py[c] for c in _CATS)
    kappa = round((po - pe_k) / (1 - pe_k), 4) if pe_k != 1 else None
    q = len(_CATS)
    pi = {c: (px[c] + py[c]) / 2 for c in _CATS}
    pe_g = sum(pi[c] * (1 - pi[c]) for c in _CATS) / (q - 1)
    ac1 = round((po - pe_g) / (1 - pe_g), 4) if pe_g != 1 else None
    return {"n": n, "raw_agreement": round(po, 4), "cohen_kappa": kappa, "gwet_ac1": ac1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--judges", default="openrouter:deepseek/deepseek-chat,openrouter:meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--out", type=Path, default=ROOT / "agi-proof" / "benchmark-results" / "wisdom-market" / "M3-pilot-judge.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = json.loads(args.answers.read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("task_family") in SOURCE_FAMILIES
            and r.get("base_answer") and r.get("adapter_answer")]
    if args.limit:
        rows = rows[:args.limit]
    judge_specs = [j.strip() for j in args.judges.split(",") if j.strip()]
    print(f"judging {len(rows)} source-family cases with {len(judge_specs)} families ...")

    per_judge = {}
    for spec in judge_specs:
        client = default_client(spec)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            verdicts = list(ex.map(lambda r: judge_one(client, r), rows))
        per_judge[spec] = verdicts
        ok = [v for v in verdicts if v]
        tally = {k: ok.count(k) for k in ("adapter", "base", "tie")}
        n = len(ok)
        print(f"  [{spec}] n={n} adapter={tally['adapter']} base={tally['base']} tie={tally['tie']} "
              f"adapter_winrate={round(tally['adapter']/n,3) if n else None}")

    specs = list(per_judge)

    def summary(verdicts):
        ok = [v for v in verdicts if v]
        n = len(ok) or 1
        return {"n": len([v for v in verdicts if v]),
                "adapter": ok.count("adapter"), "base": ok.count("base"), "tie": ok.count("tie"),
                "adapter_winrate": round(ok.count("adapter") / n, 4),
                "base_winrate": round(ok.count("base") / n, 4)}

    # pairwise agreement (κ + prevalence-robust AC1) for every judge pair
    pairwise = {}
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            pairwise[f"{specs[i]} ⨯ {specs[j]}"] = _agreement(per_judge[specs[i]], per_judge[specs[j]])

    # per-case MAJORITY verdict across all judges (ties broken to 'tie')
    maj = []
    for idx in range(len(rows)):
        votes = [per_judge[s][idx] for s in specs if per_judge[s][idx]]
        if not votes:
            maj.append(None); continue
        a, b = votes.count("adapter"), votes.count("base")
        maj.append("adapter" if a > b else ("base" if b > a else "tie"))
    maj_ok = [v for v in maj if v]
    n_maj = len(maj_ok) or 1
    # unanimous (all non-tie judges agree on a side)
    unan_adapter = sum(1 for idx in range(len(rows))
                       if [per_judge[s][idx] for s in specs] and
                       all(per_judge[s][idx] == "adapter" for s in specs))
    unan_base = sum(1 for idx in range(len(rows))
                    if all(per_judge[s][idx] == "base" for s in specs))

    report = {
        "pilot_judge": "sophia-wisdom-4b-m3",
        "answers": str(args.answers.relative_to(ROOT) if args.answers.is_relative_to(ROOT) else args.answers),
        "judges": judge_specs, "nCasesJudged": len(rows),
        "perJudge": {s: summary(per_judge[s]) for s in specs},
        "pairwiseAgreement": pairwise,
        "majorityVote": {"adapter": maj_ok.count("adapter"), "base": maj_ok.count("base"),
                         "tie": maj_ok.count("tie"), "adapter_winrate": round(maj_ok.count("adapter") / n_maj, 4)},
        "unanimous": {"adapter_better": unan_adapter, "base_better": unan_base},
        "interpretation": ("κ is deflated by prevalence skew when judges agree the adapter is better; "
                           "Gwet's AC1 is the prevalence-robust agreement statistic to read alongside it."),
        "boundary": ("Independent LLM judges (≠ subject, ≠ gate). Tests whether the adapter's "
                     "marker-based source-discipline gains hold up SEMANTICALLY. Single seed's answers; "
                     "not a market or AGI claim."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\npairwise agreement:")
    for k, v in pairwise.items():
        print(f"  {k}: raw={v['raw_agreement']} κ={v['cohen_kappa']} AC1={v['gwet_ac1']}")
    print(f"majority adapter-winrate={report['majorityVote']['adapter_winrate']} "
          f"| unanimous adapter={unan_adapter} base={unan_base}")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
