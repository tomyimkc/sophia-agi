#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Council panel benchmark — does a HETEROGENEOUS model team beat one model wearing
N hats? (the "team agents mode" head-to-head.)

The design intent was "council members deliberate and decide". The open question is
empirical: a council of one model in N costumes has CORRELATED errors (shared
weights) — does using N *different* models (genuinely independent voters) actually
win? This runs three conditions over the provenance cases (gold-labelled) and
majority-votes each panel's judged verdicts:

  single  — one model answers once (baseline).
  homo    — the SAME model answers N times (more votes, correlated errors). This is
            the control: it isolates whether *diversity* — not just *more votes* —
            is the active ingredient.
  hetero  — N DIFFERENT models answer once each (a real team). "team agents mode".

Each member answers the case's natural question; the independent judge
(provenance_bench.judge) labels each answer hallucinated/affirmed-gold; the panel's
verdict is the MAJORITY vote; we score the vote against external gold. Deterministic
lexical judge by default (offline-capable). Reports per-condition hallucination
rate (false cases) + gold-affirm rate (true cases) + the team-vs-solo delta.

    python tools/run_council_panel.py --models mock --homo-n 3            # offline
    python tools/run_council_panel.py \\
        --models ollama:qwen3:30b-a3b,ollama:dolphin-llama3:8b,ollama:llama3.2:3b \\
        --limit 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.dataset import build_cases  # noqa: E402
from provenance_bench.judge import judge_answer  # noqa: E402
from provenance_bench.runner import NEUTRAL_SYSTEM  # noqa: E402

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "council-panel.public-report.json"


def _answer(client, prompt: str) -> str:
    res = client.generate(NEUTRAL_SYSTEM, prompt)
    return (getattr(res, "text", "") or "") if getattr(res, "ok", True) else ""


def _verdict(answer: str, case) -> dict:
    """Independent judge of one answer -> {hallucinated, affirmed_gold}."""
    j = judge_answer(answer, case)
    return {"hallucinated": bool(j.hallucinated), "affirmed_gold": bool(j.affirmed_gold)}


def _majority(verdicts: list, key: str) -> bool:
    votes = sum(1 for v in verdicts if v[key])
    return votes * 2 > len(verdicts)  # strict majority


def _panel_verdict(answers: list, case) -> dict:
    verdicts = [_verdict(a, case) for a in answers]
    return {
        "hallucinated": _majority(verdicts, "hallucinated"),
        "affirmed_gold": _majority(verdicts, "affirmed_gold"),
        "votes": verdicts,
    }


def run(cases, *, single_client, homo_clients, hetero_clients) -> dict:
    rows = []
    for case in cases:
        # Each condition issues its own generate() call. When the single baseline IS
        # hetero seat 0 (same object), reuse that one answer as the first hetero vote
        # to avoid a duplicate paid/non-deterministic call against the same model.
        single_ans = _answer(single_client, case.prompt)
        homo_ans = [_answer(c, case.prompt) for c in homo_clients]
        hetero_ans = []
        for j, c in enumerate(hetero_clients):
            if j == 0 and c is single_client:
                hetero_ans.append(single_ans)
            else:
                hetero_ans.append(_answer(c, case.prompt))
        rows.append({
            "case_id": case.id, "label": case.label,
            "single": _verdict(single_ans, case),
            "homo": _panel_verdict(homo_ans, case),
            "hetero": _panel_verdict(hetero_ans, case),
        })
    return {"rows": rows}


def _rate(rows, cond, label, key) -> float:
    sub = [r for r in rows if r["label"] == label]
    if not sub:
        return 0.0
    return round(sum(1 for r in sub if r[cond][key]) / len(sub), 4)


def summarize(rows) -> dict:
    conds = ("single", "homo", "hetero")
    out = {
        "cases": len(rows),
        "falseCases": sum(1 for r in rows if r["label"] == "false"),
        "trueCases": sum(1 for r in rows if r["label"] == "true"),
        "hallucinationByCondition": {c: _rate(rows, c, "false", "hallucinated") for c in conds},
        "goldAffirmedByCondition": {c: _rate(rows, c, "true", "affirmed_gold") for c in conds},
    }
    h = out["hallucinationByCondition"]
    out["deltas"] = {
        "homoVsSingle": round(h["single"] - h["homo"], 4),
        "heteroVsSingle": round(h["single"] - h["hetero"], 4),
        "heteroVsHomo": round(h["homo"] - h["hetero"], 4),  # >0 => diversity helps beyond more-votes
    }
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="mock",
                    help="comma list of model specs; first is the 'single' baseline + the homo model; "
                         "the full list is the hetero panel")
    ap.add_argument("--homo-n", type=int, default=0,
                    help="homogeneous panel size (default = len(models), so homo and hetero have equal votes)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N cases (target a harder slice)")
    ap.add_argument("--shuffle", action="store_true", help="seeded shuffle for a mixed true/false slice")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    specs = [s.strip() for s in args.models.split(",") if s.strip()]
    from agent.model import default_client

    def mk(spec):
        c = default_client(spec)
        try:
            c.spec = spec  # tag for reporting
        except Exception:
            # tagging is best-effort cosmetics; some clients reject attribute
            # assignment (e.g. __slots__) — safe to skip, reporting falls back.
            pass
        return c

    hetero_clients = [mk(s) for s in specs]
    single_client = hetero_clients[0]
    homo_n = args.homo_n or len(specs)
    homo_clients = [mk(specs[0]) for _ in range(homo_n)]  # same model, N voters

    cases = build_cases()
    if args.shuffle:
        import random

        random.Random(0).shuffle(cases)
    if args.offset:
        cases = cases[args.offset:]
    if args.limit:
        cases = cases[: args.limit]

    result = run(cases, single_client=single_client, homo_clients=homo_clients, hetero_clients=hetero_clients)
    summary = summarize(result["rows"])

    report = {
        "benchmark": "council-panel",
        "models": specs,
        "homoN": homo_n,
        "heteroN": len(specs),
        "scopeNote": (
            "Tests whether a heterogeneous model team beats one model wearing N hats. "
            "homo = same model x N (correlated errors, the control); hetero = N different "
            "models (independent voters). Deterministic lexical judge; single-run illustrative."
        ),
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"report -> {args.out}")

    h = summary["hallucinationByCondition"]
    g = summary["goldAffirmedByCondition"]
    print(f"\ncouncil panel — N={summary['cases']} (false {summary['falseCases']} / true {summary['trueCases']})")
    print(f"{'condition':10} {'halluc(false)':>14} {'gold(true)':>12}")
    for c in ("single", "homo", "hetero"):
        print(f"  {c:8} {h[c]:>13.1%} {g[c]:>12.1%}")
    d = summary["deltas"]
    print(f"\nΔ homo  vs single (halluc): {d['homoVsSingle'] * 100:+.1f}%")
    print(f"Δ hetero vs single (halluc): {d['heteroVsSingle'] * 100:+.1f}%")
    print(f"Δ hetero vs homo  (diversity effect): {d['heteroVsHomo'] * 100:+.1f}%  (>0 => team diversity helps beyond more votes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
