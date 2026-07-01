#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Council vs monolith — does a discipline-routed, per-seat-verified council catch more errors?

The central hypothesis of the council design (`agent/council_registry.py`): a task routed to its
discipline seat, gated by THAT discipline's verifier, catches errors a single general gate cannot.
This harness measures it deterministically: each labelled answer is gated two ways —

  * COUNCIL  — route to the discipline, gate by its verifier (chemistry/biology/... standalone gates).
  * MONOLITH — one general provenance gate (`agent.gate`) for everything.

and reports, for the wrong answers, how many each setup CATCHES (rejects). The council should catch
the discipline-specific errors (an unbalanced equation, an invalid DNA sequence) that the general
provenance gate has no oracle for, while both catch attribution errors.

Stub seats with labelled answers (no model). The real 3B adapters plug in as the answer source; the
verifiers are unchanged. Deterministic, offline, CI-testable. Makes no capability claim.

    python tools/eval_council_vs_monolith.py
    python tools/eval_council_vs_monolith.py --emit agi-proof/benchmark-results/council-vs-monolith.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.council_registry import GATE_BINDINGS, route, verify  # noqa: E402

# Each case: a task, the gold discipline, a candidate answer, and whether that answer is correct.
# The discipline-specific BAD cases (unbalanced chem, invalid DNA) are the council's reason to exist.
CASES = [
    {"task": "Is this chemistry reaction balanced: H2 + O2 -> H2O ?", "gold": "chemistry",
     "answer": "H2 + O2 -> H2O", "correct": False},                       # council catches, monolith misses
    {"task": "Is this chemistry reaction balanced: 2 H2 + O2 -> 2 H2O ?", "gold": "chemistry",
     "answer": "2 H2 + O2 -> 2 H2O", "correct": True},
    {"task": "Is this DNA gene sequence valid: ACGTXG ?", "gold": "biology",
     "answer": "The DNA sequence ACGTXG encodes it.", "correct": False},  # council catches, monolith misses
    {"task": "Is this DNA gene sequence valid: ACGTACGT ?", "gold": "biology",
     "answer": "The DNA sequence ACGTACGT encodes it.", "correct": True},
    {"task": "In philosophy, who wrote the Dao De Jing?", "gold": "philosophy",
     "answer": "Confucius wrote the Dao De Jing.", "correct": False},     # both catch (attribution)
    {"task": "In philosophy, who wrote the Dao De Jing?", "gold": "philosophy",
     "answer": "No — Confucius did not write the Dao De Jing; it is a Daoist text attributed to Laozi.",
     "correct": True},
    {"task": "Mathematics: compute 100000 / 5000 in months.", "gold": "mathematics",
     "answer": "100000 / 5000 = 25 months", "correct": False},           # both catch (numeric in both gates)
    {"task": "Mathematics: compute 100000 / 5000 in months.", "gold": "mathematics",
     "answer": "100000 / 5000 = 20 months", "correct": True},
]


def _monolith_gate(answer: str, question: str) -> bool:
    """One general provenance gate for everything (passed == clean)."""
    passed, _ = GATE_BINDINGS["provenance"](answer, question, None)
    return passed


def _norm(c: dict) -> dict:
    """Accept both harness fixtures ({gold}) and pack rows ({discipline})."""
    return {"task": c["task"], "gold": c.get("gold") or c.get("discipline"),
            "answer": c["answer"], "correct": bool(c["correct"])}


def load_pack(path) -> "list[dict]":
    from pathlib import Path as _P
    rows = []
    for line in _P(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(_norm(json.loads(line)))
    return rows


def evaluate(cases=None) -> dict:
    cases = [_norm(c) for c in (cases if cases is not None else CASES)]
    routed_ok = 0
    council = {"caught_bad": 0, "passed_good": 0}
    monolith = {"caught_bad": 0, "passed_good": 0}
    n_bad = sum(1 for c in cases if not c["correct"])
    n_good = sum(1 for c in cases if c["correct"])
    per_case = []
    per_discipline: dict = {}

    for c in cases:
        d = route(c["task"])
        routed_ok += int(d.id == c["gold"])
        # Gate on the GOLD discipline to isolate the verifier's value from routing (reported
        # separately). In production the routed discipline is used; routing is a distinct concern.
        cv = verify(c["gold"], c["answer"], question=c["task"])
        council_pass = cv["passed"] and not cv["abstained"]
        mono_pass = _monolith_gate(c["answer"], c["task"])

        pd = per_discipline.setdefault(c["gold"], {"nBad": 0, "councilCaught": 0, "monolithCaught": 0})
        if not c["correct"]:
            council["caught_bad"] += int(not council_pass)
            monolith["caught_bad"] += int(not mono_pass)
            pd["nBad"] += 1
            pd["councilCaught"] += int(not council_pass)
            pd["monolithCaught"] += int(not mono_pass)
        else:
            council["passed_good"] += int(council_pass)
            monolith["passed_good"] += int(mono_pass)

        per_case.append({"task": c["task"][:40], "routed": d.id, "gold": c["gold"],
                         "correct": c["correct"], "councilPass": council_pass, "monolithPass": mono_pass})

    return {
        "nCases": len(cases), "nBad": n_bad, "nGood": n_good,
        "routingAccuracy": round(routed_ok / len(cases), 4) if cases else 0.0,
        "council": {**council, "catchRate": round(council["caught_bad"] / n_bad, 4) if n_bad else 1.0},
        "monolith": {**monolith, "catchRate": round(monolith["caught_bad"] / n_bad, 4) if n_bad else 1.0},
        "councilCatchesMore": council["caught_bad"] > monolith["caught_bad"],
        "perDiscipline": per_discipline,
        "note": ("Council uses each discipline's verifier; monolith uses one general provenance gate. "
                 "The delta is the discipline-specific errors (unbalanced equation, invalid DNA, bad "
                 "balance sheet, implausible dose) the general gate has no oracle for. Stub answers; "
                 "real 3B adapters plug in unchanged."),
        "perCase": per_case,
    }


def offline_invariants() -> "tuple[bool, dict]":
    r = evaluate()
    checks = {
        "routing_accurate": r["routingAccuracy"] >= 0.75,
        "council_catches_more_bad": r["councilCatchesMore"],
        "council_passes_good": r["council"]["passed_good"] == r["nGood"],
        "monolith_misses_some_discipline_errors": r["monolith"]["caught_bad"] < r["nBad"],
    }
    return all(checks.values()), {"checks": checks, "result": {k: v for k, v in r.items() if k != "perCase"}}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", type=Path, help="write the result JSON here")
    ap.add_argument("--pack", type=Path, help="evaluate a held-out pack JSONL instead of the fixtures")
    args = ap.parse_args(argv)
    r = evaluate(load_pack(args.pack) if args.pack else None)
    print(json.dumps({k: v for k, v in r.items() if k != "perCase"}, ensure_ascii=False, indent=2))
    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote -> {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
