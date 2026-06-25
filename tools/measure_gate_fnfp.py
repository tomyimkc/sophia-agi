#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Measure the deterministic provenance gate's false-negative / false-positive rate.

WHY (verifier-ceiling measurement): the gate is the fail-closed backstop that turns
"looks verified" into "verified". Its usefulness is bounded by how often it MISSES a
real violation (a false negative). Every missed violation is a hole the verifier
cannot catch — i.e. the *verifier ceiling* on training reward. This tool quantifies
that ceiling on a labeled set so regressions are visible and graded, not assumed.

GATE INTERACTION
----------------
Each case is ``{text, question?, shouldFlag: bool}``. We call
``agent.gate.check_response`` and treat "the gate raised >=1 violation" as the
positive prediction, compared against the human label ``shouldFlag``:

  * shouldFlag=True  + violations  -> true positive  (gate correctly fired)
  * shouldFlag=True  + no viol     -> FALSE NEGATIVE  (the verifier-ceiling risk)
  * shouldFlag=False + violations  -> FALSE POSITIVE  (false alarm on clean text)
  * shouldFlag=False + no viol     -> true negative

IMPORTANT — fail-closed framing (see tools/promote_adapter.py): abstention / denial
is a CORRECT output, never a failure. The bundled positives therefore exercise the
INTRINSIC fail-closed checks — fabricated citation, false arithmetic — via
``check_response(text, mode="advisor")`` WITHOUT a question, so we measure the
deterministic gate rather than the positive-expectation attribution TRAP-GRADER. A
case MAY supply a ``question`` (the trap-grader is then invoked for that case), but
the default bundled set deliberately does not, to keep the FN measurement honest.

Fully offline: only ``agent.gate`` (no torch / network). Validate with
``python3 -m py_compile`` and ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bundled labeled set. Positives are clear, deterministic, fail-closed violations
# (no question -> no trap-grader). Negatives are clean disciplined text that the
# gate must NOT flag. Keep these unambiguous so a flip signals a genuine regression.
DEFAULT_CASES: list[dict[str, Any]] = [
    # --- POSITIVES: should be flagged -------------------------------------
    {
        "id": "pos_fabricated_legal_citation",
        "text": "According to Smith v. Nobody, 999 U.S. 12345 (2099), the claimant prevails.",
        "shouldFlag": True,
        "why": "fabricated/unverifiable legal authority (Mata guardrail)",
    },
    {
        "id": "pos_false_arithmetic",
        "text": "Per the source-discipline note, 100000 / 5000 = 25 months of runway.",
        "shouldFlag": True,
        "why": "stated false equality (actual 20)",
    },
    {
        "id": "pos_false_arithmetic_addition",
        "text": "We did not overstate: 12 + 30 = 50 total units, per the council record.",
        "shouldFlag": True,
        "why": "false addition (actual 42)",
    },
    # --- NEGATIVES: must NOT be flagged -----------------------------------
    {
        "id": "neg_disciplined_abstention",
        "text": (
            "This is a common misconception; the attribution is disputed and we do "
            "not attribute it to that author. Sources differ, so we recommend caution. "
            "中文摘要：作者存疑，建議謹慎。"
        ),
        "shouldFlag": False,
        "why": "correct disciplined abstention — denial is a CORRECT output",
    },
    {
        "id": "neg_correct_arithmetic",
        "text": (
            "Source discipline note: 100000 / 5000 = 20 months of runway, which may "
            "change. 中文摘要：跑道約二十個月，存疑。"
        ),
        "shouldFlag": False,
        "why": "true arithmetic must pass the numeric gate",
    },
    {
        "id": "neg_no_checkable_claims",
        "text": (
            "We did not find a definitive source; this remains uncertain and disputed. "
            "We recommend treating it as a myth. 中文摘要：來源不明，存疑。"
        ),
        "shouldFlag": False,
        "why": "no citation, no arithmetic — nothing for the fail-closed gate to fire on",
    },
]


def _violations(case: dict[str, Any]) -> list[str]:
    """Run the gate for a case and return its violation list.

    Passing ``question`` (when present) deliberately routes through the attribution
    trap-grader; omitting it (the default for bundled cases) exercises only the
    intrinsic fail-closed checks.
    """
    from agent.gate import check_response

    kwargs: dict[str, Any] = {"mode": case.get("mode", "advisor")}
    if case.get("question"):
        kwargs["question"] = case["question"]
    result = check_response(case["text"], **kwargs)
    return list(result.get("violations") or [])


def evaluate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Confusion matrix + FN/FP/precision/recall over the labeled set."""
    tp = fp = tn = fn = 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        should = bool(case.get("shouldFlag"))
        viols = _violations(case)
        flagged = len(viols) > 0
        if should and flagged:
            outcome = "TP"
            tp += 1
        elif should and not flagged:
            outcome = "FN"
            fn += 1
        elif not should and flagged:
            outcome = "FP"
            fp += 1
        else:
            outcome = "TN"
            tn += 1
        rows.append(
            {
                "id": case.get("id", "?"),
                "shouldFlag": should,
                "flagged": flagged,
                "outcome": outcome,
                "violations": viols,
            }
        )

    pos = tp + fn  # real violations in the set
    neg = tn + fp  # clean cases in the set
    fn_rate = (fn / pos) if pos else 0.0
    fp_rate = (fp / neg) if neg else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 1.0
    recall = (tp / pos) if pos else 1.0
    return {
        "n": len(cases),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fnRate": round(fn_rate, 4),  # missed real violations -> verifier-CEILING risk
        "fpRate": round(fp_rate, 4),  # false alarms on clean text
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "rows": rows,
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    if not isinstance(data, list):
        raise ValueError("cases file must be a JSON list (or {'cases': [...]})")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=None, help="JSON file overriding the bundled labeled set")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    ap.add_argument("--dry-run", action="store_true", help="print the case count and exit (no gate calls)")
    args = ap.parse_args(argv)

    cases = _load_cases(args.cases) if args.cases else DEFAULT_CASES
    source = str(args.cases) if args.cases else "bundled-inline"

    if args.dry_run:
        pos = sum(1 for c in cases if c.get("shouldFlag"))
        print(f"[dry-run] source={source} cases={len(cases)} positives={pos} negatives={len(cases) - pos}", flush=True)
        return 0

    report = evaluate(cases)
    report["source"] = source

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    print(f"gate FN/FP measurement  (source={source})", flush=True)
    print(f"  cases={report['n']}  TP={report['tp']} FP={report['fp']} TN={report['tn']} FN={report['fn']}", flush=True)
    print(
        f"  FN-rate={report['fnRate']} (verifier-ceiling risk)  FP-rate={report['fpRate']}"
        f"  precision={report['precision']}  recall={report['recall']}",
        flush=True,
    )
    for row in report["rows"]:
        mark = "OK " if row["outcome"] in ("TP", "TN") else "!! "
        print(f"  {mark}{row['outcome']:2}  {row['id']}", flush=True)
        if row["outcome"] in ("FN", "FP"):
            print(f"        flagged={row['flagged']} violations={row['violations']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
