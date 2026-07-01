#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Score the step-verifier against the misstep-injection pack.

This is an **objective** verifier eval, the math/physics analogue of
``tools/run_legal_citation_bench.py``: every case in ``data/misstep_pack.jsonl``
carries a ground-truth ``expectVerified`` (a clean derivation that should fully
verify, or a one-error variant that must be caught), and ``agent.step_verifier``
is deterministic — so there is **no LLM judge**, and the number measures the
*verifier's* accuracy at catching a single injected misstep (sign flip, dropped
factor, arithmetic slip, wrong unit/dimension, wrong value), not any model's.

Positive class = "derivation should verify":
  TP  clean derivation correctly accepted
  TN  corrupted derivation correctly rejected        <- the safety win
  FP  corrupted derivation MISSED (accepted)          <- the dangerous failure
  FN  clean derivation wrongly rejected (false alarm)
``abstain`` is tracked separately: a clean case that abstains is an
uncovered/unverified case (counts against coverage, NOT as a false alarm); a
corrupted case that abstains was not caught-as-wrong but was also not falsely
passed. With sympy absent the math oracle fails closed to abstain, so coverage
falls but no fabrication slips through — exactly the intended fail-closed behavior.

Honestly bounded: small, constructed pack (one injected error per case). This
validates the step-verification + fail-closed logic end-to-end — it is NOT a
headline capability claim. ``canClaimAGI`` is unaffected.

    python tools/run_misstep_bench.py            # print summary
    python tools/run_misstep_bench.py --json     # machine-readable
    python tools/run_misstep_bench.py --write     # also write run artifact
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

from agent.step_verifier import verify_derivation  # noqa: E402

PACK = ROOT / "data" / "misstep_pack.jsonl"
ARTIFACT = ROOT / "agi-proof" / "benchmark-results" / "misstep-bench.json"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(pack_path: Path = PACK) -> dict:
    cases = _load(pack_path)
    tp = tn = fp = fn = 0
    abstained_clean = abstained_corrupt = 0
    misses: list[str] = []
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "caught": 0, "abstain": 0})
    for case in cases:
        res = verify_derivation(
            case["steps"], gold=case.get("gold"), default_domain=case.get("domain", "math"),
        )
        verdict = res.verdict
        expect = bool(case["expectVerified"])
        if expect:
            if verdict == "accepted":
                tp += 1
            elif verdict == "abstain":
                abstained_clean += 1
            else:
                fn += 1
                misses.append(f"false alarm (rejected a clean derivation): {case['id']}")
        else:
            etype = case.get("errorType", "unknown")
            by_type[etype]["total"] += 1
            if verdict == "rejected":
                tn += 1
                by_type[etype]["caught"] += 1
            elif verdict == "abstain":
                abstained_corrupt += 1
                by_type[etype]["abstain"] += 1
            else:  # accepted a corrupted derivation -> missed the misstep
                fp += 1
                misses.append(f"MISSED misstep ({etype}): {case['id']}")
    n = len(cases)
    decided = tp + tn + fp + fn
    corrupt = tn + fp + abstained_corrupt
    decided_clean = tp + fn  # clean cases that produced a verdict (abstentions reported separately)
    return {
        "benchmark": "misstep_injection",
        "n": n,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "abstained": {"clean": abstained_clean, "corrupt": abstained_corrupt},
        "accuracy": round((tp + tn) / decided, 4) if decided else 0.0,
        "misstepCatchRecall": round(tn / corrupt, 4) if corrupt else None,
        # False alarms among DECIDED clean cases only (FN/(TP+FN)); abstentions are not
        # diluted into the denominator — they are reported via abstained.clean.
        "falseAlarmRate": round(fn / decided_clean, 4) if decided_clean else None,
        "verifiedStepCoverage": round((tp + tn + fp + fn) / n, 4) if n else 0.0,
        "catchByErrorType": {
            t: {"caught": v["caught"], "total": v["total"], "abstain": v["abstain"],
                "recall": round(v["caught"] / v["total"], 4) if v["total"] else None}
            for t, v in sorted(by_type.items())
        },
        "misses": misses,
        "scoring": "objective vs ground-truth expectVerified; deterministic step-verifier, no LLM judge",
        "canClaimAGI": False,
    }


def _pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ap.add_argument("--write", action="store_true", help="write run artifact under agi-proof/")
    args = ap.parse_args(argv)

    result = run()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        c = result["confusion"]
        a = result["abstained"]
        print(f"misstep-injection benchmark — N={result['n']}")
        print(f"  accuracy (decided)           {result['accuracy'] * 100:.1f}%")
        print(f"  misstep-catch recall         {_pct(result['misstepCatchRecall'])}  (TN={c['tn']}, missed FP={c['fp']})")
        print(f"  false-alarm rate             {_pct(result['falseAlarmRate'])}  (FN={c['fn']}, TP={c['tp']})")
        print(f"  abstained (uncovered)        clean={a['clean']} corrupt={a['corrupt']}")
        print("  catch by error type:")
        for t, v in result["catchByErrorType"].items():
            print(f"    {t:16s} {v['caught']}/{v['total']} caught  ({_pct(v['recall'])}), abstain={v['abstain']}")
        for m in result["misses"]:
            print(f"  ! {m}")
        if not result["misses"]:
            print("  no misses (every injected misstep flagged or abstained; no false alarms)")
    if args.write:
        ARTIFACT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {ARTIFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
