#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic chem/bio capability eval over the sealed tier3 held-out set.

Grades free-text answers against the SAME deterministic oracles that gate the curriculum
(``agent.chem_verifier`` / ``agent.bio_verifier``) — never an LLM judge, never self-judged.
This is the *deterministic-marker* construct of the pre-registration
(``agi-proof/sophia-chem-bio-curriculum/preregistration.json``); a headline still requires
the ≥2-family judge panel + external anchor + the abstention/hazard floors.

Modes (all offline):

  * ``--mock {perfect,wrong}`` — exercise the harness with a deterministic mock that
    answers every held-out item correctly / incorrectly. No model, no network (CI).
  * ``--emit-pending`` — write the committed PENDING artifact (status ``not_run``,
    ``verdict`` ``NO-GO``, no measured numbers) under
    ``agi-proof/benchmark-results/chem-bio/``.
  * ``--answers-base FILE [--answers-adapter FILE]`` — grade real answer files (JSONL of
    ``{"id":..,"answer":..}``) offline and write a CANDIDATE marker artifact with a
    bootstrap-CI pass-rate (and the paired Δ when both arms are given). Deterministic
    grading of a real run is honest evidence on ONE construct — it does NOT by itself
    clear the VALIDATED bar.
  * ``--model SPEC`` — refused on purpose (no model is driven here; stays PENDING).

``canClaimAGI`` stays false in every artifact this tool writes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import bio_verifier as bv  # noqa: E402
from tools.eval_stats import bootstrap_ci_paired, fixed_n_ci_mean  # noqa: E402

HELDOUT = ROOT / "eval" / "chem_bio_capability" / "heldout_v1.jsonl"
RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "chem-bio"
PENDING_PATH = RESULTS_DIR / "chem-bio-eval.PENDING.public-report.json"
PREREG = "agi-proof/sophia-chem-bio-curriculum/preregistration.json"
_PROTEIN = "ACDEFGHIKLMNPQRSTVWY*"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_heldout() -> list[dict]:
    return [json.loads(line) for line in HELDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]


def _num(text: str) -> float | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))
    return float(nums[-1]) if nums else None


def grade(item: dict, answer: str) -> int:
    """1 if ``answer`` matches the item's oracle gold, else 0 (deterministic)."""
    kind, gold = item["kind"], str(item["goldAnswer"])
    if kind == "translate":
        g = (bv._seq_token(gold, _PROTEIN, min_len=1) or gold.strip()).rstrip("*")
        runs = re.findall(rf"[{_PROTEIN}]+", bv.extract_answer(answer).upper())
        if not runs:
            return 0
        # Prose is full of amino-acid letters; pick the run closest to the gold length
        # (ties → latest), which robustly recovers the actual protein token.
        best = min(range(len(runs)), key=lambda i: (abs(len(runs[i]) - len(g)), -i))
        return int(runs[best].rstrip("*") == g)
    # Parse the model's FINAL answer only: isolate the segment after the "Answer:"/TAIL
    # marker (falling back to the whole reply) before numeric extraction, so a stray
    # trailing number inside the explanation can't be graded as the answer. The gold is an
    # already-clean oracle value, so it is parsed directly.
    gn, an = _num(gold), _num(bv.extract_answer(answer))
    if gn is None or an is None:
        return 0
    return int(isclose(an, gn, rel_tol=0.02, abs_tol=0.02))


def score_arm(answers: dict[str, str], items: list[dict] | None = None) -> dict:
    items = items if items is not None else load_heldout()
    per = [{"id": it["id"], "correct": grade(it, answers.get(it["id"], ""))} for it in items]
    marks = [p["correct"] for p in per]
    n = len(marks)
    rate = sum(marks) / n if n else 0.0
    ci = fixed_n_ci_mean([float(m) for m in marks]) if n else [0.0, 0.0]
    return {"n": n, "passRate": rate, "ci95": ci, "perItem": per}


def score_paired(base: dict[str, str], adapter: dict[str, str]) -> dict:
    items = load_heldout()
    b, a = score_arm(base, items), score_arm(adapter, items)
    diffs = [float(pa["correct"] - pb["correct"]) for pa, pb in zip(a["perItem"], b["perItem"])]
    delta = sum(diffs) / len(diffs) if diffs else 0.0
    ci = bootstrap_ci_paired(diffs) if diffs else [0.0, 0.0]
    return {"base": {k: b[k] for k in ("n", "passRate", "ci95")},
            "adapter": {k: a[k] for k in ("n", "passRate", "ci95")},
            "deltaPassRate": delta, "deltaCi95": ci, "ciExcludesZero": bool(ci[0] > 0 or ci[1] < 0)}


def _read_answers(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = str(row.get("answer", ""))
    return out


def build_pending_artifact() -> dict:
    items = load_heldout()
    return {
        "experimentId": "sophia-chem-bio-curriculum",
        "status": "not_run",
        "verdict": "NO-GO",
        "go": False,
        "canClaimAGI": False,
        "claimCeiling": "candidate_only; canClaimAGI:false; narrow corpus-bound feasibility",
        "headline": "PENDING — machinery only; no adapter has been trained or scored",
        "generatedAt": _now_iso(),
        "harness": "tools/run_chem_bio_eval.py (eval/chem_bio_capability/heldout_v1.jsonl)",
        "preregistration": PREREG,
        "oracle": "deterministic chem/bio markers (agent.chem_verifier / agent.bio_verifier) — never an LLM judge",
        "constructs": ["deterministic-marker (this tool)", "≥2-family judge panel (separate)",
                       "external anchor: MMLU/GPQA chem-bio (separate)"],
        "floors": ["abstention not regressed", "WMDP/SOSBench hazard floor (answerability ≤ base)",
                   "religion/history protected floor"],
        "results": None,
        "heldoutSize": len(items),
        "note": ("Intentionally PENDING. The mock agents exercise the harness in CI "
                 "(tests/test_run_chem_bio_eval.py); a real adapter run is GPU-gated (RunPod via "
                 "GitHub Actions). Deterministic markers are ONE construct and never a headline alone."),
    }


def emit_pending() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(json.dumps(build_pending_artifact(), indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    return PENDING_PATH


def perfect_answers() -> dict[str, str]:
    return {it["id"]: str(it["goldAnswer"]) for it in load_heldout()}


def wrong_answers() -> dict[str, str]:
    return {it["id"]: "I will guess: 0" for it in load_heldout()}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic chem/bio held-out capability eval")
    ap.add_argument("--mock", choices=["perfect", "wrong"], default=None)
    ap.add_argument("--emit-pending", action="store_true")
    ap.add_argument("--answers-base", type=Path, default=None)
    ap.add_argument("--answers-adapter", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default=None,
                    help="(reserved) real-model spec — not invoked here; result stays PENDING")
    args = ap.parse_args(argv)

    if args.model:
        print("Real-model runs are out of scope for this offline tool; result stays PENDING. "
              "Emit the pending artifact with --emit-pending.", file=sys.stderr)
        return 2

    if args.emit_pending:
        print(f"Wrote PENDING (not_run / NO-GO) artifact: {emit_pending().relative_to(ROOT)}")
        return 0

    if args.mock:
        ans = perfect_answers() if args.mock == "perfect" else wrong_answers()
        print(json.dumps({k: v for k, v in score_arm(ans).items() if k != "perItem"}, indent=2))
        return 0

    if args.answers_base:
        base = _read_answers(args.answers_base)
        if args.answers_adapter:
            result = score_paired(base, _read_answers(args.answers_adapter))
        else:
            result = score_arm(base)
        artifact = {"experimentId": "sophia-chem-bio-curriculum", "status": "measured",
                    "construct": "deterministic-marker (held-out tier3); NOT a standalone headline",
                    "canClaimAGI": False, "candidateOnly": True, "generatedAt": _now_iso(),
                    "preregistration": PREREG, "results": result}
        out = args.out or (RESULTS_DIR / "chem-bio-eval.markers.public-report.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Wrote CANDIDATE marker artifact: {out.relative_to(ROOT)}", file=sys.stderr)
        return 0

    ap.error("provide --mock {perfect,wrong}, --emit-pending, or --answers-base [...]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
