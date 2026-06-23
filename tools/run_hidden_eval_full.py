#!/usr/bin/env python3
"""Aggregate hidden-eval comparisons across Sophia modes.

This runner expects a hidden pack plus one response JSON per mode. It scores all
modes with the existing hidden_eval_protocol, emits a comparative aggregate, and
writes a manual-review checklist for semantic/pending cases.

It does not generate model responses itself; that remains the job of
run_hidden_eval_sophia.py or another private runner. This separation keeps hidden
prompts private and makes raw/raw+tools/RAG/gate/full comparisons reproducible.
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
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from hidden_eval_protocol import load_json, score_pack, validate_pack  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "hidden-reviewer-packs" / "full-eval-aggregate.json"


def _parse_mode_arg(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--mode expects name=responses.json, got {value!r}")
        name, path = value.split("=", 1)
        out[name.strip()] = Path(path)
    return out


def _manual_review_markdown(pack: dict[str, Any], scored: dict[str, dict]) -> str:
    lines = [f"# Manual semantic review checklist — {pack['packId']}", ""]
    for mode, report in scored.items():
        lines.append(f"## Mode: {mode}")
        for result in report["results"]:
            if result.get("requiresManualReview") or result.get("operationalFailures") or result.get("missedRubric"):
                lines.extend([
                    f"### {result['id']}",
                    f"- Auto score: {result['score']} / {result['maxPoints']}",
                    f"- Passed: {result['passed']}",
                    f"- Missed rubric: `{json.dumps(result.get('missedRubric', []), ensure_ascii=False)}`",
                    "- Human judgement: pending",
                    "- Notes:",
                    "",
                ])
        lines.append("")
    return "\n".join(lines)


def run(pack_path: Path, mode_paths: dict[str, Path], *, out: Path, manual_out: Path | None) -> dict:
    pack = load_json(pack_path)
    errors = validate_pack(pack)
    if errors:
        raise SystemExit(json.dumps({"ok": False, "errors": errors}, indent=2, ensure_ascii=False))
    scored = {}
    for mode, path in mode_paths.items():
        responses = load_json(path)
        report = score_pack(pack, responses)
        report["mode"] = mode
        scored[mode] = report
    ordered = sorted(scored.values(), key=lambda r: (r["scorePct"], r["passed"]), reverse=True)
    best = ordered[0]["mode"] if ordered else None
    baseline = scored.get("raw") or (ordered[-1] if ordered else {"scorePct": 0, "passed": 0})
    sophia = scored.get("sophia_full") or scored.get("full") or scored.get("sophia")
    delta = None
    if sophia:
        delta = {
            "scorePctVsRaw": round(sophia["scorePct"] - baseline["scorePct"], 2),
            "strictPassVsRaw": sophia["passed"] - baseline["passed"],
        }
    aggregate = {
        "packId": pack["packId"],
        "visibility": pack.get("visibility"),
        "claimStatus": "Comparative hidden-eval aggregate. Validated claim still requires unspent third-party pack, >=3 runs where applicable, judge/manual review, and CI.",
        "modes": {mode: {k: v for k, v in report.items() if k != "results"} for mode, report in scored.items()},
        "bestMode": best,
        "sophiaDelta": delta,
        "reports": scored,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if manual_out:
        manual_out.parent.mkdir(parents=True, exist_ok=True)
        manual_out.write_text(_manual_review_markdown(pack, scored), encoding="utf-8")
    print(f"wrote {out}")
    if manual_out:
        print(f"wrote {manual_out}")
    return aggregate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--mode", action="append", default=[], help="name=responses.json; repeat for raw,raw_tools,rag_only,gate_only,sophia_full")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manual-review-out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not args.mode:
        raise SystemExit("provide at least one --mode name=responses.json")
    run(args.pack, _parse_mode_arg(args.mode), out=args.out, manual_out=args.manual_review_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
