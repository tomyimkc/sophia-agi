#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run objective coding evals through Sophia's executable code verifier.

JSONL case format:
{"id":"case1", "prompt":"...", "response":"```python\n...\n```", "verifier":"code_tests_pass", "timeoutSec":30}

For model-generated runs, provide a responses JSON mapping {case_id: response};
otherwise the committed case.response is used. This makes the lane CI-friendly
without pretending a model was evaluated.
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

from agent.verifiers import code_tests_pass  # noqa: E402

DEFAULT_CASES = ROOT / "eval" / "coding" / "smoke.jsonl"
DEFAULT_OUT = ROOT / "eval" / "results" / "coding_eval.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_responses(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "responses" in obj:
        obj = obj["responses"]
    return {str(k): str(v) for k, v in obj.items()}


def run(cases_path: Path, *, responses_path: Path | None, out: Path, syntax_only: bool) -> dict:
    cases = _read_jsonl(cases_path)
    responses = _load_responses(responses_path)
    rows = []
    passed = 0
    for case in cases:
        cid = str(case.get("id"))
        response = responses.get(cid, str(case.get("response", "")))
        verifier = code_tests_pass(
            timeout_sec=int(case.get("timeoutSec", 30)),
            allow_execution=not syntax_only,
        )
        result = verifier(response, case, {})
        ok = bool(result.get("passed"))
        passed += int(ok)
        rows.append({
            "id": cid,
            "passed": ok,
            "reasons": result.get("reasons", []),
            "detail": result.get("detail", {}),
            "prompt": case.get("prompt", ""),
        })
    report = {
        "benchmark": "coding-executable-verifier",
        "cases": str(cases_path),
        "n": len(cases),
        "passed": passed,
        "passRate": round(passed / len(cases), 4) if cases else 0.0,
        "mode": "syntax-only" if syntax_only else "execute-tempdir",
        "claimStatus": "Harness evidence only unless run on a held-out coding pack with model-generated responses.",
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("CODING EVAL PASS ✓" if passed == len(cases) else "CODING EVAL FAIL ✗")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--responses", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--syntax-only", action="store_true", help="compile code but do not execute it")
    args = ap.parse_args(argv)
    report = run(args.cases, responses_path=args.responses, out=args.out, syntax_only=args.syntax_only)
    return 0 if report["passed"] == report["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
