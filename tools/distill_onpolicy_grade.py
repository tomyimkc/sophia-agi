#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase 2 (data step) of the Spark-MoE workflow: on-policy distillation grader.

The 2026 frontier distillation recipe is ON-POLICY: sample a rollout from the STUDENT,
then have a TEACHER grade + correct it, yielding a dense per-example signal (student vs
teacher-corrected) that feeds the Phase-2 weight update (DPO / corrected-SFT). This is
more sample-efficient than sparse RL rewards and complements Phase 1's teacher-authored
trajectories.

Distinct from ``tools/run_correction_loop.py`` (which drafts corrections for FAILED
benchmark cases): this grades ON-POLICY student rollouts across all three domains. The
weight UPDATE itself runs on x86 RunPod (``tools/train_lora.py``); this script only
produces the grading data.

Offline-safe: ``--student mock --teacher mock`` (CI-friendly, no keys). Metered with real
specs (``vllm:...`` student on the Spark, ``glm:glm-5.2`` teacher, etc.).

    python tools/distill_onpolicy_grade.py --student mock --teacher mock --domain all --n 2 --dry-run
    python tools/distill_onpolicy_grade.py --student vllm:Qwen/Qwen3-Next-80B-A3B --teacher glm:glm-5.2 --n 50
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUDENT_TASK_SYSTEM = "You are a student model. Attempt the task directly and concisely."

GRADE_SYSTEM = (
    "You are a strict, source-disciplined teacher grading a student's answer. Output EXACTLY:\n"
    "VERDICT: PASS\n"
    "REASONS: <one short sentence>\n"
    "CORRECTED:\n"
    "<a verified, corrected full answer; if the student was already correct, restate it cleanly>\n"
    "Provenance: never merge lineages. Math: state the final answer. Code: ensure the stated checks pass."
)


def _parse_grade(text: str) -> tuple[str, str, str]:
    """Best-effort parse of VERDICT/REASONS/CORRECTED. Unknown if markers absent."""
    verdict = "unknown"
    m = re.search(r"VERDICT:\s*(PASS|FAIL)", text or "", re.I)
    if m:
        verdict = m.group(1).lower()
    reasons = ""
    m = re.search(r"REASONS:\s*(.+)", text or "")
    if m:
        reasons = m.group(1).strip()
    corrected = ""
    m = re.search(r"CORRECTED:\s*(.*)", text or "", re.S)
    if m:
        corrected = m.group(1).strip()
    return verdict, reasons, corrected


def _rollout(student: str, prompt: str, *, max_tokens: int) -> str:
    from agent.model import complete  # noqa: PLC0415

    return complete(STUDENT_TASK_SYSTEM, prompt, spec=student, max_tokens=max_tokens)


def _grade(teacher: str, prompt: str, student_answer: str, *, max_tokens: int) -> str:
    from agent.model import complete  # noqa: PLC0415

    user = f"Task:\n{prompt}\n\nStudent answer:\n{student_answer}"
    return complete(GRADE_SYSTEM, user, spec=teacher, max_tokens=max_tokens)


def build(*, student: str, teacher: str, domain: str, n: int, out_path: Path,
          max_tokens: int, dry_run: bool) -> dict:
    """Grade up to ``n`` on-policy student rollouts per selected domain; append JSONL records."""
    from tools.build_distillation_corpus import _SEEDS, _domains  # reuse the Phase-1 seed set

    records: list[dict] = []
    for dom in _domains(domain):
        prompts = _SEEDS[dom]
        for i in range(n):
            prompt = prompts[i % len(prompts)]
            rec: dict = {"domain": dom, "prompt": prompt, "studentSpec": student, "teacherSpec": teacher}
            if dry_run:
                rec.update({"studentAnswer": "", "verdict": "dry-run", "reasons": "", "corrected": ""})
                records.append(rec)
                continue
            try:
                student_ans = _rollout(student, prompt, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001
                rec.update({"studentAnswer": "", "error": f"student: {type(exc).__name__}: {exc}"})
                records.append(rec)
                continue
            try:
                grade = _grade(teacher, prompt, student_ans, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001
                rec.update({"studentAnswer": student_ans, "error": f"teacher: {type(exc).__name__}: {exc}"})
                records.append(rec)
                continue
            verdict, reasons, corrected = _parse_grade(grade)
            rec.update({"studentAnswer": student_ans, "verdict": verdict,
                        "reasons": reasons, "corrected": corrected})
            records.append(rec)

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"student": student, "teacher": teacher, "domain": domain, "n": n,
            "outPath": str(out_path), "records": len(records), "dryRun": dry_run,
            "passCount": sum(1 for r in records if r.get("verdict") == "pass"),
            "failCount": sum(1 for r in records if r.get("verdict") == "fail"),
            "unknownCount": sum(1 for r in records if r.get("verdict") in {"unknown", "dry-run"}),
            "errors": sum(1 for r in records if "error" in r)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="mock", help="student model spec (default mock; e.g. vllm:Qwen/Qwen3-Next-80B-A3B)")
    ap.add_argument("--teacher", default="mock", help="teacher model spec (default mock; e.g. glm:glm-5.2)")
    ap.add_argument("--domain", default="all", help="provenance | math | code | all")
    ap.add_argument("--n", type=int, default=4, help="rollouts per domain")
    ap.add_argument("--out", type=Path, default=ROOT / "training" / "distillation_onpolicy" / "grades.jsonl")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true", help="plan only; no model calls, no file writes")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = build(student=args.student, teacher=args.teacher, domain=args.domain, n=args.n,
                out_path=args.out, max_tokens=args.max_tokens, dry_run=args.dry_run)
    res["elapsedSec"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
