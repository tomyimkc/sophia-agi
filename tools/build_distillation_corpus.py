#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase 1 of the Spark-MoE workflow: build a teacher-distillation corpus.

Generates verified reasoning trajectories from a (frontier) teacher over the three
machine-checkable domains the repo already supports -- provenance, math, code -- and emits
them in the repo's ``training/examples`` format so they feed the Phase-2 distillation
SFT/LoRA step. This is MOPD-style (multi-teacher on-policy distillation) DATA; it is NOT
the live GRPO weight update (Phase 3).

VISION-aligned: the teacher transfers *verified, source-disciplined* reasoning. Every
trajectory is reviewable and (for provenance) must clear the attribution gate before merge
(see CONTRIBUTING.md "Phase 2 -- Claude teacher examples"). Sophia does not out-train the
frontier; it distills + RLVR-grounds it.

Offline by default: ``--teacher mock`` uses the deterministic mock provider (CI-safe, no key).
Use ``--teacher <frontier spec>`` (e.g. ``glm:glm-5.2``, ``anthropic:claude-sonnet-4-6``,
``openai:gpt-4o``) to generate real trajectories (metered).

    python tools/build_distillation_corpus.py --teacher mock --domain all --n 4 --dry-run
    python tools/build_distillation_corpus.py --teacher glm:glm-5.2 --domain provenance --n 50
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

TEACHER_SYSTEM = (
    "You are a precise, source-disciplined reasoning teacher. Produce a verified, step-by-step "
    "trajectory for the task. For provenance, separate traditional/legendary attribution from "
    "evidenced authorship and never merge intellectual lineages. For math, show the derivation "
    "and state the final answer. For code, give a correct, runnable solution that passes the "
    "stated checks. Be honest about uncertainty; never fabricate citations."
)

# Small deterministic seed prompts per domain. For real runs, scale --n and/or feed the repo's
# benchmark packs (tests/attribution_bench.json, provenance_bench data, benchmark/code_tasks.json).
_SEEDS: dict[str, list[str]] = {
    "provenance": [
        "Who actually authored the Dao De Jing, and why does the uncertainty matter for source discipline?",
        "Did Socrates write any books? Contrast with Plato's authorship of the Republic.",
        "Is the 'Mozart effect' a well-evidenced claim? Trace its provenance honestly.",
    ],
    "math": [
        "Show by induction that the sum of the first n positive integers is n(n+1)/2.",
        "Solve x^2 - 5x + 6 = 0 and verify both roots by substitution.",
        "Compute the derivative of x*ln(x) and show each step.",
    ],
    "code": [
        "Write a Python function fib(n) iteratively. Must pass: fib(0)==0, fib(1)==1, fib(10)==55.",
        "Write a Python function is_prime(n) handling n<2 and n=2. Must pass: is_prime(1)==False, is_prime(2)==True, is_prime(15)==False.",
        "Write a Python function reversing a string into a char list. Must pass: reverse('spark') == ['k','r','a','p','s'].",
    ],
}


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:n] or "prompt"


def _domains(arg: str) -> list[str]:
    if arg == "all":
        return list(_SEEDS)
    if arg in _SEEDS:
        return [arg]
    raise SystemExit(f"unknown domain {arg!r}; choose one of {','.join(list(_SEEDS) + ['all'])}")


def _teach_one(teacher: str, prompt: str, *, max_tokens: int) -> str:
    from agent.model import complete  # noqa: PLC0415 — lazy; keeps --dry-run/import light

    return complete(TEACHER_SYSTEM, prompt, spec=teacher, max_tokens=max_tokens)


def build(*, teacher: str, domain: str, n: int, out_dir: Path, max_tokens: int, dry_run: bool) -> dict:
    """Generate up to ``n`` trajectories per selected domain; emit training/examples JSONs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in out_dir.glob("*.json")}
    written: list[str] = []
    skipped: list[dict] = []
    idx = 0
    for dom in _domains(domain):
        prompts = _SEEDS[dom]
        for i in range(n):
            prompt = prompts[i % len(prompts)]  # deterministic cycle to reach n
            idx += 1
            fname = f"{idx:03d}-distill-{dom}-{_slug(prompt)}.json"
            if fname in existing:
                continue
            if dry_run:
                written.append(fname)
                continue
            try:
                answer = _teach_one(teacher, prompt, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001 — a single teacher failure must not kill the batch
                skipped.append({"file": fname, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not (answer or "").strip():
                skipped.append({"file": fname, "error": "empty teacher response"})
                continue
            example = {
                "messages": [
                    {"role": "system", "content": TEACHER_SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer.strip()},
                ],
                "metadata": {
                    "source": "distillation-teacher",
                    "project": "sophia-agi",
                    "domain": dom,
                    "teacherSpec": teacher,
                    "notes": (
                        f"Phase-1 distillation trajectory ({dom}); teacher-generated. Review per "
                        "CONTRIBUTING.md (attribution traps, confidence labels, no invented citations) "
                        "and run tools/validate_attribution.py before merge."
                    ),
                },
            }
            (out_dir / fname).write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(fname)
    return {"teacher": teacher, "domain": domain, "n": n, "outDir": str(out_dir),
            "written": written, "skipped": skipped, "dryRun": dry_run}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="mock", help="model spec (default mock=offline; e.g. glm:glm-5.2)")
    ap.add_argument("--domain", default="all", help="provenance | math | code | all")
    ap.add_argument("--n", type=int, default=4, help="trajectories per domain")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "training" / "examples")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true", help="list files it would write; no teacher calls")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = build(teacher=args.teacher, domain=args.domain, n=args.n, out_dir=args.out_dir,
                max_tokens=args.max_tokens, dry_run=args.dry_run)
    res["elapsedSec"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
