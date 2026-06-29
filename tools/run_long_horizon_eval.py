#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""CLI for the deterministic long-horizon EXECUTION eval (per-step verifiable).

This is distinct from ``tools/run_long_horizon.py`` (the autonomy/intervention LOGGER
under ``agi-proof/long-horizon-runs/``). This tool drives an agent over a suite of
dependent-step tasks whose every step is scored by a DETERMINISTIC checkpoint verifier
(never an LLM judge), and reports completion rate, step-level success, and horizon length
(longest fully-correct prefix) with CIs from ``tools/eval_stats.py``.

Modes (all offline):

  * ``--mock {perfect,fail-at-step}`` — run the deterministic mock agent over the
    synthetic example suite and print the scored result. No model, no network.
  * ``--emit-pending`` — write a PENDING artifact (status ``not_run``, ``verdict``
    ``NO-GO``, no measured numbers) under
    ``agi-proof/benchmark-results/long-horizon/``. This is the committed artifact: the
    machinery exists and is pre-registered, but a real model run has NOT been performed.

A REAL run is wired but intentionally NOT invoked here (out of scope; would need a model
adapter): pass ``--model <spec>`` together with an agent adapter to drive a live engine —
see the module docstring in ``eval/long_horizon/__init__.py``. Until then the headline is
PENDING and ``canClaimAGI`` stays false.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.long_horizon import example_tasks, run_tasks  # noqa: E402
from eval.long_horizon.tasks import FailAtStepAgent, PerfectMockAgent  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "long-horizon"
PENDING_PATH = RESULTS_DIR / "long-horizon-execution.PENDING.public-report.json"
SPEC_PATH = RESULTS_DIR / "measurement_spec.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_pending_artifact() -> dict:
    """The committed not-run / NO-GO artifact. Carries NO measured numbers — only the
    suite shape, the pre-registration pointer, and explicit not-run/NO-GO labels."""
    tasks = example_tasks()
    return {
        "experimentId": "long-horizon-execution",
        "status": "not_run",
        "verdict": "NO-GO",
        "go": False,
        "canClaimAGI": False,
        "claimCeiling": "candidate_only; canClaimAGI:false",
        "headline": "PENDING — machinery only; no real model run has been performed",
        "generatedAt": _now_iso(),
        "harness": "tools/run_long_horizon_eval.py (eval/long_horizon/)",
        "preregistration": "agi-proof/benchmark-results/long-horizon/measurement_spec.json",
        "oracle": (
            "deterministic per-step checkpoints (pure functions over agent output + "
            "threaded task state) — never an LLM judge, never self-judged"
        ),
        "constructs": ["completionRate", "stepLevelSuccess", "horizonLength"],
        "results": None,
        "taskSuite": [
            {"taskId": t.task_id, "length": t.length, "description": t.description}
            for t in tasks
        ],
        "note": (
            "This artifact is intentionally PENDING. The synthetic example suite and the "
            "deterministic mock agent exercise the harness in CI (tests/test_long_horizon_eval.py), "
            "but a real-model long-horizon result is OUT OF SCOPE and NOT claimed. "
            "Long-horizon execution is a Level-3+ gate in agi-proof/preregistered-thresholds.md; "
            "no GO until a real run clears the pre-registered thresholds."
        ),
    }


def emit_pending() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = build_pending_artifact()
    PENDING_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return PENDING_PATH


def run_mock(kind: str, *, fail_step_id: str = "mul2", seed: int = 0) -> dict:
    if kind == "perfect":
        agent = PerfectMockAgent()
    elif kind == "fail-at-step":
        agent = FailAtStepAgent(fail_step_id=fail_step_id)
    else:
        raise ValueError(f"unknown mock kind: {kind}")
    result = run_tasks(agent, example_tasks(), seed=seed)
    return result.to_dict()


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic long-horizon execution eval")
    parser.add_argument("--mock", choices=["perfect", "fail-at-step"], default=None,
                        help="Run the deterministic mock agent over the synthetic suite")
    parser.add_argument("--fail-step-id", default="mul2",
                        help="Step id where the fail-at-step mock slips (default: mul2)")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap CI seed (deterministic)")
    parser.add_argument("--emit-pending", action="store_true",
                        help="Write the committed PENDING/NO-GO not-run artifact and exit")
    parser.add_argument("--model", default=None,
                        help="(reserved) real-model spec — not invoked here; result stays PENDING")
    args = parser.parse_args(argv)

    if args.model:
        # A real run is intentionally not wired into this offline tool: refuse rather
        # than silently fabricate. The committed artifact must stay PENDING/not-run.
        print("Real-model runs are out of scope for this offline tool; result stays PENDING. "
              "Emit the pending artifact with --emit-pending.", file=sys.stderr)
        return 2

    if args.emit_pending:
        path = emit_pending()
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"Wrote PENDING (not_run / NO-GO) artifact: {shown}")
        return 0

    if args.mock:
        result = run_mock(args.mock, fail_step_id=args.fail_step_id, seed=args.seed)
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
        return 0

    parser.error("provide --mock {perfect,fail-at-step}, --emit-pending, or --model")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
