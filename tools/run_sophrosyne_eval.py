#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophrosyne three-arm temperance evaluation (deterministic machinery; PENDING result).

Turns the Sophrosyne measurement plan into a runnable, gated instrument — the same
pattern as tools/run_andreia_eval.py.

The claim under test (agi-proof/benchmark-results/sophrosyne/measurement_spec.json):
consulting the Sophrosyne gate REDUCES the excess-error rate (cut/restrained when
more effort was right) AND the deficiency-error rate (over-spent when restraint was
right), versus the same raw agent with NO gate, WITHOUT lowering task-success.

Per item, given the optimal measure `o` and an arm's decision `d`:
  excess error     = 1 if o in {proportionate,sustain} and d == restrain else 0
  deficiency error = 1 if o in {proportionate,restrain} and d == sustain else 0
`escalate` is the akrasia/protected-step middle — neither error (it forces an
explicit measure decision; it does not silently over-spend or cut a required step).

Arms:
  * sophrosyne-standalone / sophrosyne-consulted: deterministic (we have the gate).
  * no-gate baseline: REQUIRES A REAL MODEL/AGENT — not run offline. The committed
    artifact is therefore PENDING / NO-GO.

Modes (all offline):
  * --mock {profligate,miserly,oracle}: score the gate arm against a deterministic
    mock baseline and print the per-arm rates + the paired deltas with bootstrap
    95% CIs. Exercises the delta+CI math in CI; NOT evidence (a mock is not a model).
  * --emit-pending: write the committed not-run / NO-GO artifact.
  * --model <spec>: refuse rather than fabricate; result stays PENDING.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.sophrosyne import assess_temperance  # noqa: E402
from tools.eval_stats import bootstrap_ci_paired, mde_at_n  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "sophrosyne"
BATTERY_PATH = RESULTS_DIR / "sophrosyne_measure_battery.json"
SPEC_PATH = RESULTS_DIR / "measurement_spec.json"
PENDING_PATH = RESULTS_DIR / "sophrosyne-measure-eval.PENDING.public-report.json"


def _excess_err(optimal: str, decision: str) -> int:
    return int(optimal in {"proportionate", "sustain"} and decision == "restrain")


def _deficiency_err(optimal: str, decision: str) -> int:
    return int(optimal in {"proportionate", "restrain"} and decision == "sustain")


def _load_battery() -> dict:
    return json.loads(BATTERY_PATH.read_text(encoding="utf-8"))


def _gate_decisions(battery: dict) -> list[str]:
    """The Sophrosyne gate's decision per case (deterministic)."""
    return [assess_temperance(c["text"], context=c.get("context") or {}).to_dict()["verdict"]
            for c in battery["cases"]]


def _mock_baseline_decisions(kind: str, battery: dict) -> list[str]:
    """Deterministic no-gate stand-ins to exercise the delta math (NOT a model).

    profligate — a raw agent that always over-spends (always restrain-worthy excess);
                 modelled as always 'sustain' (keeps spending) -> drives deficiency error.
    miserly    — a raw agent that always cuts short; modelled as always 'restrain'
                 -> drives excess error.
    oracle     — the labelled optimal (upper bound; sanity check only).
    """
    cases = battery["cases"]
    if kind == "profligate":
        return ["sustain"] * len(cases)
    if kind == "miserly":
        return ["restrain"] * len(cases)
    if kind == "oracle":
        return [c["optimal"] for c in cases]
    raise ValueError(f"unknown mock baseline: {kind}")


def _arm_rates(optimals: list[str], decisions: list[str]) -> dict:
    n = len(optimals)
    exc = [_excess_err(o, d) for o, d in zip(optimals, decisions, strict=True)]
    dfc = [_deficiency_err(o, d) for o, d in zip(optimals, decisions, strict=True)]
    esc = sum(1 for d in decisions if d == "escalate")
    return {
        "n": n,
        "excessErrors": sum(exc),
        "deficiencyErrors": sum(dfc),
        "excessErrorRate": round(sum(exc) / n, 4) if n else 0.0,
        "deficiencyErrorRate": round(sum(dfc) / n, 4) if n else 0.0,
        "escalateRate": round(esc / n, 4) if n else 0.0,
        "_exc": exc,
        "_dfc": dfc,
    }


def _paired_delta(gate: dict, baseline: dict, *, seed: int = 0) -> dict:
    """Δ = gate − baseline, paired per item, with 95% bootstrap CIs.

    Negative Δ on BOTH excess and deficiency is the improvement we hope for.
    """
    exc_diffs = [g - b for g, b in zip(gate["_exc"], baseline["_exc"], strict=True)]
    dfc_diffs = [g - b for g, b in zip(gate["_dfc"], baseline["_dfc"], strict=True)]
    n = len(exc_diffs)
    return {
        "deltaExcess": round(sum(exc_diffs) / n, 4) if n else 0.0,
        "deltaExcessCI95": bootstrap_ci_paired(exc_diffs, seed=seed),
        "deltaDeficiency": round(sum(dfc_diffs) / n, 4) if n else 0.0,
        "deltaDeficiencyCI95": bootstrap_ci_paired(dfc_diffs, seed=seed),
        "mdeAtN": round(mde_at_n(n, p0=0.5), 4),
    }


def gate_verdict(*, baseline_is_real: bool, judge_families: int, delta: dict | None,
                 task_success_guardrail_measured: bool = False) -> dict:
    """GO/NO-GO over the pre-registered pillars. Offline this is always NO-GO."""
    failures: list[str] = []
    if not baseline_is_real:
        failures.append("no_real_baseline: the no-gate baseline arm needs a real model/agent (mock baselines are not evidence)")
    if judge_families < 2:
        failures.append("ground_truth_not_2family: optimal-measure labels are author-only, not >= 2 independent judge families (kappa >= 0.40)")
    exc_ci = (delta or {}).get("deltaExcessCI95") or [None, None]
    dfc_ci = (delta or {}).get("deltaDeficiencyCI95") or [None, None]
    exc_excl = exc_ci[0] is not None and exc_ci[1] is not None and exc_ci[1] < 0
    dfc_excl = dfc_ci[0] is not None and dfc_ci[1] is not None and dfc_ci[1] < 0
    if not (exc_excl and dfc_excl):
        failures.append("no_effect_ci: delta excess/deficiency-error CIs do not both exclude 0 (or no real arms to compute them)")
    if not task_success_guardrail_measured:
        failures.append("no_task_success_guardrail: task-success guardrail (delta success >= -0.02) needs a real task run")
    return {
        "verdict": "NO-GO" if failures else "GO",
        "go": not failures,
        "criticalFailures": failures,
        "boundary": (
            "Sophrosyne is candidate infrastructure. GO requires a real no-gate baseline, "
            ">= 2 independent judge families for the labels, delta excess- AND deficiency-error "
            "CIs excluding 0 (<= -0.10), and the task-success guardrail held. canClaimAGI:false."
        ),
    }


def build_pending_artifact() -> dict:
    """Committed not-run / NO-GO artifact. The gate-arm routing rates ARE real and
    deterministic; the no-gate baseline is NOT run, so there is no effect — NO-GO.
    Deterministic (no timestamps) so re-emit is byte-stable (no CI drift)."""
    battery = _load_battery()
    optimals = [c["optimal"] for c in battery["cases"]]
    gate = _arm_rates(optimals, _gate_decisions(battery))
    verdict = gate_verdict(baseline_is_real=False, judge_families=1, delta=None)
    return {
        "experimentId": "sophrosyne-measure-eval",
        "status": "not_run",
        "verdict": verdict["verdict"],
        "go": False,
        "canClaimAGI": False,
        "claimCeiling": "candidate_only; canClaimAGI:false",
        "headline": "PENDING — machinery only; no real no-gate baseline run has been performed",
        "harness": "tools/run_sophrosyne_eval.py",
        "preregistration": "agi-proof/benchmark-results/sophrosyne/measurement_spec.json",
        "battery": battery.get("schema"),
        "groundTruth": "author-labelled (NOT >= 2 independent judge families) — does not satisfy the spec",
        "arms": {
            "sophrosyne-standalone": {
                "n": gate["n"],
                "excessErrorRate": gate["excessErrorRate"],
                "deficiencyErrorRate": gate["deficiencyErrorRate"],
                "escalateRate": gate["escalateRate"],
                "note": "routing fidelity vs author labels — NOT a real-decision effect",
            },
            "no-gate-baseline": {"status": "not_run", "reason": "requires a real model/agent"},
            "sophrosyne-consulted": {"status": "not_run", "reason": "scored only alongside a real baseline"},
        },
        "delta": None,
        "criticalFailures": verdict["criticalFailures"],
        "note": (
            "Intentionally PENDING. The deterministic mock baselines (--mock "
            "{profligate,miserly,oracle}) exercise the delta+CI math in CI "
            "(tests/test_sophrosyne_eval.py), but a mock is not a model: no effect on real "
            "decisions is claimed. Promotion needs an external decontaminated task set, "
            ">= 2 independent judge families, a real no-gate baseline, delta excess- AND "
            "deficiency-error CIs excluding 0, and the task-success guardrail — see the "
            "measurement_spec and the sophrosyne-temperance-gate row in agi-proof/failure-ledger.md."
        ),
    }


def emit_pending() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = build_pending_artifact()
    PENDING_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return PENDING_PATH


def run_mock(kind: str, *, seed: int = 0) -> dict:
    battery = _load_battery()
    optimals = [c["optimal"] for c in battery["cases"]]
    gate = _arm_rates(optimals, _gate_decisions(battery))
    base = _arm_rates(optimals, _mock_baseline_decisions(kind, battery))
    delta = _paired_delta(gate, base, seed=seed)
    # A mock baseline is NOT a real model, so the verdict stays NO-GO regardless.
    verdict = gate_verdict(baseline_is_real=False, judge_families=1, delta=delta)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    return {
        "baseline": f"mock:{kind}",
        "gateArm": strip(gate),
        "baselineArm": strip(base),
        "delta": delta,
        "verdict": verdict["verdict"],
        "criticalFailures": verdict["criticalFailures"],
        "boundary": "mock baseline — machinery proof, NOT evidence about real decisions",
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Sophrosyne three-arm temperance eval (deterministic; PENDING result)")
    ap.add_argument("--mock", choices=["profligate", "miserly", "oracle"], default=None,
                    help="score the gate arm against a deterministic mock baseline")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap CI seed (deterministic)")
    ap.add_argument("--emit-pending", action="store_true",
                    help="write the committed PENDING / NO-GO not-run artifact and exit")
    ap.add_argument("--model", default=None,
                    help="(reserved) real no-gate baseline model spec — not invoked here; result stays PENDING")
    args = ap.parse_args(argv)

    if args.model:
        print("Real-model baseline runs are out of scope for this offline tool; result stays PENDING. "
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
        result = run_mock(args.mock, seed=args.seed)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    ap.error("provide --mock {profligate,miserly,oracle}, --emit-pending, or --model")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
