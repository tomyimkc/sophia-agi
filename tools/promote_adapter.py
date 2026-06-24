#!/usr/bin/env python3
"""C1 — wire local-training eval-ladder output through the W2 bounded-RSI promotion gate.

This converts the *hand-written* promotion note (the failure-ledger entry
`local-sophia-v2-mlx-trained-not-promoted-2026-06-24`) into a **reproducible,
machine-checked** W2 artifact. It:

1. reads the baseline-vs-adapter eval ladder (`eval_ladder_*.json`),
2. builds an `UpdateCandidate` with one suite per domain (Religion/History marked
   **protected**) plus a `total` target suite,
3. attaches the dataset contamination flag from the build manifest and the run artifacts,
4. runs an **independent** `formal_verifier` lattice proof of the protected-floor safety
   invariant ("no protected suite falls below incumbent − tolerance"),
5. calls `agent.continual_plasticity.evaluate_update`, and
6. writes the resulting `PromotionDecision` to a public report.

No weights change here; no GPU. This is a verified *promotion* decision — **not** a verified
capability and **not** an AGI claim. External gates still enforce correctness.
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

from agent.continual_plasticity import (  # noqa: E402
    EvalMetric,
    Goal,
    RetentionEvidence,
    UpdateCandidate,
    evaluate_update,
    evaluate_update_multigoal,
)
from agent.formal_verifier import check_lattice_consistency  # noqa: E402

DEFAULT_PROTECTED = ("religion", "history")


def _parse_goals(raw: list[str], *, default_min_delta: float) -> tuple[Goal, ...]:
    """Parse repeated --goal "suite[:min_delta]" flags into a Goal tuple.

    A goal with no explicit delta inherits --min-target-delta; use ":0" for a hold-steady
    goal (must not regress, need not improve).
    """
    goals: list[Goal] = []
    for item in raw:
        suite, _, delta = item.partition(":")
        suite = suite.strip()
        if not suite:
            raise SystemExit(f"invalid --goal {item!r}: empty suite")
        try:
            min_delta = float(delta) if delta.strip() else default_min_delta
        except ValueError:
            raise SystemExit(f"invalid --goal {item!r}: min_delta must be a number")
        goals.append(Goal(suite=suite, min_delta=min_delta))
    return tuple(goals)


def _retention_from_shift_report(report: dict[str, Any], *, source: str) -> RetentionEvidence:
    """Build a RetentionEvidence from a learning-under-shift public report.

    Reads the old-task stability the shift protocol already measured so the
    promotion gate can refuse to reward catastrophic forgetting.
    """
    return RetentionEvidence(
        old_benchmark_delta_pct=report.get("oldBenchmarkDeltaPct"),
        passing_signal=report.get("passingSignal"),
        evaluable=str(report.get("stabilityEvaluable", "evaluated")),
        source=source,
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rung(ladder: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Return the per-domain score map + total for a named ladder rung, as fractions."""
    for r in ladder.get("rungs", []):
        if r.get("rung") == name:
            s = r.get("summary", {})
            domains = {k: float(v.get("score_pct", 0.0)) / 100.0 for k, v in s.get("domains", {}).items()}
            total = float(s.get("score_pct", 0.0)) / 100.0
            return {"domains": domains, "total": total, "gateFailures": s.get("gateFailures")}
    return None


def _formal_protected_floor_proof(
    metrics: list[EvalMetric], *, tolerance: float
) -> dict[str, Any]:
    """Independent proof that no protected suite regresses below (incumbent − tolerance).

    Encoded as integer-rank ordering constraints over the formal verifier's decidable
    lattice check, so the proof is exact whether or not z3 is installed. `accepted` means
    the protected-floor invariant holds; `rejected` means a protected suite breaches it.
    """
    assignments: dict[str, int] = {}
    constraints: list[tuple[str, str, str]] = []
    for m in metrics:
        if not m.protected:
            continue
        after_rank = round(float(m.after) * 1000)
        floor_rank = round((float(m.before) - tolerance) * 1000)
        var = f"after_{m.suite}"
        assignments[var] = after_rank
        constraints.append((var, ">=", str(floor_rank)))
    if not constraints:
        return {"verdict": "accepted", "status": "no_protected_suites",
                "reasons": ["no protected suites to constrain"], "backend": "n/a"}
    return check_lattice_consistency(assignments, constraints)


def build_candidate(
    *,
    candidate_id: str,
    kind: str,
    baseline_ladder: dict[str, Any] | None,
    adapter_ladder: dict[str, Any],
    contaminated: bool,
    protected: tuple[str, ...],
    extra_artifacts: tuple[str, ...],
    proof_tag: str,
) -> UpdateCandidate:
    # "before" prefers a standalone baseline ladder; falls back to the adapter ladder's
    # own `base` rung (identical numbers, same eval harness).
    before = (_rung(baseline_ladder, "base") if baseline_ladder else None) or _rung(adapter_ladder, "base")
    after = _rung(adapter_ladder, "adapter")
    if before is None or after is None:
        raise SystemExit("could not find both a 'base' and an 'adapter' rung in the ladders")

    protected_set = {p.lower() for p in protected}
    metrics: list[EvalMetric] = []
    for dom in sorted(set(before["domains"]) | set(after["domains"])):
        metrics.append(
            EvalMetric(
                suite=dom,
                before=before["domains"].get(dom, 0.0),
                after=after["domains"].get(dom, 0.0),
                protected=dom.lower() in protected_set,
            )
        )
    metrics.append(EvalMetric(suite="total", before=before["total"], after=after["total"]))

    artifacts = tuple(a for a in (*extra_artifacts, proof_tag) if a)
    return UpdateCandidate(
        id=candidate_id,
        kind=kind,
        metrics=tuple(metrics),
        verifier_artifacts=artifacts,
        contaminated=contaminated,
        notes="local-Sophia MLX adapter promotion via eval ladder; external gates still enforce truth.",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a trained adapter through the W2 promotion gate.")
    ap.add_argument("--adapter-ladder", default="training/local_sophia_v2/eval_ladder_adapter.json")
    ap.add_argument("--baseline-ladder", default="training/local_sophia_v2/eval_ladder_baseline.json")
    ap.add_argument("--manifest", default="training/local_sophia_v2/manifest.json")
    ap.add_argument("--out", default="agi-proof/continual-plasticity/sophia-v2-promotion.public-report.json")
    ap.add_argument("--candidate-id", default="local-sophia-v2-mlx")
    ap.add_argument("--kind", default="lora_adapter")
    ap.add_argument("--target-suite", default="total")
    ap.add_argument("--protected", default=",".join(DEFAULT_PROTECTED),
                    help="comma-separated suites that may not regress beyond tolerance")
    ap.add_argument("--min-target-delta", type=float, default=0.03)
    ap.add_argument("--goal", action="append", default=[], metavar="SUITE[:MIN_DELTA]",
                    help="multi-goal mode: a suite that must improve by >= MIN_DELTA (default "
                         "--min-target-delta); repeatable. Promotion requires every goal to clear "
                         "its floor with no regression on any goal or protected suite (Pareto).")
    ap.add_argument("--max-protected-regression", type=float, default=0.01)
    ap.add_argument("--require-artifacts", type=int, default=2)
    ap.add_argument("--shift-report", default=None,
                    help="learning-under-shift public report; its old-task stability gates "
                         "promotion so an adapter that catastrophically forgets cannot promote")
    ap.add_argument("--max-retention-regression", type=float, default=5.0,
                    help="max tolerated old-task regression in percentage points (default 5.0, "
                         "matching the learning-under-shift stability rule)")
    ap.add_argument("--require-retention", action="store_true",
                    help="quarantine unless a verifiable old-task retention signal is supplied")
    ap.add_argument("--extra-artifact", action="append", default=[],
                    help="extra verifier-artifact identifiers (e.g. a SEIB report path)")
    ap.add_argument("--fail-on-reject", action="store_true",
                    help="exit non-zero unless the verdict is 'promote' (for CI gating)")
    ap.add_argument("--dry-run", action="store_true", help="evaluate and print; do not write the report")
    args = ap.parse_args(argv)

    adapter_ladder = _load(ROOT / args.adapter_ladder)
    baseline_path = ROOT / args.baseline_ladder
    baseline_ladder = _load(baseline_path) if baseline_path.exists() else None

    manifest_path = ROOT / args.manifest
    contaminated = False
    if manifest_path.exists():
        contaminated = not bool(_load(manifest_path).get("contamination", {}).get("clean", True))

    protected = tuple(p.strip() for p in args.protected.split(",") if p.strip())

    # Old-task retention from a learning-under-shift report (if supplied). This lets the
    # gate fail closed on catastrophic forgetting instead of rewarding a target-suite gain
    # that was bought by destroying previously-learned knowledge.
    retention: RetentionEvidence | None = None
    if args.shift_report:
        shift_path = ROOT / args.shift_report
        if shift_path.exists():
            retention = _retention_from_shift_report(_load(shift_path), source=args.shift_report)
        elif args.require_retention:
            raise SystemExit(f"--shift-report not found: {args.shift_report}")

    # Collect real, on-disk artifacts so the artifact count reflects genuine evidence.
    extra: list[str] = []
    for cand in (args.adapter_ladder, args.baseline_ladder, args.manifest, args.shift_report, *args.extra_artifact):
        if cand and (ROOT / cand).exists():
            extra.append(cand)

    # Build provisional metrics for the formal proof first (proof outcome becomes an artifact).
    provisional = build_candidate(
        candidate_id=args.candidate_id, kind=args.kind, baseline_ladder=baseline_ladder,
        adapter_ladder=adapter_ladder, contaminated=contaminated, protected=protected,
        extra_artifacts=(), proof_tag="",
    )
    proof = _formal_protected_floor_proof(list(provisional.metrics), tolerance=args.max_protected_regression)
    proof_tag = f"formal_verifier:protected_floor[{proof.get('backend')}:{proof.get('verdict')}]"

    candidate = build_candidate(
        candidate_id=args.candidate_id, kind=args.kind, baseline_ladder=baseline_ladder,
        adapter_ladder=adapter_ladder, contaminated=contaminated, protected=protected,
        extra_artifacts=tuple(extra), proof_tag=proof_tag,
    )

    goals = _parse_goals(args.goal, default_min_delta=args.min_target_delta)
    if goals:
        decision = evaluate_update_multigoal(
            candidate,
            goals=goals,
            max_regression=args.max_protected_regression,
            require_artifacts=args.require_artifacts,
            retention=retention,
            max_retention_regression_pct=args.max_retention_regression,
            require_retention=args.require_retention,
        )
    else:
        decision = evaluate_update(
            candidate,
            target_suite=args.target_suite,
            min_target_delta=args.min_target_delta,
            max_protected_regression=args.max_protected_regression,
            require_artifacts=args.require_artifacts,
            retention=retention,
            max_retention_regression_pct=args.max_retention_regression,
            require_retention=args.require_retention,
        )

    # The formal proof is an INDEPENDENT check: if it rejects the protected floor, the
    # promotion may not stand even if the scorecard somehow cleared it. Fail closed.
    final_verdict = decision.verdict
    if proof.get("verdict") == "rejected" and final_verdict == "promote":
        final_verdict = "reject"

    report = {
        "schema": "sophia.adapter_promotion.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "claimBoundary": "Machine-checked PROMOTION decision for a local adapter; not a "
                         "validated capability and not an AGI claim. External gates enforce truth.",
        "candidateId": args.candidate_id,
        "verdict": final_verdict,
        "scorecardVerdict": decision.verdict,
        "formalProof": {
            "invariant": "no protected suite regresses below (incumbent - tolerance)",
            "tolerance": args.max_protected_regression,
            "protectedSuites": list(protected),
            **proof,
        },
        "retention": {
            "invariant": "old-task score may not regress below baseline by more than tolerance "
                         "(no catastrophic forgetting)",
            "tolerancePct": args.max_retention_regression,
            "required": args.require_retention,
            "shiftReport": args.shift_report if retention else None,
            "oldBenchmarkDeltaPct": retention.old_benchmark_delta_pct if retention else None,
            "passingSignal": retention.passing_signal if retention else None,
            "evaluable": retention.evaluable if retention else "not-provided",
            "forgetting": bool(retention and retention.forgot(args.max_retention_regression)),
        },
        "decision": decision.to_dict(),
        "inputs": {
            "adapterLadder": args.adapter_ladder,
            "baselineLadder": args.baseline_ladder if baseline_ladder else None,
            "manifest": args.manifest if manifest_path.exists() else None,
            "shiftReport": args.shift_report if retention else None,
            "datasetContaminated": contaminated,
        },
    }

    print(f"candidate:        {args.candidate_id}")
    if goals:
        print(f"mode:             multigoal ({len(goals)} goals; Pareto, max reg -{args.max_protected_regression:.4f})")
        for g in decision.metrics["goals"]:
            d = "n/a" if g["delta"] is None else f"{g['delta']:+.4f}"
            print(f"  goal {g['suite']:<14} delta={d}  floor>={g['minDelta']:+.4f}  cleared={g['clearedFloor']}")
        if decision.metrics["paretoViolations"]:
            print(f"  Pareto violations: {', '.join(decision.metrics['paretoViolations'])}")
    else:
        print(f"target suite:     {args.target_suite}  delta={decision.metrics['targetDelta']:+.4f}")
        print(f"max protected reg:{decision.metrics['maxProtectedRegression']:+.4f}")
    ret = decision.metrics["retention"]
    if ret["evaluable"] == "not-provided":
        print("old-task retention:no shift report supplied (retention unverified)")
    else:
        print(f"old-task retention:{ret['oldBenchmarkDeltaPct']:+.2f}pp (tol -{args.max_retention_regression:.2f}pp) "
              f"forgetting={ret['forgetting']}")
    print(f"formal proof:     {proof.get('verdict')} ({proof.get('backend')}) — {'; '.join(proof.get('reasons', []))}")
    print(f"scorecard verdict:{decision.verdict}")
    print(f"FINAL VERDICT:    {final_verdict}")
    for r in decision.reasons:
        print(f"  - {r}")

    if not args.dry_run:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    if args.fail_on_reject and final_verdict != "promote":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
