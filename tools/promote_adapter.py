#!/usr/bin/env python3
"""C1 — wire local-training eval-ladder output through the W2 bounded-RSI promotion gate.

Reads baseline-vs-adapter eval ladder (FORMAT / CONTENT / COMBINED channels), builds an
``UpdateCandidate`` with **protected suites scored on CONTENT channel**, runs the full
decidable invariant suite via ``agent.godel_oracle``, and writes a public promotion report.

Protected-floor COMBINED-only checks are retained for continuity but are **not** the gate
input — CONTENT channel + invariant oracle decide promotion.
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
from agent.godel_oracle import evaluate_for_promotion  # noqa: E402

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


def _domain_channel(dom: dict[str, Any], channel: str) -> float:
    block = dom.get(channel)
    if isinstance(block, dict):
        return float(block.get("score_pct", 0.0)) / 100.0
    if channel == "combined":
        return float(dom.get("score_pct", 0.0)) / 100.0
    return float(dom.get("score_pct", 0.0)) / 100.0


def _rung(ladder: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Return per-domain COMBINED + CONTENT maps and totals for a ladder rung."""
    for r in ladder.get("rungs", []):
        if r.get("rung") != name:
            continue
        s = r.get("summary", {})
        domains_raw = s.get("domains", {})
        combined_domains = {k: _domain_channel(v, "combined") for k, v in domains_raw.items()}
        content_domains = {k: _domain_channel(v, "content") for k, v in domains_raw.items()}
        channels = s.get("channels", {})
        if isinstance(channels.get("combined"), dict):
            total = float(channels["combined"].get("score_pct", s.get("score_pct", 0.0))) / 100.0
            content_total = float(channels.get("content", {}).get("score_pct", s.get("score_pct", 0.0))) / 100.0
        else:
            total = float(s.get("score_pct", 0.0)) / 100.0
            content_total = total
        return {
            "domains": combined_domains,
            "contentDomains": content_domains,
            "total": total,
            "contentTotal": content_total,
            "gateFailures": s.get("gateFailures"),
        }
    return None


def _formal_protected_floor_proof(
    metrics: list[EvalMetric], *, tolerance: float
) -> dict[str, Any]:
    """Legacy COMBINED protected-floor proof (continuity only; not the promotion gate)."""
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
        return {
            "verdict": "accepted",
            "status": "no_protected_suites",
            "reasons": ["no protected suites to constrain"],
            "backend": "n/a",
        }
    return check_lattice_consistency(assignments, constraints)


def _load_jsonl_traces(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


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
) -> tuple[UpdateCandidate, list[dict[str, Any]]]:
    before = (_rung(baseline_ladder, "base") if baseline_ladder else None) or _rung(adapter_ladder, "base")
    after = _rung(adapter_ladder, "adapter")
    if before is None or after is None:
        raise SystemExit("could not find both a 'base' and an 'adapter' rung in the ladders")

    protected_set = {p.lower() for p in protected}
    metrics: list[EvalMetric] = []
    content_metrics: list[dict[str, Any]] = []
    for dom in sorted(set(before["domains"]) | set(after["domains"])):
        is_protected = dom.lower() in protected_set
        before_score = (
            before["contentDomains"].get(dom, 0.0)
            if is_protected
            else before["domains"].get(dom, 0.0)
        )
        after_score = (
            after["contentDomains"].get(dom, 0.0)
            if is_protected
            else after["domains"].get(dom, 0.0)
        )
        metrics.append(
            EvalMetric(
                suite=dom,
                before=before_score,
                after=after_score,
                protected=is_protected,
            )
        )
        if is_protected:
            content_metrics.append(
                {
                    "suite": dom,
                    "contentBefore": before["contentDomains"].get(dom, 0.0),
                    "contentAfter": after["contentDomains"].get(dom, 0.0),
                }
            )
    metrics.append(
        EvalMetric(suite="total", before=before["total"], after=after["total"])
    )

    artifacts = tuple(a for a in (*extra_artifacts, proof_tag) if a)
    candidate = UpdateCandidate(
        id=candidate_id,
        kind=kind,
        metrics=tuple(metrics),
        verifier_artifacts=artifacts,
        contaminated=contaminated,
        notes=(
            "local-Sophia MLX adapter promotion via eval ladder; "
            "protected suites gated on CONTENT channel; external gates still enforce truth."
        ),
    )
    return candidate, content_metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a trained adapter through the W2 promotion gate.")
    ap.add_argument("--adapter-ladder", default="training/local_sophia_v2/eval_ladder_adapter.json")
    ap.add_argument("--baseline-ladder", default="training/local_sophia_v2/eval_ladder_baseline.json")
    ap.add_argument("--manifest", default="training/local_sophia_v2/manifest.json")
    ap.add_argument("--traces", default="training/council/religion_repair_c4.jsonl",
                    help="training traces audited by provenance_complete (empty file → zero lacking)")
    ap.add_argument("--adapter-config", default="training/mlx_adapters/sophia-v2/sophia_lora_config.json",
                    help="trainer-emitted config; its seed is recorded for reproducibility provenance")
    ap.add_argument("--out", default="agi-proof/continual-plasticity/sophia-v2-promotion.public-report.json")
    ap.add_argument("--candidate-id", default="local-sophia-v2-mlx")
    ap.add_argument("--kind", default="lora_adapter")
    ap.add_argument("--target-suite", default="total")
    ap.add_argument("--protected", default=",".join(DEFAULT_PROTECTED),
                    help="comma-separated suites that may not regress beyond tolerance on CONTENT")
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
    ap.add_argument(
        "--allow-fallback-proof",
        action="store_true",
        help="OFF by default: allow promotion without z3 solver attestation (stamps solverChecked:false)",
    )
    ap.add_argument("--dry-run", action="store_true", help="evaluate and print; do not write the report")
    args = ap.parse_args(argv)

    adapter_ladder = _load(ROOT / args.adapter_ladder)
    baseline_path = ROOT / args.baseline_ladder
    baseline_ladder = _load(baseline_path) if baseline_path.exists() else None

    manifest_path = ROOT / args.manifest
    manifest = _load(manifest_path) if manifest_path.exists() else None
    contaminated = False
    if manifest is not None:
        contaminated = not bool(manifest.get("contamination", {}).get("clean", True))

    traces_path = ROOT / args.traces
    traces = _load_jsonl_traces(traces_path) if traces_path.exists() else []

    protected = tuple(p.strip() for p in args.protected.split(",") if p.strip())

    training_seed = None
    cfg_path = ROOT / args.adapter_config
    if cfg_path.exists():
        try:
            training_seed = _load(cfg_path).get("seed")
        except Exception:  # noqa: BLE001 - config provenance is best-effort
            training_seed = None

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
    for cand in (args.adapter_ladder, args.baseline_ladder, args.manifest, args.adapter_config, args.shift_report, *args.extra_artifact):
        if cand and (ROOT / cand).exists():
            extra.append(cand)
    if training_seed is not None:
        extra.append(f"training_seed:{training_seed}")

    before_rung = (_rung(baseline_ladder, "base") if baseline_ladder else None) or _rung(adapter_ladder, "base")
    after_rung = _rung(adapter_ladder, "adapter")
    if before_rung is None or after_rung is None:
        raise SystemExit("could not find both a 'base' and an 'adapter' rung in the ladders")

    candidate, content_metrics = build_candidate(
        candidate_id=args.candidate_id,
        kind=args.kind,
        baseline_ladder=baseline_ladder,
        adapter_ladder=adapter_ladder,
        contaminated=contaminated,
        protected=protected,
        extra_artifacts=(),
        proof_tag="",
    )

    legacy_proof = _formal_protected_floor_proof(list(candidate.metrics), tolerance=args.max_protected_regression)

    oracle_ok, oracle_bundle, oracle_path = evaluate_for_promotion(
        candidate_id=args.candidate_id,
        protected_content_metrics=content_metrics,
        before_total=before_rung["total"],
        after_total=after_rung["total"],
        manifest=manifest,
        traces=traces,
        tolerance=args.max_protected_regression,
        protected_suites=protected,
        input_summary={
            "adapterLadder": args.adapter_ladder,
            "baselineLadder": args.baseline_ladder if baseline_ladder else None,
            "manifest": args.manifest if manifest_path.exists() else None,
            "traces": args.traces if traces_path.exists() else None,
            "protectedChannel": "content",
            "allowFallbackProof": args.allow_fallback_proof,
        },
        allow_fallback_proof=args.allow_fallback_proof,
    )

    proof_tag = (
        f"invariant_oracle:{oracle_path.relative_to(ROOT)}"
        if oracle_path.is_relative_to(ROOT)
        else f"invariant_oracle:{oracle_path}"
    )
    candidate = build_candidate(
        candidate_id=args.candidate_id,
        kind=args.kind,
        baseline_ladder=baseline_ladder,
        adapter_ladder=adapter_ladder,
        contaminated=contaminated,
        protected=protected,
        extra_artifacts=tuple(extra),
        proof_tag=proof_tag,
    )[0]

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

    final_verdict = decision.verdict
    if not oracle_ok and final_verdict == "promote":
        final_verdict = "reject"
    if legacy_proof.get("verdict") == "rejected" and final_verdict == "promote":
        final_verdict = "reject"

    report = {
        "schema": "sophia.adapter_promotion.v2",
        "candidateOnly": True,
        "level3Evidence": False,
        "claimBoundary": (
            "Machine-checked PROMOTION decision for a local adapter; protected suites use "
            "CONTENT channel; invariant oracle required. Not AGI proof."
        ),
        "candidateId": args.candidate_id,
        "verdict": final_verdict,
        "solverChecked": oracle_bundle.get("solverChecked", False),
        "scorecardVerdict": decision.verdict,
        "oraclePromote": oracle_ok,
        "oracleProofPath": str(oracle_path.relative_to(ROOT)) if oracle_path.is_relative_to(ROOT) else str(oracle_path),
        "breachingInvariants": oracle_bundle.get("breachingInvariants", []),
        "formalProofLegacyCombined": {
            "invariant": "legacy COMBINED protected floor (continuity only)",
            "tolerance": args.max_protected_regression,
            "protectedSuites": list(protected),
            **legacy_proof,
        },
        "invariantOracle": oracle_bundle,
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
            "traces": args.traces if traces_path.exists() else None,
            "shiftReport": args.shift_report if retention else None,
            "datasetContaminated": contaminated,
            "trainingSeed": training_seed,
            "protectedChannel": "content",
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
        print(f"max protected reg:{decision.metrics['maxProtectedRegression']:+.4f} (CONTENT channel)")
    ret = decision.metrics["retention"]
    if ret["evaluable"] == "not-provided":
        print("old-task retention:no shift report supplied (retention unverified)")
    else:
        print(f"old-task retention:{ret['oldBenchmarkDeltaPct']:+.2f}pp (tol -{args.max_retention_regression:.2f}pp) "
              f"forgetting={ret['forgetting']}")
    print(f"oracle promote:   {oracle_ok}  breaching={oracle_bundle.get('breachingInvariants')}")
    print(f"solver checked:   {oracle_bundle.get('solverChecked')}")
    if oracle_bundle.get("solverNotes"):
        for note in oracle_bundle["solverNotes"]:
            print(f"  solver note: {note}")
    for name, inv in oracle_bundle.get("invariants", {}).items():
        print(f"  {name}: {inv.get('verdict')} ({inv.get('backend')})")
    print(f"scorecard verdict:{decision.verdict}")
    print(f"FINAL VERDICT:    {final_verdict}")
    for r in decision.reasons:
        print(f"  - {r}")

    if not args.dry_run:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
        print(f"wrote {oracle_path}")

    if args.fail_on_reject and final_verdict != "promote":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
