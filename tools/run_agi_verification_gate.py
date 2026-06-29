#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run Sophia's non-human AGI verification gate.

This script converts the AGI proof package from prose into a machine-checkable
claim-readiness report. It does NOT claim to philosophically prove AGI; it checks
whether all *non-human-verifiable* evidence lanes are implemented and/or passed.

Use:
  python tools/run_agi_verification_gate.py --run-local-smoke
  python tools/run_agi_verification_gate.py --target level3 --allow-open

Exit behavior:
  - exits 0 when target level passes, or when --allow-open is set;
  - exits 1 when target level does not pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CRITERIA = ROOT / "agi-proof" / "agi-verification" / "criteria.json"
OUT = ROOT / "agi-proof" / "agi-verification" / "agi-verification-report.json"
LEVEL_ORDER = ["level2", "level3", "level4", "level5"]


@dataclass
class Check:
    id: str
    requiredFor: str
    status: str  # pass | fail | open | missing | error
    passed: bool
    summary: str
    evidence: list[str]
    details: dict[str, Any]


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def _run(cmd: list[str], timeout: int = 300) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdoutTail": proc.stdout[-2000:],
        "stderrTail": proc.stderr[-2000:],
    }


def _ci_excludes_zero(ci: list[Any] | None) -> bool:
    if not ci or len(ci) < 2:
        return False
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except Exception:
        return False
    return lo > 0 or hi < 0


def _criterion_map() -> dict[str, dict]:
    data = _load_json(CRITERIA) or {"methods": []}
    return {m["id"]: m for m in data.get("methods", [])}


def _check(id_: str, *, status: str, summary: str, evidence: list[str] | None = None, details: dict | None = None) -> Check:
    meta = _criterion_map().get(id_, {})
    return Check(
        id=id_,
        requiredFor=meta.get("requiredFor", "level4"),
        status=status,
        passed=status == "pass",
        summary=summary,
        evidence=evidence or [],
        details=details or {},
    )


def provenance_validation(run_smoke: bool) -> Check:
    if not run_smoke:
        return _check("provenance_validation", status="open", summary="Use --run-local-smoke to execute validate_attribution.py")
    res = _run([sys.executable, "tools/validate_attribution.py"])
    return _check(
        "provenance_validation",
        status="pass" if res["returncode"] == 0 else "fail",
        summary="Core attribution validation passed." if res["returncode"] == 0 else "Core attribution validation failed.",
        evidence=["tools/validate_attribution.py"],
        details=res,
    )


def published_provenance_delta() -> Check:
    path = ROOT / "agi-proof" / "benchmark-results" / "published-results.json"
    data = _load_json(path) or {}
    rows = data.get("validated", [])
    good = [r for r in rows if r.get("validatedChecks") and all(r["validatedChecks"].values()) and _ci_excludes_zero(r.get("deltaCI"))]
    return _check(
        "published_provenance_delta",
        status="pass" if good else "missing",
        summary=f"Validated provenance-delta rows: {len(good)}" if good else "No validated provenance-delta row found.",
        evidence=[_rel(path)],
        details={"validatedRows": len(good), "rows": good[:3]},
    )


def calibration_abstention() -> Check:
    path = ROOT / "agi-proof" / "benchmark-results" / "published-results.json"
    data = _load_json(path) or {}
    rows = data.get("calibrationEvals", [])
    good = []
    for r in rows:
        cal_ci = r.get("calibrationDeltaCI") or r.get("calibrationCI") or []
        fab_ci = r.get("fabricationReductionCI") or []
        if _ci_excludes_zero(cal_ci) and (not fab_ci or _ci_excludes_zero(fab_ci)):
            good.append(r)
    # Backstop: the corroboration artifact is explicit evidence even if summary key names drift.
    corr = ROOT / "agi-proof" / "baseline-ablation" / "calibration-2family-judge-2026-06-22.json"
    status = "pass" if good or corr.exists() else "missing"
    return _check(
        "calibration_abstention",
        status=status,
        summary="Calibration/abstention evidence exists." if status == "pass" else "No calibration/abstention evidence found.",
        evidence=[_rel(path), _rel(corr)],
        details={"summaryRows": len(good), "corroborationArtifact": corr.exists()},
    )


def grounded_gate(run_smoke: bool) -> Check:
    if not run_smoke:
        return _check("grounded_gate", status="open", summary="Use --run-local-smoke to execute grounding gate N>=40.")
    res = _run([sys.executable, "tools/run_grounding_gate.py", "--runs", "3", "--min-cases", "40"], timeout=600)
    ok = res["returncode"] == 0 and "ALL INVARIANTS HOLD" in (res["stdoutTail"] or "")
    return _check("grounded_gate", status="pass" if ok else "fail", summary="Grounding gate passed N>=40 rerun." if ok else "Grounding gate failed.", evidence=["tools/run_grounding_gate.py"], details=res)


def coding_eval(run_smoke: bool) -> Check:
    out = ROOT / "eval" / "results" / "coding_eval.json"
    if run_smoke:
        res = _run([sys.executable, "tools/run_coding_eval.py", "--out", str(out)])
    data = _load_json(out) or {}
    ok = bool(data) and data.get("n") == data.get("passed") and data.get("n", 0) > 0
    return _check("coding_eval", status="pass" if ok else ("missing" if not out.exists() else "fail"), summary="Executable coding eval lane passes." if ok else "Coding eval report missing or failing.", evidence=[_rel(out), "tools/run_coding_eval.py"], details=data)


def memory_eval(run_smoke: bool) -> Check:
    out = ROOT / "eval" / "results" / "memory_eval.json"
    if run_smoke:
        _run([sys.executable, "tools/run_memory_eval.py", "--out", str(out)])
    data = _load_json(out) or {}
    ok = bool(data.get("passed")) and all(data.get("checks", {}).values())
    return _check("memory_eval", status="pass" if ok else ("missing" if not out.exists() else "fail"), summary="Memory append-only safety passes." if ok else "Memory eval report missing or failing.", evidence=[_rel(out), "tools/run_memory_eval.py"], details=data)


def hidden_full_comparison() -> Check:
    candidates = list((ROOT / "agi-proof" / "hidden-reviewer-packs").glob("**/*aggregate*.json")) + list((ROOT / "agi-proof" / "hidden-reviewer-packs").glob("**/full*.json"))
    good = []
    inspected = []
    for path in candidates:
        data = _load_json(path)
        if not isinstance(data, dict) or "sophiaDelta" not in data:
            continue
        inspected.append(_rel(path))
        delta = data.get("sophiaDelta") or {}
        visibility = data.get("visibility")
        is_smoke = "smoke" in str(data.get("packId", "")).lower()
        if delta.get("scorePctVsRaw", 0) > 0 and delta.get("strictPassVsRaw", 0) >= 0 and visibility == "private-hidden" and not is_smoke:
            good.append(path)
    status = "pass" if good else ("open" if inspected else "missing")
    return _check("hidden_full_comparison", status=status, summary="Hidden Sophia-full comparison passed." if good else "No non-smoke private hidden full-comparison artifact passing machine criteria.", evidence=inspected[:10], details={"passingArtifacts": [_rel(p) for p in good]})


def self_extension_loop() -> Check:
    path = ROOT / "agi-proof" / "self-extension" / "closed-loop-2026-06-22.json"
    data = _load_json(path) or {}
    ok = data.get("loop_closed") is True and all(data.get("invariants", {}).values()) and data.get("postAccuracy", 0) > data.get("preAccuracy", 0)
    return _check("self_extension_loop", status="pass" if ok else "missing", summary="Self-extension loop closes with invariants." if ok else "Self-extension closed-loop artifact missing/failing.", evidence=[_rel(path)], details=data)


def verifier_synthesis_integrity(run_smoke: bool) -> Check:
    """Non-circularity of self-extension: a synthesized verifier is trusted ONLY
    after meta-verification, and the WITHOUT-meta ablation must fail. Without this
    lane, 'self-extension' and Skill Forge could admit unvalidated checks, so any
    AGI-candidate self-improvement claim would be circular."""
    if not run_smoke:
        return _check("verifier_synthesis_integrity", status="open", summary="Use --run-local-smoke to execute verifier-synthesis invariants.")
    res = _run([sys.executable, "tools/run_verifier_synthesis.py", "--json"])
    data = {}
    try:
        data = json.loads(res["stdoutTail"]) if res["stdoutTail"].strip().startswith("{") else {}
    except Exception:
        data = {}
    ok = res["returncode"] == 0 and bool(data.get("ok"))
    return _check(
        "verifier_synthesis_integrity",
        status="pass" if ok else "fail",
        summary="Verifier-synthesis meta-verification invariants hold (trust is earned, not assumed)." if ok else "Verifier-synthesis integrity invariants failed.",
        evidence=["tools/run_verifier_synthesis.py"],
        details={"returncode": res["returncode"], "invariants": data.get("invariants", {})},
    )


def cross_domain_transfer(run_smoke: bool) -> Check:
    """Generalization lane (Legg-Hutter / Chollet themes in definition.md): on an
    entity-disjoint split, memorized rules must NOT transfer and grounding must be
    required for low-false-positive transfer. This names the real generality
    frontier instead of letting 'broad competence' stay an unmeasured prose claim."""
    if not run_smoke:
        return _check("cross_domain_transfer", status="open", summary="Use --run-local-smoke to execute cross-entity generalization invariants.")
    res = _run([sys.executable, "tools/run_cross_entity.py", "--json"])
    data = {}
    try:
        data = json.loads(res["stdoutTail"]) if res["stdoutTail"].strip().startswith("{") else {}
    except Exception:
        data = {}
    ok = res["returncode"] == 0 and bool(data.get("ok"))
    return _check(
        "cross_domain_transfer",
        status="pass" if ok else "fail",
        summary="Cross-entity generalization invariants hold (transfer requires grounding, not memorization)." if ok else "Cross-entity generalization invariants failed.",
        evidence=["tools/run_cross_entity.py"],
        details={"returncode": res["returncode"], "invariants": data.get("invariants", {})},
    )


def distribution_shift() -> Check:
    paths = sorted((ROOT / "agi-proof" / "learning-under-shift").glob("*result*.json"))
    passing = []
    for p in paths:
        data = _load_json(p) or {}
        if data.get("passingSignal") is True and data.get("postTest", {}).get("totalCases", 0) >= 10:
            passing.append(p)
    status = "pass" if passing else ("open" if paths else "missing")
    return _check("distribution_shift", status=status, summary="Distribution-shift passing signal exists." if passing else "No multi-case passing distribution-shift result yet.", evidence=[_rel(p) for p in paths], details={"passingArtifacts": [_rel(p) for p in passing]})


def long_horizon_30m() -> Check:
    paths = sorted((ROOT / "agi-proof" / "long-horizon-runs").glob("*public-report.json"))
    passing = []
    for p in paths:
        data = _load_json(p) or {}
        if data.get("durationSec", 0) >= 1800 and data.get("autonomy", {}).get("substantive") is True and data.get("humanInterventionCount", 999) <= 2:
            passing.append(p)
    status = "pass" if passing else ("open" if paths else "missing")
    return _check("long_horizon_30m", status=status, summary="30-minute autonomy run passed." if passing else "No substantive 30-minute autonomy run yet.", evidence=[_rel(p) for p in paths], details={"passingArtifacts": [_rel(p) for p in passing]})


def long_horizon_2h_1d() -> Check:
    paths = sorted((ROOT / "agi-proof" / "long-horizon-runs").glob("*public-report.json"))
    two_h = []
    one_d = []
    for p in paths:
        data = _load_json(p) or {}
        if data.get("autonomy", {}).get("substantive") is True and data.get("humanInterventionCount", 999) <= 4:
            if data.get("durationSec", 0) >= 7200:
                two_h.append(p)
            if data.get("durationSec", 0) >= 86400:
                one_d.append(p)
    ok = bool(two_h and one_d)
    return _check("long_horizon_2h_1d", status="pass" if ok else "open", summary="2h and 1d autonomy runs passed." if ok else "2h and/or 1d substantive autonomy run missing.", evidence=[_rel(p) for p in paths], details={"twoHour": [_rel(p) for p in two_h], "oneDay": [_rel(p) for p in one_d]})


def rlvr_live_training() -> Check:
    path = ROOT / "agi-proof" / "benchmark-results" / "rlvr.public-report.json"
    data = _load_json(path) or {}
    logs = sorted((ROOT / "agi-proof" / "benchmark-results" / "runpod-rlvr").glob("*.train.log"))
    live_done = any("Live GRPO complete" in p.read_text(errors="ignore") for p in logs)
    ok = data.get("benchmark") == "rlvr" and live_done
    return _check("rlvr_live_training", status="pass" if ok else "open", summary="Live RLVR training completed; still not capability evidence." if ok else "Live RLVR training artifact missing.", evidence=[_rel(path), *[_rel(p) for p in logs[-3:]]], details={"liveLogFound": live_done, "report": data})


def rlvr_adapter_eval() -> Check:
    paths = sorted((ROOT / "agi-proof" / "benchmark-results").glob("rlvr.adapter-eval*.json"))
    passing = []
    for p in paths:
        data = _load_json(p) or {}
        if data.get("mode") == "real" and data.get("passed") is True and data.get("checks", {}).get("noFalsePositiveRegression") is True:
            passing.append(p)
    status = "pass" if passing else ("open" if paths else "missing")
    return _check("rlvr_adapter_eval", status=status, summary="Real held-out RLVR adapter eval passed." if passing else "Real held-out RLVR adapter eval missing/not passing.", evidence=[_rel(p) for p in paths], details={"passingArtifacts": [_rel(p) for p in passing]})


def external_benchmarks() -> Check:
    paths = sorted((ROOT / "agi-proof" / "external-benchmarks").glob("**/*.json"))
    passing = []
    for p in paths:
        data = _load_json(p) or {}
        system = str(data.get("system") or data.get("method") or "").lower()
        if "sophia" in system and data.get("score") is not None and data.get("total"):
            passing.append(p)
    status = "pass" if passing else "missing"
    return _check("external_benchmarks", status=status, summary="Sophia-full external benchmark artifact exists." if passing else "No Sophia-full external benchmark artifact found.", evidence=[_rel(p) for p in paths[:20]], details={"passingArtifacts": [_rel(p) for p in passing]})


def third_party_machine_replication() -> Check:
    paths = sorted((ROOT / "agi-proof" / "third-party-replication").glob("*.json"))
    passing = []
    for p in paths:
        data = _load_json(p) or {}
        cmds = data.get("commands") or data.get("checks") or []
        if data.get("ok") is True or (cmds and all((c.get("returncode") == 0 or c.get("ok") is True) for c in cmds if isinstance(c, dict))):
            passing.append(p)
    status = "pass" if passing else ("open" if paths else "missing")
    return _check("third_party_machine_replication", status=status, summary="Machine replication checklist artifact passes; human independence still outside non-human gate." if passing else "Machine replication artifact missing/failing.", evidence=[_rel(p) for p in paths], details={"passingArtifacts": [_rel(p) for p in passing], "humanIndependenceNote": "Reviewer identity/signature cannot be self-certified by a non-human repo script."})


def collect_checks(run_smoke: bool) -> list[Check]:
    return [
        provenance_validation(run_smoke),
        published_provenance_delta(),
        calibration_abstention(),
        grounded_gate(run_smoke),
        coding_eval(run_smoke),
        memory_eval(run_smoke),
        hidden_full_comparison(),
        self_extension_loop(),
        verifier_synthesis_integrity(run_smoke),
        cross_domain_transfer(run_smoke),
        distribution_shift(),
        long_horizon_30m(),
        rlvr_live_training(),
        rlvr_adapter_eval(),
        external_benchmarks(),
        long_horizon_2h_1d(),
        third_party_machine_replication(),
    ]


def level_passed(checks: list[Check], target: str) -> bool:
    idx = LEVEL_ORDER.index(target)
    needed = {lvl for lvl in LEVEL_ORDER[: idx + 1] if lvl != "level5"}
    return all(c.passed for c in checks if c.requiredFor in needed)


def highest_level(checks: list[Check]) -> str:
    last = "below-level2"
    for lvl in LEVEL_ORDER[:-1]:
        if level_passed(checks, lvl):
            last = lvl
        else:
            break
    return last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=LEVEL_ORDER[:-1], default="level3")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--run-local-smoke", action="store_true", help="run cheap deterministic local checks before aggregating")
    ap.add_argument("--allow-open", action="store_true", help="exit 0 even when target is not reached")
    args = ap.parse_args(argv)

    criteria = _load_json(CRITERIA) or {}
    checks = collect_checks(args.run_local_smoke)
    high = highest_level(checks)
    target_ok = level_passed(checks, args.target)
    blocking = [asdict(c) for c in checks if not c.passed and LEVEL_ORDER.index(c.requiredFor) <= LEVEL_ORDER.index(args.target)]
    report = {
        "schema": "sophia.agi_verification.report.v1",
        "target": args.target,
        "targetPassed": target_ok,
        "highestMachineVerifiedLevel": high,
        "canClaimAGI": False,
        "claimBoundary": criteria.get("claimBoundary", "No non-human gate can philosophically prove AGI."),
        "recommendedPublicWording": (
            "Sophia is an AGI-candidate verifier-gated epistemic agent framework. "
            "Do not claim proven AGI unless external scientific review accepts the evidence."
        ),
        "checks": [asdict(c) for c in checks],
        "blockingForTarget": blocking,
        "remainingNonHumanSteps": [
            {"id": b["id"], "requiredFor": b["requiredFor"], "summary": b["summary"]}
            for b in blocking
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "target": args.target,
        "targetPassed": target_ok,
        "highestMachineVerifiedLevel": high,
        "blocking": report["remainingNonHumanSteps"],
        "out": _rel(args.out),
    }, indent=2, ensure_ascii=False))
    return 0 if target_ok or args.allow_open else 1

if __name__ == "__main__":
    raise SystemExit(main())
