#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verification-of-verification: a monitor over the DEPLOYED verifiers.

The verifiers in :mod:`agent.verifiers` are the trust root of the whole
verifier-gated loop — "quality follows verifiability". But a verifier can rot:
the world drifts, an oracle split reveals it now passes answers it should reject.
A gate you never re-verify is a gate you have stopped trusting on faith.

This monitor takes each deployed verifier's MEASURED precision/recall time-series
on FRESH oracle splits (from a provided JSON) and:

  * AUTO-DEMOTES any verifier whose latest measured precision has fallen below its
    PRE-REGISTERED floor (from ``agi-proof/verify-verifiers/drift_floors.json``),
    provided the drop is RESOLVABLE at the split's N. The floor is committed
    *before* any live re-scoring run, so a demotion is a mechanical rule, never a
    post-hoc judgment call. An underpowered breach is HELD (abstain), not demoted —
    a gate that fires on noise is worse than no gate. A POWERED breach ALWAYS
    demotes so the gate cannot be starved into never firing.

  * HALTS fail-closed if the provided WITH/WITHOUT-meta ablation gap has collapsed
    below its floor. Meta-verification is the trust root: if we can no longer show
    that verifying-the-verifiers adds precision, NO downstream verdict is
    trustworthy, so we halt and demote nothing silently — a human must
    re-establish the ablation.

Exit codes (house style): 0 all-trusted / 1 demotion-or-halt / 2 unreadable.
Prints a JSON receipt to stdout; human prose to stderr.

Scope (honest / no-overclaim): the monitor + the seed-degraded self-test
(:mod:`tools.vov_selftest`) are REAL and unit-tested. The LIVE standing
re-scoring on fresh oracle splits needs oracle labels that do NOT exist in-repo
(N=0); that arm is PRE-REGISTERED in
``agi-proof/verify-verifiers/measurement_spec.json`` (status
``preregistration_only``, go:false, canClaimAGI:false) and is NOT-PROVEN.
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
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from tools import eval_stats  # noqa: E402

DEFAULT_FLOORS = ROOT / "agi-proof" / "verify-verifiers" / "drift_floors.json"

EXIT_TRUSTED = 0
EXIT_DEMOTE_OR_HALT = 1
EXIT_UNREADABLE = 2


class Unreadable(Exception):
    """Raised when an input is missing or malformed — maps to exit 2 (fail-closed)."""


def _load_json(path: Path, *, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Unreadable(f"{what} not found: {path}") from exc
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise Unreadable(f"{what} unreadable ({path}): {exc}") from exc


def _floor_for(name: str, floors: dict) -> dict:
    """Resolve the pre-registered floor spec for a verifier, applying per-verifier
    overrides over the committed defaults."""
    defaults = dict(floors.get("defaults") or {})
    override = (floors.get("perVerifierOverrides") or {}).get(name) or {}
    merged = dict(defaults)
    for k in ("precisionFloor", "recallFloor", "minN", "driftTolerance", "requirePowered"):
        if k in override:
            merged[k] = override[k]
    return merged


def _latest(series: Any) -> "dict | None":
    """Pull the most recent measurement point from a verifier's time-series.

    Accepts either a list of ``{precision, recall, n, ...}`` points (last = latest)
    or a single dict. Returns None if there is no usable point (fail-closed:
    a verifier with no fresh measurement cannot be vouched for)."""
    if isinstance(series, list):
        pts = [p for p in series if isinstance(p, dict)]
        return pts[-1] if pts else None
    if isinstance(series, dict):
        return series
    return None


def _powered_breach(precision: float, floor: float, n: int) -> "tuple[bool, float]":
    """Is a precision-below-floor breach RESOLVABLE at this N?

    Returns (powered, mde). The breach magnitude is |floor - precision|; the probe
    at N can resolve it iff eval_stats.mde_at_n(N) <= that magnitude. p0 is the
    floor (the null we test the observed precision against)."""
    drop = abs(floor - precision)
    p0 = min(max(floor, 1e-6), 1 - 1e-6)
    mde = eval_stats.mde_at_n(max(1, int(n)), p0=p0)
    return (mde <= drop + 1e-9, round(mde, 4))


def evaluate(report: dict, floors: dict) -> dict:
    """Core monitor logic (pure, no I/O). Returns a receipt dict.

    ``report`` shape::

        {
          "cycle": <int|str>,
          "ablation": {"withMetaPrecision": .., "withoutMetaPrecision": ..},   # optional but recommended
          "verifiers": {
             "<verifier_name>": [ {"precision":.., "recall":.., "n":..}, ... ]  # time-series, last=latest
             # or a single {"precision":.., "recall":.., "n":..}
          }
        }
    """
    defaults = dict(floors.get("defaults") or {})
    ab_spec = dict(floors.get("ablationGap") or {})
    ab_floor = float(ab_spec.get("floor", 0.05))
    halt_on_collapse = bool(ab_spec.get("haltOnCollapse", True))

    checks: list[dict] = []
    demoted: list[str] = []
    held: list[str] = []

    # --- meta-verify ablation (the trust root) checked FIRST: a collapse halts everything ---
    halt = False
    ablation = report.get("ablation")
    ab_result: dict
    if isinstance(ablation, dict) and (
        "withMetaPrecision" in ablation and "withoutMetaPrecision" in ablation
    ):
        try:
            gap = float(ablation["withMetaPrecision"]) - float(ablation["withoutMetaPrecision"])
        except (TypeError, ValueError):
            gap = None
        if gap is None:
            ab_result = {"status": "unreadable", "gap": None, "floor": ab_floor,
                         "note": "ablation values not numeric — fail-closed HALT"}
            halt = True
        elif gap < ab_floor:
            ab_result = {"status": "COLLAPSED", "gap": round(gap, 4), "floor": ab_floor,
                         "note": "with/without-meta precision gap below floor — meta-verify no "
                                 "longer demonstrably adds signal; HALT fail-closed"}
            halt = halt_on_collapse
        else:
            ab_result = {"status": "healthy", "gap": round(gap, 4), "floor": ab_floor}
    else:
        # No ablation provided: cannot demonstrate the trust root. Fail-closed HALT
        # (the safest reading — never assume meta-verify is fine when unmeasured).
        ab_result = {"status": "missing", "gap": None, "floor": ab_floor,
                     "note": "no with/without-meta ablation provided — trust root undemonstrated; "
                             "HALT fail-closed"}
        halt = True

    if halt:
        return {
            "tool": "verify_verifiers",
            "cycle": report.get("cycle"),
            "trusted": False,
            "halt": True,
            "reason": "meta-verify ablation gap collapsed/absent — trust root undemonstrated",
            "ablation": ab_result,
            "demoted": [],
            "held": [],
            "checks": [],
            "canClaimAGI": False,
            "note": "HALT fail-closed: no downstream verifier verdict is trustworthy without "
                    "a demonstrated meta-verify gap. Demoted NOTHING silently.",
        }

    # --- per-verifier precision-below-floor -> auto-demote (powered breaches only) ---
    verifiers = report.get("verifiers")
    if not isinstance(verifiers, dict) or not verifiers:
        # A cycle with no verifier measurements cannot vouch for anything.
        return {
            "tool": "verify_verifiers",
            "cycle": report.get("cycle"),
            "trusted": False,
            "halt": False,
            "reason": "no per-verifier measurements provided",
            "ablation": ab_result,
            "demoted": [],
            "held": [],
            "checks": [],
            "canClaimAGI": False,
        }

    for name in sorted(verifiers):
        spec = _floor_for(name, floors)
        p_floor = float(spec.get("precisionFloor", 0.90))
        r_floor = float(spec.get("recallFloor", 0.70))
        min_n = int(spec.get("minN", defaults.get("minN", 30)))
        tol = float(spec.get("driftTolerance", defaults.get("driftTolerance", 0.05)))
        require_powered = bool(spec.get("requirePowered", defaults.get("requirePowered", True)))

        point = _latest(verifiers[name])
        if point is None or "precision" not in point:
            # Fail-closed: a verifier with no fresh measurement is HELD (not silently trusted).
            held.append(name)
            checks.append({"verifier": name, "status": "held", "reason": "no fresh measurement",
                           "precisionFloor": p_floor})
            continue

        precision = float(point.get("precision"))
        recall = float(point.get("recall")) if point.get("recall") is not None else None
        n = int(point.get("n", 0))

        below_p = precision < p_floor
        below_r = recall is not None and recall < r_floor
        drift = round(p_floor - precision, 4)
        exceeds_tol = drift > tol

        if not (below_p or below_r):
            checks.append({"verifier": name, "status": "trusted", "precision": round(precision, 4),
                           "recall": (round(recall, 4) if recall is not None else None),
                           "precisionFloor": p_floor, "recallFloor": r_floor, "n": n})
            continue

        # A breach exists. Is it POWERED at this N? (Only demote resolvable breaches.)
        # Use the tighter (more-breached) of the two metrics to decide power.
        breach_precision = precision if below_p else (recall if recall is not None else precision)
        breach_floor = p_floor if below_p else r_floor
        powered, mde = _powered_breach(breach_precision, breach_floor, n)
        underpowered_split = n < min_n or not powered

        if require_powered and underpowered_split:
            held.append(name)
            checks.append({
                "verifier": name, "status": "held (underpowered)",
                "precision": round(precision, 4),
                "recall": (round(recall, 4) if recall is not None else None),
                "precisionFloor": p_floor, "recallFloor": r_floor, "n": n, "minN": min_n,
                "mde": mde, "drift": drift, "exceedsDriftTolerance": exceeds_tol,
                "reason": "floor breach but split cannot resolve a drop this size (N<minN or "
                          "mde>drop) — HELD, not demoted (gate must not fire on noise)",
            })
            continue

        # Powered breach -> AUTO-DEMOTE.
        demoted.append(name)
        reasons = []
        if below_p:
            reasons.append(f"precision {precision:.4f} < floor {p_floor:.2f}")
        if below_r:
            reasons.append(f"recall {recall:.4f} < floor {r_floor:.2f}")
        checks.append({
            "verifier": name, "status": "DEMOTED",
            "precision": round(precision, 4),
            "recall": (round(recall, 4) if recall is not None else None),
            "precisionFloor": p_floor, "recallFloor": r_floor, "n": n,
            "mde": mde, "drift": drift, "exceedsDriftTolerance": exceeds_tol,
            "reasons": reasons,
            "action": "demote to advisory (verdict no longer gates; human re-cert required)",
        })

    trusted = not demoted  # held items don't break trust, but they are surfaced
    return {
        "tool": "verify_verifiers",
        "cycle": report.get("cycle"),
        "trusted": trusted,
        "halt": False,
        "ablation": ab_result,
        "demoted": demoted,
        "held": held,
        "checks": checks,
        "canClaimAGI": False,
        "note": ("all deployed verifiers within pre-registered floors"
                 if trusted else
                 f"auto-demoted {len(demoted)} verifier(s) below pre-registered precision floor"),
    }


def run(report_path: Path, floors_path: Path = DEFAULT_FLOORS) -> "tuple[dict, int]":
    """Load inputs, evaluate, and return (receipt, exit_code). Fail-closed on unreadable."""
    floors = _load_json(floors_path, what="drift_floors")
    if not isinstance(floors, dict):
        raise Unreadable(f"drift_floors is not a JSON object: {floors_path}")
    report = _load_json(report_path, what="oracle-split report")
    if not isinstance(report, dict):
        raise Unreadable(f"report is not a JSON object: {report_path}")
    receipt = evaluate(report, floors)
    receipt["floorsSource"] = str(floors_path)
    receipt["reportSource"] = str(report_path)
    if receipt.get("halt") or receipt.get("demoted"):
        return receipt, EXIT_DEMOTE_OR_HALT
    if not receipt.get("trusted"):
        return receipt, EXIT_DEMOTE_OR_HALT
    return receipt, EXIT_TRUSTED


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verification-of-verification monitor: auto-demote drifted verifiers, "
                    "HALT fail-closed if the meta-verify ablation gap collapses.")
    ap.add_argument("report", help="JSON with each deployed verifier's measured "
                                   "precision/recall time-series on fresh oracle splits, plus "
                                   "an optional with/without-meta ablation.")
    ap.add_argument("--floors", default=str(DEFAULT_FLOORS),
                    help="pre-registered drift floors JSON (default: "
                         "agi-proof/verify-verifiers/drift_floors.json)")
    args = ap.parse_args(argv)

    try:
        receipt, code = run(Path(args.report), Path(args.floors))
    except Unreadable as exc:
        print(json.dumps({"tool": "verify_verifiers", "trusted": False,
                          "error": str(exc), "canClaimAGI": False}), flush=True)
        print(f"UNREADABLE: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE

    print(json.dumps(receipt, indent=2), flush=True)
    if receipt.get("halt"):
        print("HALT (exit 1): meta-verify trust root undemonstrated — no verdict trustworthy.",
              file=sys.stderr)
    elif receipt.get("demoted"):
        print(f"DEMOTED (exit 1): {', '.join(receipt['demoted'])} below pre-registered floor.",
              file=sys.stderr)
    else:
        print("ALL VERIFIERS TRUSTED (exit 0): within pre-registered floors.", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
