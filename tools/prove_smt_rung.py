#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prove (or NO-GO) the H3 certificate-carrying SMT rung — the abstention-reclaim gate.

Pre-registration: ``agi-proof/smt-rung/measurement_spec.json`` (frozen set
``smt-decidable-abstained-v1``, primaryMetric = abstention-reclaim rate, mde 0.10,
requiredN 200). Without z3 the rung abstains on every claim (fail-closed); WITH z3 it
should DECIDE each decidable-but-abstained claim correctly and carry a re-checkable
certificate. This harness measures the reclaim.

Independence contract (why ``agreement`` is meaningful): the frozen set's ground-truth
labels are computed HERE from first principles — dimensional-vector equality for
``unit_consistency``, exhaustive enumeration of the bounded domain for ``bounded_int``,
and exact rational containment for ``interval`` — NEVER by asking ``agent.smt_verifier``.
So ``agreement = 1.00`` means the solver agrees with an independent oracle, not itself.

GO iff (correctnessRule) label-agreement == 1.00 on every decided claim AND
(certificateRule) independent re-checker acceptance == 1.00 on every certificate AND
(magnitudeRule) reclaim-rate lower 95% CI >= 0.10 — and the out-of-band guardrail set
still abstains (no force-fit). Deterministic (seed 0); set + labels carry no tuning.

Usage:
  python tools/prove_smt_rung.py            # generate frozen set, run, write receipt
  python tools/prove_smt_rung.py --n 240    # frozen-set size (default 240, >= requiredN)
Requires z3 (pip install -r requirements-smt.txt); abstains-only report if z3 absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from agent import smt_verifier as smt  # noqa: E402
import eval_stats  # noqa: E402

OUT_DIR = ROOT / "agi-proof" / "smt-rung"
FROZEN = OUT_DIR / "frozen_set_v1.jsonl"
GUARDRAIL = OUT_DIR / "guardrail_oob.jsonl"
RESULT = OUT_DIR / "smt-rung.result.json"
SI = ("m", "kg", "s", "A", "K", "mol", "cd")


# --------------------------------------------------------------------------- #
# Independent ground truth (NOT via smt_verifier)
# --------------------------------------------------------------------------- #
def _vec(units: dict) -> tuple:
    return tuple(int(units.get(b, 0)) for b in SI)


def _unit_label(claim: dict) -> str:
    return "pass" if _vec(claim["lhs"]) == _vec(claim["rhs"]) else "fail"


def _bounded_sat(claim: dict) -> bool:
    """Exhaustively decide satisfiability over the bounded integer domain."""
    names = list(claim["vars"])
    ranges = [range(claim["vars"][n][0], claim["vars"][n][1] + 1) for n in names]
    ops = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
           ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
           "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
    for combo in product(*ranges):
        env = dict(zip(names, combo))
        if all(ops[op](sum(c * env[v] for v, c in coeffs.items()), rhs)
               for coeffs, op, rhs in claim["constraints"]):
            return True
    return False


def _bounded_label(claim: dict) -> str:
    true_sat = _bounded_sat(claim)
    return "pass" if (claim["expected"] == ("sat" if true_sat else "unsat")) else "fail"


def _interval_label(claim: dict) -> str:
    v, lo, hi = (Fraction(str(claim[k])) for k in ("value", "lo", "hi"))
    return "pass" if (bool(claim["expected"]) == (lo <= v <= hi)) else "fail"


# --------------------------------------------------------------------------- #
# Deterministic frozen-set generation
# --------------------------------------------------------------------------- #
def _gen(n: int) -> list[dict]:
    rng = random.Random(0)
    n_unit = n_bi = n // 3 + (1 if n % 3 else 0)
    n_iv = n - n_unit - n_bi
    rows: list[dict] = []

    for i in range(n_unit):  # half consistent (pass), half a one-base perturbation (fail)
        base = {b: rng.randint(-3, 3) for b in rng.sample(SI, rng.randint(1, 4))}
        lhs = {b: e for b, e in base.items() if e}
        rhs = dict(lhs)
        if i % 2:  # make it inconsistent
            b = rng.choice(SI)
            rhs[b] = rhs.get(b, 0) + rng.choice([-2, -1, 1, 2])
            rhs = {k: v for k, v in rhs.items() if v}
        c = {"kind": "unit_consistency", "lhs": lhs or {"m": 0}, "rhs": rhs or {"m": 0}}
        rows.append({"claim": c, "label": _unit_label(c)})

    for i in range(n_bi):  # half with the correct expected (pass), half wrong (fail)
        names = ["x", "y"] if i % 2 == 0 else ["x", "y", "z"]
        vars = {nm: [0, rng.randint(3, 5)] for nm in names}
        k = len(names)
        cons = []
        for _ in range(rng.randint(1, 3)):
            coeffs = {nm: rng.randint(-2, 3) for nm in rng.sample(names, rng.randint(1, k))}
            coeffs = {nm: cc for nm, cc in coeffs.items() if cc} or {names[0]: 1}
            op = rng.choice(["<=", "<", ">=", ">", "==", "!="])
            rhs = rng.randint(-3, 10)
            cons.append([coeffs, op, rhs])
        probe = {"kind": "bounded_int", "vars": vars, "constraints": cons, "expected": "sat"}
        true_sat = _bounded_sat(probe)
        correct = "sat" if true_sat else "unsat"
        wrong = "unsat" if true_sat else "sat"
        probe["expected"] = correct if i % 2 == 0 else wrong
        rows.append({"claim": probe, "label": _bounded_label(probe)})

    for i in range(n_iv):  # half correct expected (pass), half flipped (fail)
        lo = rng.randint(-10, 5)
        hi = lo + rng.randint(0, 12)
        val = rng.randint(lo - 4, hi + 4)
        true_in = lo <= val <= hi
        exp = true_in if i % 2 == 0 else (not true_in)
        c = {"kind": "interval", "value": val, "lo": lo, "hi": hi, "expected": bool(exp)}
        rows.append({"claim": c, "label": _interval_label(c)})

    rng.shuffle(rows)
    return rows


def _guardrail() -> list[dict]:
    """Out-of-band claims that MUST still abstain even with z3 (no force-fit)."""
    return [
        {"kind": "free_text", "text": "Newton discovered gravity"},
        {"kind": "unit_consistency", "lhs": {"parsec": 1}, "rhs": {"m": 1}},  # bad base
        {"kind": "bounded_int", "vars": {"x": [0, 3]},
         "constraints": [[{"x": 1.5}, "<=", 2]], "expected": "sat"},  # non-int coeff
        {"kind": "interval", "value": "not-a-number", "lo": 0, "hi": 1, "expected": True},
        {"kind": "modal_logic", "formula": "box(p -> q)"},
    ]


def _canon_hash(rows: list[dict]) -> str:
    blob = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Prove/NO-GO the H3 SMT abstention-reclaim gate.")
    ap.add_argument("--n", type=int, default=240, help="frozen-set size (>= requiredN 200)")
    args = ap.parse_args()

    rows = _gen(args.n)
    guard = _guardrail()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    GUARDRAIL.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in guard) + "\n", encoding="utf-8")
    frozen_hash = _canon_hash(rows)

    z3_present = smt.z3_available()
    z3_version = None
    if z3_present:
        import z3
        z3_version = z3.get_version_string()

    reclaim_flags: list[float] = []   # 1 if decided (not abstain), else 0
    decided = agreed = accepted = 0
    mislabels: list[dict] = []
    cert_failures: list[dict] = []
    for r in rows:
        res = smt.check(r["claim"])
        v = res["verdict"]
        is_decided = v in ("pass", "fail")
        reclaim_flags.append(1.0 if is_decided else 0.0)
        if is_decided:
            decided += 1
            if v == r["label"]:
                agreed += 1
            else:
                mislabels.append({"claim": r["claim"], "expected": r["label"], "got": v})
            if smt.recheck_certificate(res.get("certificate")):
                accepted += 1
            else:
                cert_failures.append({"claim": r["claim"], "verdict": v})

    # out-of-band guardrail: every one MUST abstain (no force-fit)
    guard_abstained = sum(1 for c in guard if smt.check(c)["verdict"] == "abstain")

    n = len(rows)
    reclaim_rate = decided / n if n else 0.0
    ci = eval_stats.fixed_n_ci_mean(reclaim_flags) if reclaim_flags else [0.0, 0.0]
    agreement = agreed / decided if decided else 0.0
    acceptance = accepted / decided if decided else 0.0

    go = bool(
        z3_present
        and decided == n
        and agreement == 1.0
        and acceptance == 1.0
        and ci[0] >= 0.10
        and guard_abstained == len(guard)
    )
    verdict = "GO" if go else ("NO-GO" if z3_present else "ABSTAIN-ONLY (z3 absent)")

    receipt = {
        "experimentId": "smt-rung-abstention-reclaim",
        "verdict": verdict,
        "canClaimAGI": False,
        "claimCeiling": "candidate_only; a decidable-band reclaim result, not a capability claim",
        "z3Present": z3_present,
        "z3Version": z3_version,
        "interpreter": sys.version.split()[0],
        "frozenSet": {"id": "smt-decidable-abstained-v1", "path": str(FROZEN.relative_to(ROOT)),
                      "n": n, "sha256": frozen_hash, "seed": 0, "deterministic": True},
        "metrics": {
            "reclaimRate": round(reclaim_rate, 6),
            "reclaimRateCI95": [round(ci[0], 6), round(ci[1], 6)],
            "decided": decided, "abstained": n - decided,
            "labelAgreement": round(agreement, 6),
            "certificateAcceptance": round(acceptance, 6),
            "oobGuardrailAbstained": f"{guard_abstained}/{len(guard)}",
        },
        "gate": {
            "reclaimLowerCI>=0.10": ci[0] >= 0.10,
            "labelAgreement==1.00": agreement == 1.0,
            "certificateAcceptance==1.00": acceptance == 1.0,
            "oobAllAbstain": guard_abstained == len(guard),
        },
        "mislabels": mislabels[:10],
        "certFailures": cert_failures[:10],
        "honestBound": (
            "Measured locally with z3 present. Labels are first-principles and independent of "
            "agent.smt_verifier (dimensional equality / exhaustive bounded enumeration / exact "
            "rational containment). Frozen set is deterministic (seed 0), no tuning. To reproduce "
            "in CI, install requirements-smt.txt. canClaimAGI stays false; this reclaims a narrow "
            "decidable band (unit/dimension, bounded-int, interval), NOT general reasoning."
        ),
    }
    RESULT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"SMT RUNG [{verdict}]  reclaim={reclaim_rate:.3f} CI{ci}  "
          f"agreement={agreement:.3f}  certAccept={acceptance:.3f}  "
          f"oobAbstain={guard_abstained}/{len(guard)}  N={n}  z3={z3_version}", file=sys.stderr)
    print(json.dumps({"verdict": verdict, "metrics": receipt["metrics"]}, ensure_ascii=False))
    return 0 if go else (2 if not z3_present else 3)


if __name__ == "__main__":
    raise SystemExit(main())
