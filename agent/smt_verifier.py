# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Certificate-carrying SMT rung — an OPTIONAL-BACKEND checker for a narrow,
DECIDABLE band (unit/dimension consistency and bounded-integer / interval
arithmetic), fail-closed when z3 is absent.

Where does this sit on the verifier ladder? ``agent/verifiers.py`` already ships
deterministic pure-Python checks (``arithmetic_sound``, ``physics_sound``) and an
optional-backend sympy tier (``math_equivalent``/``math_sound``), each of which
**abstains / fails closed** when its backend is missing rather than silently
passing (see docs/11-Platform/Verifier-Synthesis.md and the ``sympy_unavailable``
held verdict). This module adds the **solver** tier for a small decidable
fragment, following the exact same fail-closed contract as
``agent/formal_verifier.py`` (``require_z3`` -> ``held``/``z3_unavailable``).

What makes this a "certificate-carrying" rung: when z3 IS present and decides a
claim, :func:`check` returns a **re-checkable certificate** — the satisfying
model (for a SAT/consistent claim) or the unsat core (for an UNSAT/inconsistent
one), plus the reconstructed constraints. A SEPARATE dumb checker
(:func:`recheck_certificate`) replays that certificate WITHOUT re-invoking the
solver's decision procedure and must accept it. The value of the rung is that the
gate's trust does not rest on trusting the solver's yes/no — it rests on a
certificate an independent, simpler program can confirm.

Honest scope / repo law (NO-OVERCLAIM):
  - z3 is an OPTIONAL dependency (see requirements-smt.txt). When it is absent —
    which is the case in this environment — EVERY call to :func:`check` returns
    verdict ``"abstain"`` with reason ``"z3-not-installed"``. It NEVER returns a
    false ``"pass"``. A gate that cannot run is a HELD verdict, not a pass.
  - The abstention-reclaim RATE (how many currently-abstained-but-decidable
    claims this rung reclaims at independent-checker acceptance EXACTLY 1.00) is
    PRE-REGISTERED in agi-proof/smt-rung/measurement_spec.json and is NOT proven
    here — z3 is not installed, so no live reclaim number has been measured.
  - The decidable band is deliberately narrow: linear unit/dimension consistency
    and bounded-integer / interval constraints. That is a genuinely decidable
    fragment (linear integer/real arithmetic); claims outside it ABSTAIN, they are
    not force-fit.

Result shape (``check``):
    {
      "verdict": "pass" | "fail" | "abstain",
      "certificate": {...} | None,   # re-checkable; None on abstain
      "reason": "<machine-readable reason>",
      "backend": "z3" | "none",
    }
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

# The certificate format is versioned so a re-checker can refuse an unknown
# schema (fail-closed) rather than mis-replay it.
CERT_VERSION = "smt-cert-1"

# Verdict constants (so callers never typo a magic string).
PASS = "pass"
FAIL = "fail"
ABSTAIN = "abstain"

# The two claim kinds in the decidable band.
KIND_UNIT = "unit_consistency"          # SI base-dimension balance across an equality
KIND_BOUNDED_INT = "bounded_int"        # linear integer constraints over bounded vars
KIND_INTERVAL = "interval"              # rational interval-arithmetic containment

_SUPPORTED_KINDS = (KIND_UNIT, KIND_BOUNDED_INT, KIND_INTERVAL)

# SI base dimensions used for unit-consistency claims. A "dimension vector" maps
# each base to an integer exponent; two sides are consistent iff their vectors are
# equal. This is decidable (integer-vector equality) with or without a solver, but
# with z3 we get a re-checkable model over the exponents.
_SI_BASES = ("m", "kg", "s", "A", "K", "mol", "cd")


# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #
def z3_available() -> bool:
    """True iff the optional z3 backend can be imported. Mirrors
    :func:`agent.formal_verifier.z3_available`."""
    try:
        import z3  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - z3 is an optional dependency; any import error => absent
        return False


# --------------------------------------------------------------------------- #
# Result helpers
# --------------------------------------------------------------------------- #
def _abstain(reason: str, *, backend: str = "none") -> dict:
    """A HELD verdict: the rung could not decide, so it vouches for nothing."""
    return {"verdict": ABSTAIN, "certificate": None, "reason": reason, "backend": backend}


def _z3_absent() -> dict:
    """The canonical fail-closed result for the z3-absent environment (this env)."""
    return _abstain("z3-not-installed", backend="none")


# --------------------------------------------------------------------------- #
# Claim validation (shared by both backends and the re-checker)
# --------------------------------------------------------------------------- #
def _dim_vector(units: "dict[str, Any] | None") -> "dict[str, int] | None":
    """Normalise a ``{base: exponent}`` unit map to a full SI exponent vector.

    Returns None (=> abstain) if a key is not an SI base or an exponent is not an
    integer — the fragment only covers integer-exponent dimensional analysis.
    """
    if not isinstance(units, dict):
        return None
    vec = {b: 0 for b in _SI_BASES}
    for base, exp in units.items():
        if base not in vec:
            return None
        if not isinstance(exp, int) or isinstance(exp, bool):
            return None
        vec[base] = exp
    return vec


def _as_fraction(x: Any) -> "Fraction | None":
    """Exact rational parse (int / float / 'a/b' / numeric str). None if not numeric."""
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, Fraction):
            return x
        if isinstance(x, int):
            return Fraction(x)
        if isinstance(x, float):
            return Fraction(x).limit_denominator(10**12)
        if isinstance(x, str):
            return Fraction(x)
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return None


def classify(claim: "dict | None") -> "str | None":
    """Return the decidable-band kind of ``claim``, or None if out of band.

    A claim is a dict with a ``kind`` in :data:`_SUPPORTED_KINDS`. Anything else —
    a free-text claim, an unknown kind, a non-dict — is out of the decidable band
    and must ABSTAIN, never be force-fit into a solver call.
    """
    if not isinstance(claim, dict):
        return None
    kind = claim.get("kind")
    return kind if kind in _SUPPORTED_KINDS else None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def check(claim: "dict | None") -> dict:
    """Check a claim in the narrow decidable band, carrying a re-checkable cert.

    Contract:
      * z3 ABSENT (this environment): ALWAYS returns verdict ``"abstain"`` with
        reason ``"z3-not-installed"`` — fail-closed, never a false ``"pass"``.
      * claim OUT OF BAND (unknown/missing kind, malformed): ``"abstain"`` with a
        reason describing why (still never a false pass).
      * z3 PRESENT and claim in band: actually decides it and returns ``"pass"``
        (with a satisfying-model certificate) or ``"fail"`` (with an unsat-core
        certificate). The certificate is re-checkable by :func:`recheck_certificate`.

    ``claim`` schema by kind:
      unit_consistency: {"kind","lhs":{base:exp,...},"rhs":{base:exp,...}}
      bounded_int:      {"kind","vars":{name:[lo,hi],...},
                         "constraints":[[lhs_coeffs,op,rhs_int],...],
                         "expected": "sat"|"unsat"}
                        where lhs_coeffs is {name:int_coeff,...} and op in
                        {"<=","<",">=",">","==","!="}. Verdict "pass" iff the
                        solver's SAT/UNSAT matches "expected".
      interval:         {"kind","value":num,"lo":num,"hi":num,
                         "expected": true|false}  (is value in [lo,hi]?)
    """
    kind = classify(claim)
    if kind is None:
        # Distinguish "no backend" from "out of band" only after the backend
        # check below is NOT what we want: an out-of-band claim abstains
        # regardless of backend. But we still surface z3-not-installed first when
        # the claim itself is well-formed-but-undecidable-here is not the case:
        # out-of-band is a property of the claim, so report it directly.
        return _abstain("claim-not-in-decidable-band")

    if not z3_available():
        # Fail-closed: the ONLY honest verdict without the solver is abstain.
        return _z3_absent()

    try:
        if kind == KIND_UNIT:
            return _z3_check_unit(claim)  # type: ignore[arg-type]
        if kind == KIND_BOUNDED_INT:
            return _z3_check_bounded_int(claim)  # type: ignore[arg-type]
        if kind == KIND_INTERVAL:
            return _z3_check_interval(claim)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - a solver/marshalling error is a HELD verdict, not a pass
        return _abstain(f"solver-error:{type(exc).__name__}", backend="z3")
    return _abstain("claim-not-in-decidable-band")  # unreachable, defensive


def recheck_certificate(cert: "dict | None") -> bool:
    """Independently RE-VERIFY a certificate WITHOUT invoking the solver.

    This is the "separate dumb checker": it does not ask z3 anything — it replays
    the certificate's own witness against the reconstructed constraints using plain
    Python / exact rationals. For a re-checkable rung this must accept a valid
    certificate at rate EXACTLY 1.00 (that is the pre-registered acceptance bar).

    Returns True iff the certificate is well-formed AND its witness genuinely
    demonstrates the recorded verdict; False otherwise (fail-closed: an unknown
    schema, a missing witness, or a witness that does not check => False).
    """
    if not isinstance(cert, dict):
        return False
    if cert.get("version") != CERT_VERSION:
        return False
    kind = cert.get("kind")
    verdict = cert.get("verdict")
    try:
        if kind == KIND_UNIT:
            return _recheck_unit(cert, verdict)
        if kind == KIND_BOUNDED_INT:
            return _recheck_bounded_int(cert, verdict)
        if kind == KIND_INTERVAL:
            return _recheck_interval(cert, verdict)
    except Exception:  # noqa: BLE001 - any replay error means the cert does not check
        return False
    return False


# --------------------------------------------------------------------------- #
# z3 backend — unit / dimension consistency
# --------------------------------------------------------------------------- #
def _z3_check_unit(claim: dict) -> dict:
    """Decide whether both sides of an equality share the same SI dimension.

    Encodes each side's exponent vector as z3 integer constants and asserts
    per-base equality. SAT (consistent) => "pass" with the exponent model as the
    certificate; UNSAT (inconsistent) => "fail" with the mismatched bases as the
    unsat-core-style certificate.
    """
    import z3

    lhs = _dim_vector(claim.get("lhs"))
    rhs = _dim_vector(claim.get("rhs"))
    if lhs is None or rhs is None:
        return _abstain("unit-claim-out-of-band", backend="z3")

    s = z3.Solver()
    lvars = {b: z3.Int(f"L_{b}") for b in _SI_BASES}
    rvars = {b: z3.Int(f"R_{b}") for b in _SI_BASES}
    for b in _SI_BASES:
        s.add(lvars[b] == lhs[b])
        s.add(rvars[b] == rhs[b])
    # The consistency query: are all base exponents equal across the sides?
    s.add(z3.And([lvars[b] == rvars[b] for b in _SI_BASES]))

    if s.check() == z3.sat:
        m = s.model()
        model = {b: m[lvars[b]].as_long() for b in _SI_BASES}
        cert = {
            "version": CERT_VERSION,
            "kind": KIND_UNIT,
            "verdict": PASS,
            "lhs": lhs,
            "rhs": rhs,
            "model": model,  # the witnessed (equal) exponent vector
        }
        return {"verdict": PASS, "certificate": cert,
                "reason": "z3:unit-dimensions-consistent", "backend": "z3"}
    mismatched = [b for b in _SI_BASES if lhs[b] != rhs[b]]
    cert = {
        "version": CERT_VERSION,
        "kind": KIND_UNIT,
        "verdict": FAIL,
        "lhs": lhs,
        "rhs": rhs,
        "unsatCore": mismatched,  # bases whose exponents cannot be reconciled
    }
    return {"verdict": FAIL, "certificate": cert,
            "reason": "z3:unit-dimension-mismatch", "backend": "z3"}


def _recheck_unit(cert: dict, verdict: Any) -> bool:
    lhs = _dim_vector(cert.get("lhs"))
    rhs = _dim_vector(cert.get("rhs"))
    if lhs is None or rhs is None:
        return False
    if verdict == PASS:
        model = cert.get("model")
        if not isinstance(model, dict):
            return False
        # The model must equal both sides on every base (the witness of equality).
        for b in _SI_BASES:
            if model.get(b) != lhs[b] or model.get(b) != rhs[b]:
                return False
        return lhs == rhs
    if verdict == FAIL:
        core = cert.get("unsatCore")
        if not isinstance(core, list) or not core:
            return False
        # Every base in the claimed core must genuinely differ (a real witness of
        # inconsistency), and at least one difference must exist.
        for b in core:
            if b not in _SI_BASES or lhs[b] == rhs[b]:
                return False
        return lhs != rhs
    return False


# --------------------------------------------------------------------------- #
# z3 backend — bounded-integer linear constraints
# --------------------------------------------------------------------------- #
_OPS = ("<=", "<", ">=", ">", "==", "!=")


def _z3_op(z3mod, op: str):
    return {
        "<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    }[op]


def _py_op(op: str):
    return {
        "<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    }[op]


def _parse_bounded_int(claim: dict):
    """Validate + extract (var_bounds, constraints, expected) or None if out of band."""
    varspec = claim.get("vars")
    constraints = claim.get("constraints")
    expected = claim.get("expected")
    if not isinstance(varspec, dict) or not varspec:
        return None
    if not isinstance(constraints, list):
        return None
    if expected not in ("sat", "unsat"):
        return None
    bounds: dict[str, tuple[int, int]] = {}
    for name, rng in varspec.items():
        if not isinstance(name, str) or not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            return None
        lo, hi = rng
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (lo, hi)):
            return None
        if lo > hi:
            return None
        bounds[name] = (lo, hi)
    parsed: list[tuple[dict, str, int]] = []
    for c in constraints:
        if not (isinstance(c, (list, tuple)) and len(c) == 3):
            return None
        coeffs, op, rhs = c
        if op not in _OPS or not isinstance(coeffs, dict):
            return None
        if not (isinstance(rhs, int) and not isinstance(rhs, bool)):
            return None
        for vn, cf in coeffs.items():
            if vn not in bounds or not (isinstance(cf, int) and not isinstance(cf, bool)):
                return None
        parsed.append((dict(coeffs), op, rhs))
    return bounds, parsed, expected


def _z3_check_bounded_int(claim: dict) -> dict:
    """Decide satisfiability of a bounded-integer linear system and compare it to
    the claim's ``expected`` sat/unsat. A match => "pass"; a mismatch => "fail".

    Certificate:
      * SAT: the satisfying assignment (a model the dumb checker plugs back in).
      * UNSAT: the full constraint set + bounds (the dumb checker exhaustively
        confirms no in-bounds assignment satisfies them — decidable because the
        variables are finitely bounded).
    """
    import z3

    parsed = _parse_bounded_int(claim)
    if parsed is None:
        return _abstain("bounded-int-claim-out-of-band", backend="z3")
    bounds, constraints, expected = parsed

    s = z3.Solver()
    zvars = {name: z3.Int(name) for name in bounds}
    for name, (lo, hi) in bounds.items():
        s.add(zvars[name] >= lo, zvars[name] <= hi)
    for coeffs, op, rhs in constraints:
        expr = z3.IntVal(0)
        for vn, cf in coeffs.items():
            expr = expr + cf * zvars[vn]
        s.add(_z3_op(z3, op)(expr, z3.IntVal(rhs)))

    result = s.check()
    is_sat = result == z3.sat
    solver_verdict = "sat" if is_sat else "unsat"
    matches = solver_verdict == expected

    if is_sat:
        m = s.model()
        assignment = {name: m[zvars[name]].as_long() for name in bounds}
    else:
        assignment = None

    cert = {
        "version": CERT_VERSION,
        "kind": KIND_BOUNDED_INT,
        "verdict": PASS if matches else FAIL,
        "solverVerdict": solver_verdict,
        "expected": expected,
        "bounds": {k: list(v) for k, v in bounds.items()},
        "constraints": [[c, o, r] for (c, o, r) in constraints],
        "assignment": assignment,  # present iff sat; the SAT witness
    }
    if matches:
        return {"verdict": PASS, "certificate": cert,
                "reason": f"z3:bounded-int-{solver_verdict}-as-expected", "backend": "z3"}
    return {"verdict": FAIL, "certificate": cert,
            "reason": f"z3:bounded-int-{solver_verdict}-but-expected-{expected}", "backend": "z3"}


def _eval_constraints(assignment: dict, constraints: list) -> bool:
    """True iff every linear constraint holds under ``assignment`` (exact ints)."""
    for coeffs, op, rhs in constraints:
        total = 0
        for vn, cf in coeffs.items():
            total += cf * int(assignment[vn])
        if not _py_op(op)(total, int(rhs)):
            return False
    return True


def _recheck_bounded_int(cert: dict, verdict: Any) -> bool:
    bounds = cert.get("bounds")
    constraints = cert.get("constraints")
    expected = cert.get("expected")
    solver_verdict = cert.get("solverVerdict")
    if not isinstance(bounds, dict) or not isinstance(constraints, list):
        return False
    if expected not in ("sat", "unsat") or solver_verdict not in ("sat", "unsat"):
        return False
    # The recorded verdict must be internally consistent with solver-vs-expected.
    expected_verdict = PASS if solver_verdict == expected else FAIL
    if verdict != expected_verdict:
        return False

    norm_constraints = [(c, o, r) for (c, o, r) in constraints]

    if solver_verdict == "sat":
        assignment = cert.get("assignment")
        if not isinstance(assignment, dict):
            return False
        # Witness must be IN BOUNDS and SATISFY every constraint — a genuine model.
        for name, rng in bounds.items():
            if name not in assignment:
                return False
            lo, hi = rng
            v = assignment[name]
            if not isinstance(v, int) or isinstance(v, bool) or not (lo <= v <= hi):
                return False
        return _eval_constraints(assignment, norm_constraints)

    # solver_verdict == "unsat": the dumb checker EXHAUSTIVELY confirms no in-bounds
    # assignment satisfies the constraints. Decidable because bounds are finite; we
    # cap the search space so a maliciously huge cert can't wedge the re-checker
    # (a cert whose space exceeds the cap cannot be independently confirmed => False,
    # fail-closed).
    import itertools

    names = list(bounds.keys())
    ranges = []
    space = 1
    for n in names:
        lo, hi = bounds[n]
        size = hi - lo + 1
        space *= size
        if space > 5_000_000:
            return False
        ranges.append(range(lo, hi + 1))
    for combo in itertools.product(*ranges):
        assignment = dict(zip(names, combo))
        if _eval_constraints(assignment, norm_constraints):
            return False  # found a model => not really unsat => cert is bogus
    return True


# --------------------------------------------------------------------------- #
# z3 backend — rational interval containment
# --------------------------------------------------------------------------- #
def _z3_check_interval(claim: dict) -> dict:
    """Decide whether ``value`` lies within the closed interval ``[lo, hi]`` and
    compare to the claim's ``expected`` boolean. Uses z3 Reals over exact
    rationals. Certificate carries the three rationals + the decided membership;
    the dumb checker recomputes ``lo <= value <= hi`` directly.
    """
    import z3

    value = _as_fraction(claim.get("value"))
    lo = _as_fraction(claim.get("lo"))
    hi = _as_fraction(claim.get("hi"))
    expected = claim.get("expected")
    if value is None or lo is None or hi is None or not isinstance(expected, bool):
        return _abstain("interval-claim-out-of-band", backend="z3")
    if lo > hi:
        return _abstain("interval-claim-out-of-band", backend="z3")

    v = z3.Q(value.numerator, value.denominator)
    lo_z = z3.Q(lo.numerator, lo.denominator)
    hi_z = z3.Q(hi.numerator, hi.denominator)
    s = z3.Solver()
    member = z3.And(v >= lo_z, v <= hi_z)
    # Is the claimed membership actually the case? Ask z3 whether NOT(claim) is
    # satisfiable given the fixed constants; unsat => the claim holds necessarily.
    target = member if expected else z3.Not(member)
    s.push()
    s.add(z3.Not(target))
    holds = s.check() == z3.unsat
    s.pop()

    cert = {
        "version": CERT_VERSION,
        "kind": KIND_INTERVAL,
        "verdict": PASS if holds else FAIL,
        "value": str(value),
        "lo": str(lo),
        "hi": str(hi),
        "expected": expected,
    }
    if holds:
        return {"verdict": PASS, "certificate": cert,
                "reason": "z3:interval-membership-as-expected", "backend": "z3"}
    return {"verdict": FAIL, "certificate": cert,
            "reason": "z3:interval-membership-contradicts-expected", "backend": "z3"}


def _recheck_interval(cert: dict, verdict: Any) -> bool:
    value = _as_fraction(cert.get("value"))
    lo = _as_fraction(cert.get("lo"))
    hi = _as_fraction(cert.get("hi"))
    expected = cert.get("expected")
    if value is None or lo is None or hi is None or not isinstance(expected, bool):
        return False
    if lo > hi:
        return False
    actual_member = (lo <= value <= hi)
    holds = (actual_member == expected)
    expected_verdict = PASS if holds else FAIL
    return verdict == expected_verdict


# --------------------------------------------------------------------------- #
# CLI — prints a JSON receipt to stdout, prose to stderr; fail-closed exit codes.
# --------------------------------------------------------------------------- #
def _cli(argv: "list[str] | None" = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="Certificate-carrying SMT rung: check a decidable-band claim "
                    "(fail-closed ABSTAIN when z3 is absent).")
    p.add_argument("--claim", help="a JSON claim object (see agent.smt_verifier.check)")
    p.add_argument("--claim-file", help="path to a JSON file holding the claim")
    p.add_argument("--recheck", action="store_true",
                   help="treat --claim/--claim-file as a certificate and re-verify it")
    args = p.parse_args(argv)

    raw: Any = None
    if args.claim_file:
        try:
            with open(args.claim_file, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot read claim file: {exc}", file=sys.stderr)
            return 2
    elif args.claim:
        try:
            raw = json.loads(args.claim)
        except json.JSONDecodeError as exc:
            print(f"cannot parse --claim JSON: {exc}", file=sys.stderr)
            return 2
    else:
        print("provide --claim or --claim-file", file=sys.stderr)
        return 2

    if args.recheck:
        ok = recheck_certificate(raw)
        receipt = {"rechecked": bool(ok), "accepted": bool(ok)}
        print(json.dumps(receipt))
        print(f"certificate {'ACCEPTED' if ok else 'REJECTED'} by independent re-checker",
              file=sys.stderr)
        return 0 if ok else 1

    result = check(raw)
    print(json.dumps(result))
    verdict = result["verdict"]
    print(f"verdict={verdict} reason={result['reason']} backend={result['backend']}",
          file=sys.stderr)
    if verdict == PASS:
        return 0
    if verdict == ABSTAIN:
        # Abstain is a HELD verdict, not a pass. Exit 3 (NO-GO family) so a caller
        # never mistakes an un-run gate for a green one.
        return 3
    return 1  # FAIL


__all__ = [
    "check",
    "recheck_certificate",
    "classify",
    "z3_available",
    "CERT_VERSION",
    "PASS",
    "FAIL",
    "ABSTAIN",
    "KIND_UNIT",
    "KIND_BOUNDED_INT",
    "KIND_INTERVAL",
]


if __name__ == "__main__":
    import sys

    sys.exit(_cli())
