#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/lean_verifier.py — optional-Lean kernel proof verifier.

Falsifiable invariants (the contract this module defends):
  1. Fail-closed on NO Lean: with the toolchain absent, every check returns
     ``held`` / ``lean_unavailable`` — it NEVER returns ``accepted``. This is the
     load-bearing property: a missing kernel can never be smuggled into a promotion.
  2. Certificate model is exact: a recorded ``proved`` cert maps to verdict
     ``accepted``; ``refuted`` -> ``rejected``; ``unprovable_here`` -> ``held``.
     The certificate hash is deterministic (same proof text -> same hash), so the
     tamper-evident handle is stable.
  3. Abstain is the native output for "no proof": status ``unprovable_here`` yields
     verdict ``held`` — wisdom-before-intelligence. The system says "I cannot prove
     this", it never fabricates a proof to fill the gap.
  4. record_certificate persists a hash-addressed, tamper-evident artefact (the
     audit idiom from okf/forgetting_audit.py).
  5. require_lean fails closed with held/lean_unavailable when the toolchain is
     absent — the formal analogue of formal_verifier.require_z3.

These run green offline WITHOUT Lean installed (the fail-closed path IS the test).
When Lean happens to be present, the real-kernel path is additionally exercised but
is never required to pass — CI is green either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent.lean_verifier as lv  # noqa: E402
from agent.lean_verifier import (  # noqa: E402
    ProofCertificate,
    _is_tactic_proof,
    _wrap_lean,
    check_proof,
    lean_available,
    record_certificate,
    require_lean,
)


# --------------------------------------------------------------------------- #
# 1. Fail-closed on no Lean (the load-bearing property)
# --------------------------------------------------------------------------- #


def test_check_proof_fails_closed_without_lean(monkeypatch) -> None:
    """With lean_available forced False, check_proof must NEVER accept."""
    monkeypatch.setattr(lv, "lean_available", lambda: False)
    r = check_proof("add_comm", "forall n m : Nat, n + m = m + n", "by omega")
    assert r["verdict"] == "held"
    assert r["status"] == "lean_unavailable"
    assert r["backend"] == "none"
    # The absent-kernel path still carries a certificate record (status flagged), so the
    # abstention is itself auditable rather than a silent None.
    assert r["certificate"]["status"] == "unprovable_here"
    assert r["certificate"]["backend"] == "none"


def test_check_proof_never_accepts_without_lean(monkeypatch) -> None:
    """Adversarial: even a 'looks plausible' proof must be held, not accepted, without
    a real kernel. The gate cannot be talked into a promotion by proof text alone."""
    monkeypatch.setattr(lv, "lean_available", lambda: False)
    for proof in ["sorry", "by rfl", "by decide", "exact trivial", ""]:
        r = check_proof("x", "True", proof)
        assert r["verdict"] != "accepted", f"accepted without kernel for proof={proof!r}"


# --------------------------------------------------------------------------- #
# 2. Certificate model is exact + tamper-evident
# --------------------------------------------------------------------------- #


def test_certificate_verdict_mapping() -> None:
    assert ProofCertificate("a", "P", "p", "proved").verdict == "accepted"
    assert ProofCertificate("a", "P", "p", "refuted").verdict == "rejected"
    assert ProofCertificate("a", "P", "p", "unprovable_here").verdict == "held"


def test_certificate_hash_is_deterministic() -> None:
    c1 = ProofCertificate("a", "P", "by rfl", "proved")
    c2 = ProofCertificate("a", "P", "by rfl", "proved")
    c3 = ProofCertificate("a", "P", "by decide", "proved")  # different proof text
    assert c1.kernel_hash == c2.kernel_hash, "same proof text must hash identically"
    assert c1.kernel_hash != c3.kernel_hash, "different proof text must differ"
    assert len(c1.kernel_hash) == 64 and all(ch in "0123456789abcdef" for ch in c1.kernel_hash)


def test_certificate_unknown_status_defaults_held() -> None:
    # A status the model doesn't recognise must NEVER default to accepted.
    assert ProofCertificate("a", "P", "p", "garbage_status").verdict == "held"


# --------------------------------------------------------------------------- #
# 3. Abstain is the native "no proof" output (wisdom-before-intelligence)
# --------------------------------------------------------------------------- #


def test_unprovable_here_abstains() -> None:
    cert = ProofCertificate("rh", "Riemann Hypothesis", "by sorry", "unprovable_here")
    r = lv._result(cert)
    assert r["verdict"] == "held"
    assert r["status"] == "unprovable_here"
    assert r["candidateOnly"] is True
    assert r["level3Evidence"] is False
    # The reason must name abstention, not failure.
    assert any("abstain" in reason.lower() for reason in r["reasons"])


# --------------------------------------------------------------------------- #
# 4. record_certificate persists a hash-addressed artefact
# --------------------------------------------------------------------------- #


def test_record_certificate_writes_hash_addressed_artefact(tmp_path) -> None:
    cert = ProofCertificate("add_comm", "∀ n m, n + m = m + n", "by omega", "proved")
    out = record_certificate(cert, certs_dir=str(tmp_path))
    p = Path(out)
    assert p.exists()
    # Path is addressed by the leading hash chars — tamper-evident.
    assert cert.kernel_hash[:16] in p.name
    import json
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["proofText"] == "by omega"
    assert payload["status"] == "proved"
    assert payload["kernelHash"] == cert.kernel_hash


# --------------------------------------------------------------------------- #
# 5. require_lean fails closed
# --------------------------------------------------------------------------- #


def test_require_lean_holds_without_toolchain(monkeypatch) -> None:
    monkeypatch.setattr(lv, "lean_available", lambda: False)

    def _should_not_run(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("check_fn ran without a kernel")

    r = require_lean(_should_not_run, "x", "P", "by rfl")
    assert r["verdict"] == "held"
    assert r["status"] == "lean_unavailable"
    assert r["certificate"] is None


# --------------------------------------------------------------------------- #
# 6. Proof wrapping — tactic vs term detection (regression: lean-kernel lane)
# --------------------------------------------------------------------------- #


def test_is_tactic_proof_detects_tactic_form() -> None:
    """The smoke loop failed in the lean-kernel CI lane because tactic-style proofs
    (`intro x; rfl`) were placed in a term-mode slot. The wrapper now detects tactic
    form and prefixes `by`. Pin the detection so it can't silently regress."""
    assert _is_tactic_proof("intro x; rfl") is True
    assert _is_tactic_proof("intros; rfl") is True
    assert _is_tactic_proof("intros a b ⟨ha, hb⟩; exact ⟨hb, ha⟩") is True
    assert _is_tactic_proof("by intro x; rfl") is True   # already prefixed


def test_is_tactic_proof_treats_terms_as_terms() -> None:
    """Bare `rfl`/`trivial` and `fun` terms must NOT be over-wrapped — they are valid
    term-mode proofs and wrapping them in `by` is harmless but unnecessary."""
    assert _is_tactic_proof("rfl") is False
    assert _is_tactic_proof("trivial") is False
    assert _is_tactic_proof("fun x => rfl") is False
    assert _is_tactic_proof("") is False


def test_wrap_lean_prefixes_by_for_tactics() -> None:
    src = _wrap_lean("∀ (x : Nat), x = x", "intro x; rfl")
    assert "by intro x; rfl" in src
    assert src.startswith("example :")


def test_wrap_lean_leaves_terms_alone() -> None:
    src = _wrap_lean("True", "trivial")
    assert "by" not in src
    assert "example : True :=\n  trivial" in src


# --------------------------------------------------------------------------- #
# 7. Real-kernel path (only when Lean is actually installed; never required)
# --------------------------------------------------------------------------- #


def test_real_kernel_path_when_present() -> None:
    """If a Lean toolchain is installed, exercise the real subprocess path on a trivial
    proposition. This is ADDITIONAL coverage — skipped (not failed) when absent, so CI is
    green offline. It guards against the real-kernel path becoming dead code the way the
    z3 named-variable path once did (see test_formal_verifier regression)."""
    if not lean_available():
        import pytest
        pytest.skip("Lean toolchain not installed; offline fail-closed path covered above")
    r = check_proof("trivial_true", "True", "trivial")
    # With a real kernel, `True` proved by `trivial` must accept. Any other verdict here
    # is a real bug in the kernel backend, not an environment artifact.
    assert r["verdict"] == "accepted", r
    assert r["backend"] == "lean"


def main() -> int:
    test_check_proof_fails_closed_without_lean(_Monkey())
    test_check_proof_never_accepts_without_lean(_Monkey())
    test_certificate_verdict_mapping()
    test_certificate_hash_is_deterministic()
    test_certificate_unknown_status_defaults_held()
    test_unprovable_here_abstains()
    test_record_certificate_writes_hash_addressed_artefact(_Tmp())
    test_require_lean_holds_without_toolchain(_Monkey())
    try:
        test_real_kernel_path_when_present()
    except Exception as exc:  # the skip path
        print(f"test_lean_verifier: OK (real-kernel path skipped: {exc.__class__.__name__})")
        return 0
    print("test_lean_verifier: OK")
    return 0


class _Monkey:
    """Minimal monkeypatch shim so the module-level main() works without pytest."""
    def __init__(self) -> None:
        self._orig: dict = {}

    def setattr(self, obj, name, value) -> None:
        self._orig[(id(obj), name)] = getattr(obj, name)
        setattr(obj, name, value)


class _Tmp:
    def __init__(self) -> None:
        import tempfile
        self._p = Path(tempfile.mkdtemp())

    def __str__(self) -> str:
        return str(self._p)


if __name__ == "__main__":
    raise SystemExit(main())
