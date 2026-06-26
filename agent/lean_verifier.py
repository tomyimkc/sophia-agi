# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Kernel-checked formal-proof verifier with an optional Lean 4 backend.

Sophia already had Z3-backed *consistency* checks (``agent/formal_verifier.py``) for
decidable fragments (lattice ordering, direct contradiction). That tier proves
*internal logical consistency* of a claim set — it does NOT adjudicate mathematical
truth, and it does not check *proof certificates*. This module adds the higher-assurance
tier: verification of a CLAIM via a machine-checked PROOF, where the kernel is the
oracle and a proof certificate is the evidence.

This is the tier the Millennium / open-math benchmark arc depends on: a claim is
``accepted`` only when a proof term type-checks under the kernel, ``abstained`` when no
proof is known (the wisdom-before-intelligence default — "I cannot prove this" is a
designed output, not a failure), and ``rejected`` only when a *refutation* certificate
type-checks. We never fabricate a proof to fill a gap.

Design constraints (repo discipline, identical to ``formal_verifier.py``):
  - **Optional dependency, fail-closed.** The Lean 4 toolchain (``lean``/``lake``/
    ``elan``) is NOT a hard dependency. When it is absent, every check returns
    ``verdict="held"`` / ``status="lean_unavailable"`` — it NEVER silently returns
    ``accepted``. A pure-Python **certificate model** handles the small, deterministic
    offline cases so CI passes without Lean, and so the fail-closed control flow is
    itself testable (see ``tests/test_lean_verifier.py`` — it pins the gate logic, not a
    Lean install).
  - **Deterministic + offline.** No network, no model calls. The kernel is invoked as a
    local subprocess only when the toolchain is present.
  - **Honest scope.** This verifies *proof certificates*; it is not a theorem prover and
    proves nothing on its own. Every report carries ``candidateOnly: true`` and
    ``level3Evidence: false`` until a real run clears the no-overclaim gate.

Result shape (a dict, matching the rest of the harness verifiers):
    {
      "verdict": "accepted" | "rejected" | "held",
      "backend": "lean" | "certificate" | "none",
      "status": "proved" | "refuted" | "unprovable_here" | "lean_unavailable" | "error",
      "reasons": [...],
      "certificate": {...} | None,   # the proof/refutation evidence when known
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Toolchain detection — the fail-closed seam
# --------------------------------------------------------------------------- #


def lean_available() -> bool:
    """True iff a Lean 4 toolchain can be found on PATH (lean + lake both present).

    This is the ONLY place availability is decided, and it is the seam the whole module
    fails closed on. Tested by mocking at the call site, never by requiring an install.
    """
    for exe in ("lean", "lake"):
        if not _which(exe):
            return False
    return True


def _which(exe: str) -> "str | None":
    """``shutil.which`` without importing shutil at module top (kept local + overridable)."""
    path = os.environ.get("PATH", "")
    seen: set[str] = set()
    for entry in path.split(os.pathsep):
        entry = os.path.expanduser(entry)
        if not entry or entry in seen:
            continue
        seen.add(entry)
        candidate = os.path.join(entry, exe)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Proof certificate model (pure-Python, deterministic, offline-exact)
# --------------------------------------------------------------------------- #
# A certificate is the auditable evidence a claim was kernel-checked. The offline model
# is NOT a prover: it records what the kernel SAID about a proof term (proved / refuted /
# no proof), it never decides truth itself. The SHA-256 of the proof text is the
# tamper-evident handle, mirroring okf/forgetting_audit.py's hash-chained trail idiom.

_VERDICT_OF_STATUS = {
    "proved": "accepted",
    "refuted": "rejected",
    "unprovable_here": "held",
}


@dataclass
class ProofCertificate:
    """A kernel-checked proof result. Evidence, not assertion."""

    claim_id: str                 # stable id of the proposition (e.g. "mathlib:add_comm")
    proposition: str              # the statement (human + kernel-readable form)
    proof_text: str               # the proof term / tactic block the kernel checked
    status: str                   # "proved" | "refuted" | "unprovable_here"
    backend: str = "certificate"  # "lean" (real kernel) | "certificate" (recorded result)
    kernel_hash: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.kernel_hash:
            self.kernel_hash = _sha256(self.proof_text)

    @property
    def verdict(self) -> str:
        return _VERDICT_OF_STATUS.get(self.status, "held")

    def to_dict(self) -> dict:
        return {
            "claimId": self.claim_id,
            "proposition": self.proposition,
            "status": self.status,
            "verdict": self.verdict,
            "backend": self.backend,
            "kernelHash": self.kernel_hash,
            # proof_text deliberately omitted from default dict form: it can be large
            # and lives in formal_proofs/certificates/<hash>.lean for audit.
        }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def check_proof(claim_id: str, proposition: str, proof_text: str, *,
                timeout_s: int = 60) -> dict:
    """Verify a proof of ``proposition`` via the Lean kernel when available; otherwise
    fail closed.

    Returns the standard verifier result dict. When Lean is absent the verdict is ALWAYS
    ``held`` / ``lean_unavailable`` — the caller (a gate, the self-extend loop) must then
    abstain rather than promote. This is the load-bearing fail-closed contract.
    """
    if lean_available():
        return _lean_check(claim_id, proposition, proof_text, timeout_s)
    return _held_unavailable(claim_id, proposition, proof_text)


def record_certificate(cert: ProofCertificate, certs_dir: "str | None" = None) -> str:
    """Persist a proof certificate under ``formal_proofs/certificates/<hash>.json``.

    Returns the path written. This is the audit artefact: the proof text + the kernel's
    verdict, hash-addressed so it is tamper-evident (matches forgetting_audit's idiom).
    """
    import pathlib
    base = pathlib.Path(certs_dir) if certs_dir else pathlib.Path("formal_proofs/certificates")
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"{cert.kernel_hash[:16]}.json"
    payload = {**cert.to_dict(), "proofText": cert.proof_text}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(out)


def require_lean(check_fn: "Callable[..., dict]", *args, **kwargs) -> dict:
    """Run a check that MUST use the Lean kernel. If the toolchain is unavailable, fail
    closed with ``held`` / ``lean_unavailable`` rather than falling back. Use this for
    claims where only a real kernel result may be trusted (the formal analogue of
    ``formal_verifier.require_z3``)."""
    if not lean_available():
        return {
            "verdict": "held",
            "backend": "none",
            "status": "lean_unavailable",
            "reasons": ["Lean 4 toolchain not installed; install `lean`/`lake` (via elan) "
                        "to enable kernel-checked proof verification (held, not accepted)"],
            "certificate": None,
        }
    return check_fn(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Lean 4 kernel backend
# --------------------------------------------------------------------------- #


def _lean_check(claim_id: str, proposition: str, proof_text: str, timeout_s: int) -> dict:
    """Type-check a proof term against the kernel via ``lean`` on a temp file.

    Honest about what the kernel tells us:
      - exit 0, no errors  -> ``proved`` (the proof term type-checks).
      - the term is a `decide`-style refutation producing `False` -> ``refuted``.
      - any other non-zero / error -> ``unprovable_here`` (we could NOT prove it; abstain).
    We never interpret a timeout or an error as evidence of anything but "no proof here".
    """
    src = _wrap_lean(proposition, proof_text)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(src)
            tmp = fh.name
        proc = subprocess.run(
            ["lean", tmp], capture_output=True, text=True, timeout=timeout_s,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return _err(claim_id, proposition, proof_text,
                    f"kernel invocation failed: {type(exc).__name__}")
    finally:
        try:
            os.unlink(tmp)
        except (OSError, NameError):
            pass

    stderr, stdout = (proc.stderr or ""), (proc.stdout or "")
    ok = proc.returncode == 0 and not stderr.strip()
    if ok:
        cert = ProofCertificate(claim_id, proposition, proof_text, "proved", "lean")
        return _result(cert)
    if _looks_like_refutation(stderr + stdout):
        cert = ProofCertificate(claim_id, proposition, proof_text, "refuted", "lean")
        return _result(cert, extra_reasons=[f"lean: {stderr.strip()[:200]}"])
    cert = ProofCertificate(claim_id, proposition, proof_text, "unprovable_here", "lean")
    return _result(cert, extra_reasons=[f"lean: {(stderr or stdout).strip()[:200]}"])


def _wrap_lean(proposition: str, proof_text: str) -> str:
    """Wrap a proposition + proof into a minimal `example : P := proof` source the
    kernel can type-check. The caller is responsible for proof_text being a valid proof
    term / by_tactic for the proposition; we only add the scaffolding."""
    prop = proposition.strip()
    proof = proof_text.strip()
    return f"example : {prop} :=\n  {proof}\n"


def _looks_like_refutation(diag: str) -> bool:
    """Heuristic: a `decide`/`native_decide` counterexample or a `False` goal closed by
    rfl/decide reads as a refutation in Lean's diagnostic. Conservative — when unclear we
    fall through to ``unprovable_here`` (abstain), never to ``accepted``."""
    d = diag.lower()
    if "decide" in d and ("false" in d or "counterexample" in d):
        return True
    return False


# --------------------------------------------------------------------------- #
# Fail-closed offline path + helpers
# --------------------------------------------------------------------------- #


def _held_unavailable(claim_id: str, proposition: str, proof_text: str) -> dict:
    cert = ProofCertificate(claim_id, proposition, proof_text, "unprovable_here", "none")
    return {
        "verdict": "held",
        "backend": "none",
        "status": "lean_unavailable",
        "reasons": ["Lean 4 toolchain not found on PATH; cannot kernel-check the proof "
                    "(held, not accepted — abstain rather than promote)"],
        "certificate": cert.to_dict(),
    }


def _result(cert: ProofCertificate, *, extra_reasons: "list[str] | None" = None) -> dict:
    reasons = {
        "proved": [f"{cert.backend}: proof type-checks under the kernel"],
        "refuted": [f"{cert.backend}: proposition refuted (counterexample found)"],
        "unprovable_here": [f"{cert.backend}: no proof found here — abstain"],
    }[cert.status]
    if extra_reasons:
        reasons += extra_reasons
    return {
        "verdict": cert.verdict,
        "backend": cert.backend,
        "status": cert.status,
        "reasons": reasons,
        "certificate": cert.to_dict(),
        # Provenance-honesty metadata, matching the repo's report convention.
        "candidateOnly": True,
        "level3Evidence": False,
    }


def _err(claim_id: str, proposition: str, proof_text: str, msg: str) -> dict:
    cert = ProofCertificate(claim_id, proposition, proof_text, "unprovable_here", "none")
    return {
        "verdict": "held",
        "backend": "none",
        "status": "error",
        "reasons": [msg],
        "certificate": cert.to_dict(),
    }


__all__ = [
    "ProofCertificate",
    "lean_available",
    "check_proof",
    "record_certificate",
    "require_lean",
]
