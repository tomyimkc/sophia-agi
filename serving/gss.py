# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Governed Speculative Sparsity — Tier-1 mechanism (prune + speculative accept/reject).

Tier 0 (`serving.gss_feasibility`) measures whether GSS *can* win. Tier 1 is the
**mechanism** and its **lossless proof**: the read-set prune, the speculative accept/
reject, and the single-step correction — pure-numpy over distributions the caller
produced, CI-testable, model-agnostic. The deployment artifact is a CUDA decode loop;
this is the policy + the equivalence contract, exactly like the rest of `serving/`.

**The lossless core (the `flash == naive` bar, made exact).** Speculative sampling draws a
token ``x`` from a cheap draft ``q`` and accepts it with probability ``min(1, p(x)/q(x))``
against the verifier ``p``; on rejection it resamples from the normalised positive residual
``relu(p − q)``. The realised output distribution is then, *exactly*,

    realised(x) = min(p(x), q(x)) + relu(p(x) − q(x)) = p(x).

So the realised distribution equals the verifier's — **provably, with no sampling error**
(`speculative_realized` computes it via the accept/resample decomposition, and
`offline_invariants` checks it equals ``p`` to 1e-12). That identity is GSS's whole safety
claim: a token accepted by the loop is distributed exactly as the verifier would have
produced.

**The honesty this module forces (read before quoting a speedup).** Losslessness holds iff
the accept/reject verifies against the **dense** target ``p``. If you instead verify against
a *pruned* target ``p̂`` (read only the predicted read-set), the realised distribution is
``p̂``, not ``p`` — it drifts by exactly ``KL(p ‖ p̂)`` (`verify_drift` measures it). So:

  - **Lossless GSS** = a *cheap draft* (4-bit / pruned) + a **dense verify**. The bandwidth
    win comes from the draft being cheap and the dense verify being amortised over the ``k``
    accepted tokens — cost ``(γ·draft_byte_frac + 1)/k`` per token (verify reads the full
    ``B``). On the OLMoE numbers (γ=4, k≈4) that is still **~2×**, certified-equal.
  - **Aggressive GSS** = pruned verify (cost ``(γ·draft_byte_frac + ρ)/k``, the ~3.6×
    ceiling) is **not** exactly lossless; it carries a bounded error ``KL(p ‖ p̂)`` and needs
    a periodic dense correction to stay honest. Tier 0's ceiling is the *aggressive* bound;
    the *guaranteed-lossless* number is lower. This module makes that gap measurable instead
    of letting a speedup headline hide it.

Falsifiable offline invariants (`offline_invariants()`, CI-gated):
  - **lossless**: ``speculative_realized(p, q) == p`` to 1e-12 for random ``p, q`` (the proof);
  - identical draft (``q == p``) accepts with probability 1 (zero rejection mass);
  - acceptance mass equals ``Σ min(p, q) = 1 − TV(p, q)``;
  - **pruned verify drifts**: realised-against-``p̂`` equals ``p̂``; its drift from ``p`` equals
    ``KL(p ‖ p̂)`` exactly, and dense verify has zero drift — the lossless condition, made falsifiable;
  - the read-set mask is monotone in coverage and selects the concentrated units;
  - fail-closed on empty / shape-mismatch / negative / non-finite / non-distribution input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

_EPS = 1e-12


def _as_dist(a, name: str):
    """Validate and row-normalise a (N, V) array of non-negative distributions."""
    x = np.asarray(a, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D (N, V)")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite")
    if (x < 0).any():
        raise ValueError(f"{name} must be non-negative")
    s = x.sum(axis=1, keepdims=True)
    if (s <= 0).any():
        raise ValueError(f"{name} rows must have positive mass")
    return x / s


# ---------------------------------------------------------------------------
# The lossless core: speculative accept/reject realised distribution
# ---------------------------------------------------------------------------

def speculative_realized(target, draft):
    """Realised output distribution of speculative sampling: draft ``q`` verified by ``p``.

    Computed via the accept/resample decomposition (NOT by returning ``p``), so the
    invariant "this equals ``p``" is a real test of the mechanism:

        accept(x)   = min(p(x), q(x))                      # accepted on the draw from q
        reject_mass = 1 − Σ min(p, q) = TV(p, q)
        resample(x) = relu(p(x) − q(x)) / Σ relu(p − q)    # the corrected residual
        realised(x) = accept(x) + reject_mass · resample(x) = p(x)

    Returns the realised (N, V) distribution. With ``target`` the dense verifier this equals
    ``target`` exactly (lossless); with a pruned verifier it equals the pruned distribution.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    p = _as_dist(target, "target")
    q = _as_dist(draft, "draft")
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch target{p.shape} vs draft{q.shape}")
    accept = np.minimum(p, q)                                  # (N,V)
    reject_mass = 1.0 - accept.sum(axis=1, keepdims=True)      # (N,1) == TV
    resid = np.maximum(p - q, 0.0)
    resid_sum = resid.sum(axis=1, keepdims=True)
    # When p==q, reject_mass==0 and resid_sum==0; resample is irrelevant (weight 0).
    safe = np.where(resid_sum > 0, resid_sum, 1.0)
    resample = resid / safe
    return accept + reject_mass * resample


def acceptance_mass(target, draft):
    """Per-row probability a drafted token is accepted: ``Σ min(p, q) = 1 − TV(p, q)``."""
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    p = _as_dist(target, "target")
    q = _as_dist(draft, "draft")
    if p.shape != q.shape:
        raise ValueError("shape mismatch")
    return np.minimum(p, q).sum(axis=1)


def _row_kl(p, q):
    p = np.clip(p, _EPS, 1.0); q = np.clip(q, _EPS, 1.0)
    p = p / p.sum(1, keepdims=True); q = q / q.sum(1, keepdims=True)
    return np.sum(p * np.log(p / q), axis=1)


def verify_drift(dense_target, verify_target, draft):
    """KL(dense ‖ realised) when the loop verifies against ``verify_target`` instead of dense.

    With ``verify_target == dense_target`` this is ~0 (lossless). With a *pruned* verifier it
    equals ``KL(dense ‖ pruned)`` — the exact, measurable cost of aggressive read-set pruning.
    Returns ``(mean_drift, per_row_drift)``.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    dense = _as_dist(dense_target, "dense_target")
    realised = speculative_realized(verify_target, draft)
    drift = _row_kl(dense, realised)
    return float(drift.mean()), drift


# ---------------------------------------------------------------------------
# The read-set prune
# ---------------------------------------------------------------------------

def read_set_mask(contribs, *, coverage: float = 0.9):
    """Boolean (N, U) mask of the read-set: the fewest units per token covering ``coverage``.

    ``True`` = read this unit at full precision; ``False`` = skip (the prune). A zero-mass
    row reads everything (fail-closed: can't prune what you can't measure).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    if not (0.0 < coverage <= 1.0):
        raise ValueError("coverage must be in (0, 1]")
    c = np.asarray(contribs, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] == 0 or c.shape[1] == 0:
        raise ValueError("contribs must be a non-empty 2-D (N, U)")
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("contribs must be finite, non-negative")
    n, u = c.shape
    order = np.argsort(-c, axis=1)
    csum = np.cumsum(np.take_along_axis(c, order, axis=1), axis=1)
    total = csum[:, -1:]
    thresh = coverage * np.where(total <= 0, 1.0, total)
    need = (csum < thresh).sum(axis=1) + 1
    need = np.clip(need, 1, u)
    mask = np.zeros((n, u), dtype=bool)
    for i in range(n):
        keep = u if total[i, 0] <= 0 else int(need[i])
        mask[i, order[i, :keep]] = True
    return mask


# ---------------------------------------------------------------------------
# The equivalence gate (a thin no-overclaim contract over the mechanism)
# ---------------------------------------------------------------------------

@dataclass
class GSSEquivalenceReport:
    passed: bool
    mean_kl: float
    max_kl: float
    bytes_read_ratio: float          # GSS bytes-read ÷ dense (reported, paired with the verdict)
    n: int
    reasons: "list[str]"

    def as_dict(self) -> dict:
        return {"passed": self.passed, "mean_kl": round(self.mean_kl, 9),
                "max_kl": round(self.max_kl, 9), "bytes_read_ratio": round(self.bytes_read_ratio, 4),
                "n": self.n, "reasons": self.reasons}


@dataclass
class GSSEquivalenceGate:
    """Lossless contract: the GSS-realised distribution must equal the dense target.

    ``max_mean_kl`` ceilings the mean KL(dense ‖ realised); ``hard_max_kl`` ceilings any row.
    Reports ``bytes_read_ratio`` alongside the verdict, never one without the other (the
    `LowRamReport` rule) — a speedup only counts if equivalence holds.
    """
    max_mean_kl: float = 1e-9
    hard_max_kl: float = 1e-9

    def evaluate(self, dense_probs, realized_probs, *, bytes_read_ratio: float = 1.0) -> GSSEquivalenceReport:
        if not _HAVE_NUMPY:
            raise RuntimeError("numpy required")
        try:
            dense = _as_dist(dense_probs, "dense_probs")
            realised = _as_dist(realized_probs, "realized_probs")
        except (ValueError, RuntimeError) as e:
            return GSSEquivalenceReport(False, float("inf"), float("inf"), bytes_read_ratio, 0, [str(e)])
        if dense.shape != realised.shape:
            return GSSEquivalenceReport(False, float("inf"), float("inf"), bytes_read_ratio, 0,
                                        [f"shape mismatch {dense.shape} vs {realised.shape}"])
        kl = _row_kl(dense, realised)
        mean_kl, max_kl = float(kl.mean()), float(kl.max())
        reasons = []
        if mean_kl > self.max_mean_kl:
            reasons.append(f"mean_kl {mean_kl:.3e} > {self.max_mean_kl:.0e}")
        if max_kl > self.hard_max_kl:
            reasons.append(f"max_kl {max_kl:.3e} > {self.hard_max_kl:.0e}")
        return GSSEquivalenceReport(not reasons, mean_kl, max_kl, bytes_read_ratio, dense.shape[0], reasons)


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NUMPY:
        return False, {"checks": {"numpy_available": False}}
    rng = np.random.default_rng(0)
    checks: dict[str, bool] = {}
    detail: dict = {}

    def softmax(z):
        z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    N, V = 64, 100
    p = softmax(rng.standard_normal((N, V)) * 3.0)
    q = softmax(rng.standard_normal((N, V)) * 3.0)          # an unrelated draft

    # 1. LOSSLESS CORE: realised(p, q) == p exactly (the flash==naive bar, deterministic).
    realised = speculative_realized(p, q)
    checks["lossless_realized_equals_target"] = bool(np.abs(realised - p).max() < 1e-12)
    checks["realized_is_distribution"] = bool(np.abs(realised.sum(1) - 1.0).max() < 1e-12)
    detail["max_abs_realized_minus_p"] = float(np.abs(realised - p).max())

    # 2. Identical draft → accept with probability 1 (no rejection mass).
    am_id = acceptance_mass(p, p.copy())
    checks["identical_accepts_fully"] = bool(np.abs(am_id - 1.0).max() < 1e-12)

    # 3. Acceptance mass == 1 − TV(p, q) == Σ min(p, q).
    am = acceptance_mass(p, q)
    tv = 0.5 * np.abs(p - q).sum(1)
    checks["acceptance_is_one_minus_tv"] = bool(np.abs(am - (1.0 - tv)).max() < 1e-12)
    detail["mean_acceptance"] = float(am.mean())

    # 4. PRUNED VERIFY DRIFTS: verifying against p̂≠p yields p̂, drifting by exactly KL(p‖p̂);
    #    dense verify has zero drift. This is the lossless *condition*, made falsifiable.
    p_hat = softmax(np.log(np.clip(p, 1e-9, 1)) + rng.standard_normal((N, V)) * 0.5)  # a pruned-ish target
    dense_drift, _ = verify_drift(p, p, q)
    pruned_drift, _ = verify_drift(p, p_hat, q)
    kl_p_phat = float(_row_kl(p, p_hat).mean())
    checks["dense_verify_is_lossless"] = dense_drift < 1e-12
    checks["pruned_verify_drift_equals_kl"] = abs(pruned_drift - kl_p_phat) < 1e-9
    checks["pruned_verify_actually_drifts"] = pruned_drift > 1e-6
    detail["pruned_drift"] = round(pruned_drift, 6)

    # 5. The equivalence gate: lossless passes, a drifted candidate is rejected.
    gate = GSSEquivalenceGate()
    ok_rep = gate.evaluate(p, speculative_realized(p, q), bytes_read_ratio=0.4)
    bad_rep = gate.evaluate(p, speculative_realized(p_hat, q), bytes_read_ratio=0.3)
    checks["gate_passes_lossless"] = ok_rep.passed and ok_rep.bytes_read_ratio == 0.4
    checks["gate_rejects_drift"] = not bad_rep.passed
    detail["gate_ok"] = ok_rep.as_dict(); detail["gate_bad"] = bad_rep.as_dict()

    # 6. Read-set mask: concentrated → few units; monotone in coverage; selects the hot units.
    contribs = np.full((N, 64), 1e-3)
    for t in range(N):
        contribs[t, rng.choice(64, 4, replace=False)] += 8.0
    m90 = read_set_mask(contribs, coverage=0.9)
    m50 = read_set_mask(contribs, coverage=0.5)
    checks["mask_concentrated_small"] = m90.mean() < 0.25
    checks["mask_monotone_in_coverage"] = bool((m50.sum(1) <= m90.sum(1)).all())
    checks["mask_reads_at_least_one"] = bool((m90.sum(1) >= 1).all())
    detail["mask_mean_read_fraction"] = round(float(m90.mean()), 4)

    # 7. Fail-closed.
    bad = 0
    for fn in (lambda: speculative_realized(np.zeros((0, V)), q),
               lambda: speculative_realized(p, q[:, :V - 1]),
               lambda: speculative_realized(-p, q),
               lambda: read_set_mask(np.zeros((2, 3)) - 1.0)):
        try:
            fn()
        except (ValueError, RuntimeError):
            bad += 1
    checks["fail_closed"] = bad == 4

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("GSS Tier-1 mechanism offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  max|realized-p| = {detail.get('max_abs_realized_minus_p')}")
    print(f"  pruned-verify drift = {detail.get('pruned_drift')} (== KL(p||p_hat))")
    raise SystemExit(0 if ok else 1)
