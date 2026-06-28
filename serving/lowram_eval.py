# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Low-RAM serving measurement — the no-overclaim gate for streamed+quantized inference.

*Why this exists.* :mod:`serving.layer_stream` and :mod:`serving.expert_offload` are the
*mechanism* for running a big model in little fast memory; :mod:`moe.quant` shrinks the
shards. None of them, on their own, license the claim "this 70B model runs in <4GB **at
quality**". As ``serving/expert_offload.py`` says outright: a low-RAM capability claim "still
needs the quantized experts evaluated against FP16 on a held-out set to the no-overclaim
gate — this module is the mechanism, not the measurement." This module *is* the measurement.

*What it does.* Given the full-precision model's next-token distributions and the
low-RAM (streamed + quantized) model's distributions over the **same held-out calibration
batch**, it computes the degradation — mean output KL(full ‖ lowram) and top-1 agreement —
and decides pass/fail against an explicit budget, with a **protected-behavior floor** that a
configurable set of "must-not-regress" prompts has to clear regardless of the aggregate. The
gate fails closed: missing data, shape mismatch, or a protected regression all reject.

This mirrors the promotion discipline the training side already uses
(``tools/promote_adapter.py`` / ``eval_ladder.py``): a candidate is promoted only when a
bounded-degradation contract is *measured*, never asserted. Here the "candidate" is a
memory-saving deployment of an existing model, and "no overclaim" means the bytes you saved
did not silently cost quality you didn't measure.

*Model-agnostic by construction.* It consumes probability arrays the caller produced (from
whatever real engine), so it is pure-numpy and CI-testable on synthetic distributions; the
caller owns the actual full-precision and streamed forward passes.

Falsifiable offline invariants (``offline_invariants()``, CI-gated):
  - an identical model (lowram == full) passes with ~0 KL and 100% agreement;
  - a badly-degraded lowram model is rejected;
  - the protected floor can reject a candidate the aggregate would have passed;
  - byte savings are reported honestly alongside the quality verdict (never one without
    the other).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


@dataclass
class LowRamReport:
    """The verdict: quality degradation *and* the memory it bought, together."""

    passed: bool
    mean_kl: float
    top1_agreement: float
    protected_max_kl: float
    protected_min_agreement: float
    mem_ratio: float                 # fast-memory footprint reduction vs fp16 (e.g. 3.56)
    n_eval: int
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "mean_kl": round(self.mean_kl, 6),
            "top1_agreement": round(self.top1_agreement, 6),
            "protected_max_kl": round(self.protected_max_kl, 6),
            "protected_min_agreement": round(self.protected_min_agreement, 6),
            "mem_ratio": round(self.mem_ratio, 4),
            "n_eval": self.n_eval,
            "reasons": self.reasons,
        }


@dataclass
class LowRamGate:
    """The no-overclaim contract for a low-RAM serving deployment.

    ``max_mean_kl``           : aggregate ceiling on mean output KL(full ‖ lowram).
    ``min_top1_agreement``    : aggregate floor on argmax agreement (next-token match rate).
    ``protected_max_kl``      : per-example KL ceiling on the protected slice (stricter).
    ``protected_min_agreement``: top-1 floor on the protected slice.
    """

    max_mean_kl: float = 0.05
    min_top1_agreement: float = 0.97
    protected_max_kl: float = 0.10
    protected_min_agreement: float = 0.95

    def evaluate(self, full_probs, lowram_probs, *,
                 protected_mask: "Optional[list[bool]]" = None,
                 mem_ratio: float = 1.0) -> LowRamReport:
        """Score ``lowram_probs`` against ``full_probs`` and apply the contract.

        ``full_probs`` / ``lowram_probs`` : (N, V) next-token softmax distributions over the
            same N held-out positions (rows must align). ``protected_mask`` : optional length-N
            booleans marking the must-not-regress slice. ``mem_ratio`` : the measured fast-memory
            reduction this deployment achieves vs fp16 (reported, not gated — quality is the gate).
        """
        if not _HAVE_NUMPY:
            raise RuntimeError("numpy required")
        full = np.asarray(full_probs, dtype=np.float64)
        low = np.asarray(lowram_probs, dtype=np.float64)
        reasons: list[str] = []
        if full.ndim != 2 or low.ndim != 2:
            return LowRamReport(False, float("inf"), 0.0, float("inf"), 0.0, mem_ratio, 0,
                                ["probs must be 2-D (N, V)"])
        if full.shape != low.shape:
            return LowRamReport(False, float("inf"), 0.0, float("inf"), 0.0, mem_ratio, 0,
                                [f"shape mismatch full{full.shape} vs lowram{low.shape}"])
        n = full.shape[0]
        if n == 0:
            return LowRamReport(False, float("inf"), 0.0, float("inf"), 0.0, mem_ratio, 0,
                                ["no evaluation positions"])

        kl = _row_kl(full, low)                       # (N,) per-position KL(full ‖ lowram)
        agree = (full.argmax(1) == low.argmax(1))     # (N,) top-1 match
        mean_kl = float(kl.mean())
        top1 = float(agree.mean())

        # Protected slice (defaults to the whole set if no mask given).
        if protected_mask is None:
            pmask = np.ones(n, dtype=bool)
        else:
            pmask = np.asarray(protected_mask, dtype=bool)
            if pmask.shape[0] != n:
                return LowRamReport(False, mean_kl, top1, float("inf"), 0.0, mem_ratio, n,
                                    ["protected_mask length != n"])
        if pmask.any():
            prot_max_kl = float(kl[pmask].max())
            prot_min_agree = float(agree[pmask].mean())
        else:
            prot_max_kl, prot_min_agree = 0.0, 1.0

        if mean_kl > self.max_mean_kl:
            reasons.append(f"mean_kl {mean_kl:.4f} > {self.max_mean_kl}")
        if top1 < self.min_top1_agreement:
            reasons.append(f"top1_agreement {top1:.4f} < {self.min_top1_agreement}")
        if prot_max_kl > self.protected_max_kl:
            reasons.append(f"protected_max_kl {prot_max_kl:.4f} > {self.protected_max_kl}")
        if prot_min_agree < self.protected_min_agreement:
            reasons.append(f"protected_min_agreement {prot_min_agree:.4f} < {self.protected_min_agreement}")

        return LowRamReport(
            passed=not reasons,
            mean_kl=mean_kl, top1_agreement=top1,
            protected_max_kl=prot_max_kl, protected_min_agreement=prot_min_agree,
            mem_ratio=mem_ratio, n_eval=n, reasons=reasons,
        )


def _row_kl(p, q, *, eps: float = 1e-12):
    """Per-row KL(p ‖ q) over a batch of distributions; clamps for numerical safety."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(1, keepdims=True)
    q = q / q.sum(1, keepdims=True)
    return np.sum(p * np.log(p / q), axis=1)


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
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    N, V = 64, 50
    logits = rng.standard_normal((N, V)) * 3.0
    full = softmax(logits)
    gate = LowRamGate()

    # 1. Identical model passes with ~0 KL and 100% agreement.
    rep = gate.evaluate(full, full.copy(), mem_ratio=3.56)
    checks["identical_passes"] = rep.passed and rep.mean_kl < 1e-9 and rep.top1_agreement == 1.0
    detail["identical"] = rep.as_dict()

    # 2. A tiny quantization-like perturbation (preserves argmax) passes under budget.
    low_ok = softmax(logits + rng.standard_normal((N, V)) * 0.02)
    rep_ok = gate.evaluate(full, low_ok, mem_ratio=3.56)
    checks["small_perturb_passes"] = rep_ok.passed
    detail["small_perturb"] = rep_ok.as_dict()

    # 3. A badly-degraded model (heavy noise, argmax often flips) is rejected.
    low_bad = softmax(logits + rng.standard_normal((N, V)) * 6.0)
    rep_bad = gate.evaluate(full, low_bad, mem_ratio=3.56)
    checks["bad_model_rejected"] = not rep_bad.passed
    detail["bad_model"] = rep_bad.as_dict()

    # 4. Protected floor rejects a candidate the aggregate would have passed: make ONE
    #    protected position regress hard while the bulk stays clean.
    low_mixed = full.copy()
    bad_row = softmax((logits[:1] + rng.standard_normal((1, V)) * 8.0))
    low_mixed[0] = bad_row[0]
    mask = [i == 0 for i in range(N)]
    strict = LowRamGate(max_mean_kl=1.0, min_top1_agreement=0.0,   # aggregate would pass
                        protected_max_kl=0.10, protected_min_agreement=0.95)
    rep_prot = strict.evaluate(full, low_mixed, protected_mask=mask, mem_ratio=3.56)
    aggregate_would_pass = (rep_prot.mean_kl <= 1.0 and rep_prot.top1_agreement >= 0.0)
    checks["protected_floor_bites"] = aggregate_would_pass and not rep_prot.passed
    detail["protected"] = rep_prot.as_dict()

    # 5. Fail-closed on shape mismatch and empty input.
    rep_mm = gate.evaluate(full, full[:, :V - 1])
    checks["shape_mismatch_rejected"] = not rep_mm.passed
    rep_empty = gate.evaluate(np.zeros((0, V)), np.zeros((0, V)))
    checks["empty_rejected"] = not rep_empty.passed

    # 6. The report carries the memory saving alongside the verdict (never quality alone).
    checks["reports_mem_ratio"] = rep_ok.as_dict()["mem_ratio"] == round(3.56, 4)

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Low-RAM eval offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  identical: {detail.get('identical')}")
    print(f"  bad_model: {detail.get('bad_model')}")
    raise SystemExit(0 if ok else 1)
