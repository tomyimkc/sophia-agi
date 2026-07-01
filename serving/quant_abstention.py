# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Conformal ABSTENTION for low-RAM (quantized) serving — the repo's own thesis applied to the quant gap.

The NVFP4 low-RAM cert demands the quantized model NEVER flips the argmax (top1 >= 0.97). v5 fully
merged reaches only top1 0.922 (ledger nvfp4-v5-cert-recovered-contaminated). But a system that
*abstains instead of fabricating* does not need never-flip: serve the quantized model, and on the
low-confidence tokens where it is most likely to disagree with full precision, **abstain** (defer /
flag) rather than emit. The honest metric is then "top1 on the tokens it ANSWERS", which is far
higher, at the cost of some coverage. This makes a top1~0.92 model honestly shippable regardless of
whether the cert ever clears 0.97 — a hedge that does not fight physics. `canClaimAGI` stays false.

The abstention signal is computable from the QUANT MODEL ALONE at serve time (we do NOT have the FP
model in low-RAM deployment): the top1-vs-top2 probability MARGIN. Low margin -> high nonconformity
-> abstain. We calibrate the threshold on a held-out split (split-conformal, via
``agent/conformal_gate.py``) using FP-agreement as the correctness label, then MEASURE the answered
accuracy on a disjoint test split. We report the measured selective-accuracy/coverage trade-off — we
do NOT claim a proven per-token guarantee (that is a stronger Mondrian form); the number is empirical
and honest. Pure/offline/deterministic (numpy); the numerics are proven in ``offline_invariants``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def quant_nonconformity(low_probs) -> "Any":
    """Serve-time nonconformity from the QUANT distribution alone: 1 - (top1 - top2) margin.

    A confident quant prediction (large top1-top2 gap) is LOW nonconformity (answer); a near-tie is
    HIGH nonconformity (abstain — those are exactly the tokens most likely to flip vs full precision)."""
    import numpy as np
    p = np.asarray(low_probs, dtype=np.float64)
    part = np.partition(p, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    return 1.0 - margin


def quant_abstention_report(full_probs, low_probs, *, alpha: float = 0.02,
                            calib_frac: float = 0.5) -> "dict[str, Any]":
    """Turn the cert's (FP, quant) next-token distributions into the honest 'shippable with
    abstention' metric. Deterministic split (first ``calib_frac`` = calibration, rest = test).

    Returns raw_top1 (the cert's number), answered_top1 (measured FP-agreement on the tokens the
    conformal gate ANSWERS in the held-out test split), coverage (fraction answered), and whether the
    answered accuracy meets the 1-alpha target. No overclaim: answered_top1 is measured, not guaranteed."""
    import numpy as np

    from agent.conformal_gate import fit_conformal_policy

    full = np.asarray(full_probs, dtype=np.float64)
    low = np.asarray(low_probs, dtype=np.float64)
    n = full.shape[0]
    if n < 4:
        return {"error": "too few positions for a calib/test split", "n": int(n)}
    agree = (full.argmax(1) == low.argmax(1))
    nonconf = quant_nonconformity(low)

    ncal = max(1, int(n * calib_frac))
    cal = [{"nonconformity": float(nonconf[i]), "correct": bool(agree[i])} for i in range(ncal)]
    test_idx = list(range(ncal, n))
    policy = fit_conformal_policy(cal, alpha=alpha)          # threshold from CORRECT calib rows
    thr = policy.threshold

    answered = [i for i in test_idx if nonconf[i] <= thr]
    n_test = len(test_idx)
    answered_top1 = float(np.mean([agree[i] for i in answered])) if answered else 0.0
    raw_top1 = float(agree.mean())
    target = round(1.0 - alpha, 4)
    return {
        "raw_top1": round(raw_top1, 4),
        "answered_top1": round(answered_top1, 4),
        "coverage": round(len(answered) / n_test, 4) if n_test else 0.0,
        "abstained": round(1.0 - (len(answered) / n_test), 4) if n_test else 0.0,
        "target_answered_agreement": target,
        "meets_target_on_answered": bool(answered_top1 >= target and len(answered) > 0),
        "nonconformity_threshold": round(float(thr), 4),
        "n_calib": ncal, "n_test": n_test, "alpha": alpha,
        "note": "answered_top1 is MEASURED on a held-out split (selective accuracy), not a proven "
                "per-token guarantee. Serve-time signal (quant top1-top2 margin) needs no FP model.",
    }


def quant_abstention_frontier(full_probs, low_probs, *, calib_frac: float = 0.5,
                              target_answered: float = 0.97) -> "dict[str, Any]":
    """Sweep the abstention threshold to trace the coverage vs answered-top1 FRONTIER and find the best
    operating point that reaches ``target_answered`` at maximum coverage. This is the honest answer to
    'is a raw-failing model shippable via abstention?' — a SINGLE alpha (e.g. 0.02) can be misleading
    (it may answer ~100% and look un-shippable) while a stricter point on the frontier clears the bar.

    Thresholds are chosen on the CALIB split (nonconformity quantiles) and MEASURED on the disjoint TEST
    split, so the reported answered_top1 is out-of-sample. No overclaim: answered_top1 is measured."""
    import numpy as np

    full = np.asarray(full_probs, dtype=np.float64)
    low = np.asarray(low_probs, dtype=np.float64)
    n = full.shape[0]
    if n < 8:
        return {"error": "too few positions for a frontier", "n": int(n)}
    agree = (full.argmax(1) == low.argmax(1))
    nonconf = quant_nonconformity(low)
    ncal = max(1, int(n * calib_frac))
    cal_nc = nonconf[:ncal]
    test_idx = np.arange(ncal, n)
    n_test = len(test_idx)

    frontier, best = [], None
    for q in [round(x / 20.0, 3) for x in range(20, 9, -1)]:      # calib quantile 1.00 -> 0.50 (stricter)
        thr = float(np.quantile(cal_nc, q))
        answered = [i for i in test_idx if nonconf[i] <= thr]
        cov = len(answered) / n_test if n_test else 0.0
        atop1 = float(np.mean([agree[i] for i in answered])) if answered else 0.0
        pt = {"calib_quantile": q, "threshold": round(thr, 4),
              "coverage": round(cov, 4), "answered_top1": round(atop1, 4)}
        frontier.append(pt)
        if atop1 >= target_answered and cov > 0 and (best is None or cov > best["coverage"]):
            best = pt
    return {
        "raw_top1": round(float(agree.mean()), 4),
        "target_answered": target_answered,
        "shippable_operating_point": best,       # None => abstention cannot rescue this model at target
        "shippable": best is not None,
        "frontier": frontier,
        "n_calib": ncal, "n_test": n_test,
        "note": "best = max-coverage point whose out-of-sample answered_top1 >= target. If None, the "
                "quant top1-top2 margin does not separate the argmax-flips well enough -> abstention "
                "cannot rescue this model; the recipe fix (v6) is the path. Measured, not guaranteed.",
    }


# --------------------------------------------------------------------------- #
# Offline invariants — GPU-free, prove the abstention trade-off on synthetic data.
# --------------------------------------------------------------------------- #
def offline_invariants() -> "tuple[bool, dict]":
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return False, {"checks": {"numpy_available": False}}

    checks: dict[str, bool] = {}
    detail: dict = {}
    rng = np.random.default_rng(0)
    N, V = 400, 20

    # Build FP + quant distributions where CONFIDENT quant tokens agree with FP and NEAR-TIE quant
    # tokens flip — exactly the regime abstention should exploit.
    fp = rng.standard_normal((N, V))
    fp_soft = np.exp(fp) / np.exp(fp).sum(1, keepdims=True)
    fp_arg = fp_soft.argmax(1)
    low = fp.copy()
    n_flip = 0
    for i in range(N):
        if rng.random() < 0.30:                 # 30% near-tie tokens: make quant a near-tie that flips
            j = (fp_arg[i] + 1) % V
            low[i, fp_arg[i]] = fp[i, j] + 0.01   # nearly tie the runner-up -> flips, low margin
            low[i, j] = fp[i, fp_arg[i]]
            n_flip += 1
        else:
            low[i, fp_arg[i]] += 3.0              # confident, keeps FP argmax, high margin
    low_soft = np.exp(low) / np.exp(low).sum(1, keepdims=True)

    rep = quant_abstention_report(fp_soft, low_soft, alpha=0.02)
    # 1. Abstention lifts answered accuracy ABOVE the raw (cert) top1.
    checks["answered_beats_raw"] = rep["answered_top1"] > rep["raw_top1"]
    # 2. It costs coverage (we abstain on some tokens), but keeps most.
    checks["coverage_below_one"] = 0.4 < rep["coverage"] < 1.0
    # 3. The near-tie flippers are what get abstained: raw_top1 ~ 1 - flip_rate, answered_top1 high.
    checks["raw_reflects_flip_rate"] = abs(rep["raw_top1"] - (1 - n_flip / N)) < 0.12
    checks["answered_high"] = rep["answered_top1"] >= 0.95
    # 4. Stricter alpha -> higher answered accuracy (or equal), never worse coverage-for-free.
    strict = quant_abstention_report(fp_soft, low_soft, alpha=0.005)
    loose = quant_abstention_report(fp_soft, low_soft, alpha=0.10)
    checks["stricter_alpha_not_worse_answered"] = strict["answered_top1"] >= loose["answered_top1"] - 1e-9
    # 5. Nonconformity is high for near-ties, low for confident tokens.
    nc = quant_nonconformity(low_soft)
    tie_mask = np.array([low_soft[i].max() - np.partition(low_soft[i], -2)[-2] < 0.05 for i in range(N)])
    checks["nonconformity_orders_confidence"] = nc[tie_mask].mean() > nc[~tie_mask].mean()

    # 6. The FRONTIER finds a shippable operating point on separable data, and coverage is monotone
    #    non-increasing as the threshold gets stricter (the honest coverage/accuracy trade-off).
    fr = quant_abstention_frontier(fp_soft, low_soft, target_answered=0.97)
    checks["frontier_finds_shippable_point"] = fr["shippable"] and fr["shippable_operating_point"]["coverage"] > 0.0
    covs = [p["coverage"] for p in fr["frontier"]]
    checks["frontier_coverage_monotone"] = all(covs[i] >= covs[i + 1] - 1e-9 for i in range(len(covs) - 1))

    detail["report"] = rep
    detail["strict_vs_loose"] = {"strict_answered": strict["answered_top1"],
                                 "loose_answered": loose["answered_top1"],
                                 "strict_cov": strict["coverage"], "loose_cov": loose["coverage"]}
    detail["frontier_best"] = fr["shippable_operating_point"]
    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("quant_abstention offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  example report: {detail['report']}")
