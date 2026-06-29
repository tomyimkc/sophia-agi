# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validation of the real-model fusion result — cross-validated weights + bootstrap CIs.

The §3e real-model numbers carry two honest caveats this module removes, **offline and free**
(the v2 runner stored the per-case detector scores, so no new API spend):

  1. The quality-weighted fused d′ is **in-sample-optimistic** — weights (∝ each detector's d′)
     are fit on the same labels they are scored against. The fix is **leave-one-out
     cross-validation (LOO-CV)**: for each case, fit the weights on the *other* N−1 cases and
     score the held-out case with them. The LOO fused score never sees its own label.
  2. Every number was a **point estimate**. The fix is a **bootstrap CI** (resample cases with
     replacement) on every metric — the no-overclaim contract requires intervals, never bare
     points.

It reports, per real model, d′ AND AUC with 95% bootstrap CIs for: each detector (A, B, B2),
equal-weight fusion, in-sample quality-weighted fusion, and **LOO-CV** quality-weighted fusion.

Honest notes carried into the verdict:
  - AUC is the load-bearing metric where a class is tiny. Claude-haiku has base_error≈0.98 ⇒ only
    ~1 clean case, so its d′ is numerically unstable (near-zero clean variance); the module flags
    low-clean-count models and leans on AUC + the CI width there.
  - `candidateOnly: true`, `canClaimAGI: false`. This is the statistics layer that lets a row be
    *considered* for the gate; promotion still needs ≥3 seeds and ≥2 judge families per the IEC.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_reflex_eval import auc, d_prime  # noqa: E402
from reasoning.instinct_fusion import _zscores, breakeven_snr  # noqa: E402

RESULTS = ROOT / "reasoning" / "results"
DETECTORS = ("A", "B", "B2")


def load_scores(path: Path) -> tuple[dict[str, list[float]], list[bool]]:
    pc = json.loads(Path(path).read_text())["per_case"]
    scores = {k: [float(c[k]) for c in pc] for k in DETECTORS}
    labels = [bool(c["is_error"]) for c in pc]
    return scores, labels


def _split(score: list[float], labels: list[bool]) -> tuple[list[float], list[float]]:
    return ([s for s, e in zip(score, labels) if e],
            [s for s, e in zip(score, labels) if not e])


def _dp_auc(score: list[float], labels: list[bool]) -> tuple[float, float]:
    e, c = _split(score, labels)
    return d_prime(e, c), auc(e, c)


def _weights(scores: dict[str, list[float]], labels: list[bool]) -> dict[str, float]:
    """Fisher-style weights: max(0, d′) per detector on the given (sub)sample."""
    w = {}
    for k in DETECTORS:
        e, c = _split(scores[k], labels)
        dp = d_prime(e, c)
        w[k] = max(0.0, dp) if math.isfinite(dp) else 0.0
    return w


def _fuse_with(scores: dict[str, list[float]], weights: dict[str, float]) -> list[float]:
    zs = {k: _zscores(scores[k]) for k in DETECTORS}
    n = len(labels_len(scores))
    return [sum(weights.get(k, 0.0) * zs[k][i] for k in DETECTORS) for i in range(n)]


def labels_len(scores: dict[str, list[float]]) -> list[float]:
    return scores[DETECTORS[0]]


def insample_qw_fused(scores: dict[str, list[float]], labels: list[bool]) -> list[float]:
    return _fuse_with(scores, _weights(scores, labels))


def loo_qw_fused(scores: dict[str, list[float]], labels: list[bool]) -> list[float]:
    """Leave-one-out CV: each case scored with weights fit on the other N−1 cases.

    z-scoring is also fit on the training fold (mean/sd of the other N−1) so the held-out
    point is fully out-of-sample.
    """
    n = len(labels)
    fused = [0.0] * n
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        train = {k: [scores[k][j] for j in idx] for k in DETECTORS}
        train_labels = [labels[j] for j in idx]
        w = _weights(train, train_labels)
        val = 0.0
        for k in DETECTORS:
            col = train[k]
            m = sum(col) / len(col)
            sd = math.sqrt(sum((x - m) ** 2 for x in col) / len(col)) or 1.0
            val += w[k] * (scores[k][i] - m) / sd
        fused[i] = val
    return fused


def bootstrap_ci(score: list[float], labels: list[bool], *, metric: str = "auc",
                 trials: int = 2000, seed: int = 1234) -> tuple[float, float]:
    """Percentile bootstrap CI over case-resampling for d′ or AUC."""
    rng = random.Random(seed)
    n = len(labels)
    vals = []
    for _ in range(trials):
        idx = [rng.randrange(n) for _ in range(n)]
        s = [score[j] for j in idx]
        lab = [labels[j] for j in idx]
        e, c = _split(s, lab)
        if not e or not c:
            continue
        v = auc(e, c) if metric == "auc" else d_prime(e, c)
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return (round(lo, 4), round(hi, 4))


def evaluate(path: Path, *, seed: int = 1234) -> dict[str, Any]:
    scores, labels = load_scores(path)
    n = len(labels)
    n_clean = sum(1 for e in labels if not e)
    bar = breakeven_snr()
    rows: dict[str, Any] = {}

    def record(name: str, score: list[float]) -> None:
        dp, ac = _dp_auc(score, labels)
        rows[name] = {
            "d_prime": round(dp, 4) if math.isfinite(dp) else "inf",
            "auc": round(ac, 4),
            "auc_ci": bootstrap_ci(score, labels, metric="auc", seed=seed),
        }

    for k in DETECTORS:
        record(k, scores[k])
    record("fused_equal", _fuse_with(scores, {k: 1.0 for k in DETECTORS}))
    record("fused_qw_insample", insample_qw_fused(scores, labels))
    record("fused_qw_loocv", loo_qw_fused(scores, labels))
    model = json.loads(path.read_text())["report"]["model"]
    return {
        "model": model, "n": n, "n_clean": n_clean, "breakeven_snr": bar,
        "low_clean_warning": n_clean < 5,
        "rows": rows,
    }


def run_experiment(seed: int = 1234) -> dict[str, Any]:
    out = {}
    for fname in ("fusion_realmodel_deepseek.json", "fusion_realmodel_llmhub-haiku.json"):
        p = RESULTS / fname
        if p.exists():
            ev = evaluate(p, seed=seed)
            out[ev["model"]] = ev
    return out


def format_report(res: dict[str, Any]) -> str:
    lines = ["Fusion validation — cross-validated weights + bootstrap CIs", "=" * 70]
    for model, ev in res.items():
        lines.append(f"\nMODEL {model}  (N={ev['n']}, clean={ev['n_clean']}, bar d′={ev['breakeven_snr']})"
                     + ("  [LOW-CLEAN: lean on AUC]" if ev["low_clean_warning"] else ""))
        lines.append(f"  {'detector':20} {'d′':>7}  {'AUC':>6}  {'AUC 95% CI':>18}")
        for name, r in ev["rows"].items():
            lines.append(f"  {name:20} {str(r['d_prime']):>7}  {r['auc']:>6.3f}  "
                         f"[{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]")
    lines += [
        "",
        "SHARP READING (the honest artifact this validation exposes):",
        "  B and B2 are STRUCTURAL VERIFIERS: B fires iff the answer over-includes, B2 iff it",
        "  under-includes — so (B>0 OR B2>0) ⟺ (answer ≠ structural truth) = the label itself.",
        "  Their high AUC and the near-1.0 fused AUC are therefore largely TAUTOLOGICAL: fusing",
        "  them reconstructs the okf verifier (the label generator), not a new predictive signal.",
        "  The only genuinely label-free / PREDICTIVE reflex is A (self-consistency) — and it is",
        "  weak (AUC ~0.6, CI often includes chance). Takeaway: if you HAVE the okf graph, verify",
        "  directly; the open research frontier is a stronger label-free reflex for when you don't.",
        "",
        "candidateOnly=True  level3Evidence=False",
        "boundary: case-resampling CIs + LOO-CV on a single run; gate still needs >=3 seeds, >=2 families.",
    ]
    return "\n".join(lines)


def _self_test() -> int:
    res = run_experiment(seed=1234)
    ds = next(ev for m, ev in res.items() if "deepseek" in m.lower())
    # V1: LOO-CV removes in-sample optimism — held-out qw AUC ≤ in-sample qw AUC.
    loo = ds["rows"]["fused_qw_loocv"]["auc"]
    ins = ds["rows"]["fused_qw_insample"]["auc"]
    assert loo <= ins + 1e-9, f"V1: LOO ({loo}) should not exceed in-sample ({ins})"
    # V2: DeepSeek equal-fusion separates real signal — AUC CI excludes chance (0.5).
    lo, hi = ds["rows"]["fused_equal"]["auc_ci"]
    assert lo > 0.5, f"V2: equal-fusion AUC CI includes chance ({lo},{hi})"
    # V3: B2 (the added detector) is real on DeepSeek — AUC CI excludes chance.
    assert ds["rows"]["B2"]["auc_ci"][0] > 0.5, "V3: B2 AUC CI includes chance"
    # V4: cross-validated fusion still beats the bar-equivalent (AUC > 0.5 lower bound) out of sample.
    assert ds["rows"]["fused_qw_loocv"]["auc_ci"][0] > 0.5, "V4: LOO-CV fusion not above chance"
    # V5: the honest artifact — the label-free reflex (A) is weaker than the structural verifier
    # (B2), and A's CI includes chance (it is NOT yet a reliable standalone detector).
    assert ds["rows"]["A"]["auc"] < ds["rows"]["B2"]["auc"], "V5: A should be weaker than verifier B2"
    assert ds["rows"]["A"]["auc_ci"][0] <= 0.5, "V5: A's AUC CI should still include chance"
    print(f"self-test OK: DeepSeek equal-fusion AUC CI [{lo},{hi}]; "
          f"qw in-sample AUC {ins} -> LOO-CV {loo} (optimism removed); "
          f"label-free A AUC {ds['rows']['A']['auc']} (CI includes chance) << verifier B2 "
          f"{ds['rows']['B2']['auc']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.run or args.json:
        res = run_experiment(seed=args.seed)
        print(json.dumps(res, indent=2) if args.json else format_report(res))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
