#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""O4 — convergence-driven adaptive test-time compute for self-consistency.

Inspiration: Neural ODEs / DEQ / CTM / EBT all allocate compute by watching an internal
iteration converge — spend more steps on hard inputs, stop early on easy ones, and use
the convergence gap as a confidence proxy. EBT shows the benefit of 'thinking' scales
with distribution shift.

sophia does fixed-k self-consistency. This tool makes k ADAPTIVE: draw samples one at a
time, recompute the Kuramoto order parameter r over the samples so far, and STOP once r
has stabilized (|r_k - r_{k-1}| < eps for `patience` draws) or a min-r 'clearly coherent'
bar is met — capped at k_max (and, in production, by the long-horizon deadline). Persistent
non-convergence at k_max is itself an ABSTAIN signal, not a reason to keep spending.

Two things it measures on labeled data:
  1. cost: mean samples used (adaptive) vs k_max (fixed) — the compute saved.
  2. quality: does adaptive-k keep the same selective-prediction quality as fixed-k?
     (final r as confidence, audited by agent.calibration.calibration_report.)
The honest win condition is 'fewer samples at no AURC cost' — reported, never assumed.

This tool takes PRE-SAMPLED answers offline (so it is deterministic and backend-free): each
record supplies the full ordered list of samples the model produced; the tool replays the
adaptive stopping rule over that list. In production the same rule wraps the live sampler
inside agent.long_horizon (bounded by its cooperative deadline). It updates no weights.

Input JSONL:
  {"samples": ["a","a","b","a", ...], "correct": true}   # ordered as drawn; >=2 samples
Output: per-record stop point + aggregate cost/quality vs the fixed-k baseline.
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

_IMPORT_ERR = ""
try:
    import oscillator_core as oc
    from agent.calibration import calibration_report
    from agent.selective_risk import aurc, paired_aurc_delta_ci
    _REPO_OK = True
except Exception as e:  # pragma: no cover
    _REPO_OK = False
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def _env_artifact(reason: str) -> dict[str, Any]:
    return {
        "schema": "sophia.adaptive_compute.v1",
        "environmentArtifact": True, "ok": False, "reason": reason,
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
    }


def adaptive_stop(samples: list[str], *, dim: int, k_min: int = 2, k_max: int | None = None,
                  eps: float = 0.02, patience: int = 2, r_confident: float = 0.9,
                  steps: int = 30, gain: float = 2.0, seed: int = 0) -> dict[str, Any]:
    """Replay the adaptive stopping rule over an ordered sample list.

    Returns {kUsed, rFinal, converged, reason}. Stops when r stabilizes for `patience`
    consecutive draws, or r crosses r_confident, whichever first; never before k_min;
    never past k_max (defaults to all available samples).
    """
    n = len(samples)
    kmax = n if k_max is None else min(k_max, n)
    prev_r = None
    stable = 0
    r = 0.0
    for k in range(1, kmax + 1):
        r = oc.consensus_r(samples[:k], dim=dim, steps=steps, gain=gain, seed=seed)
        if k < k_min:
            prev_r = r
            continue
        if r >= r_confident:
            return {"kUsed": k, "rFinal": round(r, 4), "converged": True, "reason": "confident"}
        if prev_r is not None and abs(r - prev_r) < eps:
            stable += 1
            if stable >= patience:
                return {"kUsed": k, "rFinal": round(r, 4), "converged": True, "reason": "stable"}
        else:
            stable = 0
        prev_r = r
    # hit the cap without stabilizing -> non-convergence is itself an abstain signal
    return {"kUsed": kmax, "rFinal": round(r, 4), "converged": False, "reason": "k_max_no_convergence"}


def run(records: list[dict[str, Any]], *, dim: int = oc.EMBED_DIM_DEFAULT, k_min: int = 2,
        k_max: int | None = None, eps: float = 0.02, patience: int = 2,
        r_confident: float = 0.9, coverage: float = 0.5, n_boot: int = 5000,
        seed: int = 0) -> dict[str, Any]:
    if not _REPO_OK:
        return _env_artifact(f"repo instruments unavailable ({_IMPORT_ERR}); run with PYTHONPATH=.")
    rows = [r for r in records if len(r.get("samples", [])) >= 2]
    if not rows:
        return _env_artifact("no records with >=2 samples (fail-closed; nothing to adapt)")

    fixed_k = max(len(r["samples"]) for r in rows) if k_max is None else k_max
    per, adaptive_conf, fixed_conf, correct, ks = [], [], [], [], []
    for r in rows:
        samples = [str(s) for s in r["samples"]]
        stop = adaptive_stop(samples, dim=dim, k_min=k_min, k_max=k_max, eps=eps,
                             patience=patience, r_confident=r_confident, seed=seed)
        r_fixed = oc.consensus_r(samples[:fixed_k], dim=dim, seed=seed)
        c = bool(r.get("correct", False))
        per.append({"kUsed": stop["kUsed"], "rFinal": stop["rFinal"],
                    "converged": stop["converged"], "reason": stop["reason"], "correct": c})
        adaptive_conf.append(stop["rFinal"]); fixed_conf.append(r_fixed)
        correct.append(c); ks.append(stop["kUsed"])

    out: dict[str, Any] = {
        "schema": "sophia.adaptive_compute.v1",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "n": len(rows), "fixedK": fixed_k,
        "hashEmbedSeam": True,
        "cost": {
            "meanSamplesAdaptive": round(sum(ks) / len(ks), 3),
            "meanSamplesFixed": float(fixed_k),
            "samplesSavedFrac": round(1.0 - (sum(ks) / len(ks)) / fixed_k, 4) if fixed_k else 0.0,
            "nonConvergedAbstainRate": round(sum(1 for p in per if not p["converged"]) / len(per), 4),
        },
        "records": per,
    }
    if len(set(correct)) == 2:
        # quality: adaptive must not cost AURC vs fixed-k. delta = adaptive - fixed;
        # CI upper bound <= 0 would mean adaptive strictly better; we want CI to CONTAIN 0
        # (no quality loss) while saving samples.
        fab = [0 if c else 1 for c in correct]
        a_items = list(zip(adaptive_conf, fab)); f_items = list(zip(fixed_conf, fab))
        a_aurc, f_aurc = aurc(a_items), aurc(f_items)
        ci = paired_aurc_delta_ci(a_items, f_items, n_boot=n_boot, seed=seed)
        out["quality"] = {
            "adaptiveGate": calibration_report(adaptive_conf, correct, coverage=coverage),
            "fixedGate": calibration_report(fixed_conf, correct, coverage=coverage),
            "aurcAdaptive": round(a_aurc, 6), "aurcFixed": round(f_aurc, 6),
            "aurcDeltaCi95": [round(ci[0], 6), round(ci[1], 6)],
            "noQualityLoss": ci[0] <= 0.0 <= ci[1] or ci[1] <= 0.0,  # contains 0, or adaptive better
        }
    return out


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--records", required=True, help="JSONL of {samples:[...ordered...], correct:bool}")
    p.add_argument("--output", default=None)
    p.add_argument("--dim", type=int, default=oc.EMBED_DIM_DEFAULT)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--r-confident", type=float, default=0.9)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = _load_jsonl(args.records)
    except Exception as e:
        report = _env_artifact(f"could not read --records ({type(e).__name__}: {e})")
    else:
        report = run(records, dim=args.dim, k_min=args.k_min, k_max=args.k_max, eps=args.eps,
                     patience=args.patience, r_confident=args.r_confident, n_boot=args.n_boot,
                     seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not report.get("environmentArtifact") else 2


if __name__ == "__main__":
    raise SystemExit(main())