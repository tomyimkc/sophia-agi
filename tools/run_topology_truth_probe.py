#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""CONTRARIAN probe — does evidence TOPOLOGY predict TRUTH?

The Sophia OKF stack rests on an implicit *identity axiom*: that a confidence value derived
from the STRUCTURE of evidence (how many independent sources, how diverse, how recent,
whether replicated / consensus-endorsed, how contradicted) tracks whether a claim is
actually TRUE. This tool tries to FALSIFY that axiom on an externally-labeled seed set.

For each labeled claim it computes `topology_confidence` from ONLY reproducible evidence
features (a fixed, transparent monotone formula — NOT the truth bit), then measures
Spearman rho(topology_confidence, revealed_truth) with a deterministic permutation p-value
(tools/stats_ext.py). It reports whether the seed set is even POWERED to resolve the
pre-registered effect (tools/eval_stats.py, per agi-proof/measurement-thesis.md).

Pre-registered NO-GO (agi-proof/topology-truth-axiom/measurement_spec.json):
    rho <= 0  FALSIFIES the identity axiom. That would mean topology anti-correlates with
    (or is blind to) truth, and the right response is to SHIP AN EMPIRICAL topology->truth
    CALIBRATION LAYER rather than assume the identity holds.

IMPORTANT — NO-OVERCLAIM: the committed seed set is small (~20-30) and UNDERPOWERED, so any
rho it produces is ILLUSTRATIVE, not a validated result. When the probe is underpowered the
verdict is "UNDERPOWERED" (exit 3) regardless of the point estimate — a positive rho on an
underpowered seed set does NOT confirm the axiom, and a non-positive one does NOT yet
falsify it. A powered falsification/confirmation needs a larger externally-labeled set
(requiredN in the spec). status stays preregistration_only; canClaimAGI:false.

Exit codes:
  0  GO           — powered AND rho > 0 with p <= alpha (axiom SURVIVES this test).
  3  NO-GO/UNDER  — powered AND rho <= 0 (axiom FALSIFIED)  OR  underpowered (illustrative).
  2  unreadable/missing input.

Prints a JSON receipt to stdout; human prose to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import stats_ext  # noqa: E402
import eval_stats  # noqa: E402

# Reproducible, transparent weights for the topology->confidence map. These are a FIXED
# monotone prior over evidence structure; they are NOT fit to the truth labels (fitting to
# labels would make the probe circular). Higher independent/diverse/replicated/endorsed/
# recent evidence and FEWER contradictions -> higher topology confidence.
_WEIGHTS = {
    "independent_sources": 0.10,      # capped contribution (see _clip below)
    "source_type_diversity": 0.08,
    "has_replication": 0.15,
    "consensus_body_endorses": 0.20,
    "recency_score": 0.05,
    "contradicting_sources": -0.12,   # penalty per credible contradiction
}
_REQUIRED_FEATURES = tuple(_WEIGHTS.keys())


def topology_confidence(features: dict[str, float]) -> float:
    """Map an evidence-topology feature dict to a confidence in [0, 1] via a fixed monotone
    formula (logistic squash of a weighted sum). Deterministic and label-free."""
    missing = [k for k in _REQUIRED_FEATURES if k not in features]
    if missing:
        raise ValueError(f"features missing keys: {missing}")
    # mild caps so a single huge count cannot dominate (diminishing returns)
    isrc = min(float(features["independent_sources"]), 6.0)
    div = min(float(features["source_type_diversity"]), 6.0)
    rep = 1.0 if float(features["has_replication"]) else 0.0
    end = 1.0 if float(features["consensus_body_endorses"]) else 0.0
    rec = min(float(features["recency_score"]), 3.0)
    con = float(features["contradicting_sources"])
    score = (
        _WEIGHTS["independent_sources"] * isrc
        + _WEIGHTS["source_type_diversity"] * div
        + _WEIGHTS["has_replication"] * rep
        + _WEIGHTS["consensus_body_endorses"] * end
        + _WEIGHTS["recency_score"] * rec
        + _WEIGHTS["contradicting_sources"] * con
    )
    # center so an "empty" topology maps near 0.5, then logistic squash to (0,1)
    import math
    return 1.0 / (1.0 + math.exp(-(score - 0.5)))


def load_labeled(path: Path) -> list[dict[str, Any]]:
    """Load the JSONL seed set, skipping any `_meta` header record(s)."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                continue
            for req in ("claimId", "truth", "features"):
                if req not in obj:
                    raise ValueError(f"line {ln}: labeled row missing '{req}'")
            if obj["truth"] not in (0, 1):
                raise ValueError(f"line {ln}: truth must be 0 or 1, got {obj['truth']!r}")
            rows.append(obj)
    return rows


def run_probe(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    iters: int,
    seed: int,
    mde: float,
) -> dict[str, Any]:
    """Compute topology confidences, Spearman rho vs truth, permutation p, and power."""
    conf = [topology_confidence(r["features"]) for r in rows]
    truth = [float(r["truth"]) for r in rows]
    n = len(rows)

    perm = stats_ext.permutation_pvalue(
        conf, truth, iters=iters, seed=seed, alternative="greater"
    )
    rho = perm["rho"]
    p = perm["p"]

    # Power: is this seed set even able to resolve the pre-registered MDE? A binary-outcome
    # power proxy at the seed N (measurement-thesis Pillar 2). Underpowered => illustrative.
    required_n = eval_stats.required_n_for_mde(mde)
    mde_here = eval_stats.mde_at_n(n) if n > 0 else float("inf")
    powered = n >= required_n

    # Pre-registered verdict logic.
    if not powered:
        verdict = "UNDERPOWERED"
        axiom = "not-yet-testable"
        go = False
        exit_code = 3
    elif rho > 0.0 and p <= alpha:
        verdict = "GO"
        axiom = "survives"
        go = True
        exit_code = 0
    else:
        # powered but rho <= 0 (or not significant in the axiom's direction) => FALSIFIED
        verdict = "NO-GO"
        axiom = "falsified"
        go = False
        exit_code = 3

    return {
        "experimentId": "topology-truth-axiom",
        "status": "preregistration_only",
        "canClaimAGI": False,
        "go": go,
        "verdict": verdict,
        "axiomVerdict": axiom,
        "primaryMetric": "spearman_rho(topology_confidence, revealed_truth)",
        "rho": rho,
        "p": p,
        "alpha": alpha,
        "n": n,
        "requiredN": required_n,
        "mdePreRegistered": mde,
        "mdeAtThisN": mde_here,
        "powered": powered,
        "permutation": {
            "iters": perm["iters"],
            "seed": perm["seed"],
            "alternative": perm["alternative"],
            "count_as_extreme": perm["count_as_extreme"],
        },
        "noGoRule": "rho <= 0 FALSIFIES the identity axiom (=> ship an empirical topology->truth calibration layer instead of assuming identity)",
        "honestBound": "Seed set is small and UNDERPOWERED; any rho here is ILLUSTRATIVE, not validated. A powered test needs a larger externally-labeled set (requiredN).",
        "perClaim": [
            {"claimId": r["claimId"], "truth": r["truth"], "topologyConfidence": c}
            for r, c in zip(rows, conf)
        ],
        "exitCode": exit_code,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--labeled",
        default=str(_ROOT / "agi-proof" / "topology-truth-axiom" / "labeled_set.jsonl"),
        help="path to the labeled JSONL seed set",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="significance level")
    ap.add_argument("--iters", type=int, default=10000, help="permutation iterations")
    ap.add_argument("--seed", type=int, default=0, help="deterministic permutation seed")
    ap.add_argument(
        "--mde",
        type=float,
        default=0.30,
        help="pre-registered minimum detectable effect for the power check",
    )
    args = ap.parse_args(argv)

    path = Path(args.labeled)
    try:
        rows = load_labeled(path)
    except FileNotFoundError:
        print(f"[topology-truth-probe] cannot read labeled set: {path}", file=sys.stderr)
        return 2
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[topology-truth-probe] malformed labeled set: {exc}", file=sys.stderr)
        return 2

    if not rows:
        print("[topology-truth-probe] no labeled rows found", file=sys.stderr)
        return 2

    receipt = run_probe(
        rows, alpha=args.alpha, iters=args.iters, seed=args.seed, mde=args.mde
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))

    v = receipt["verdict"]
    print(
        f"[topology-truth-probe] verdict={v} axiom={receipt['axiomVerdict']} "
        f"rho={receipt['rho']:.4f} p={receipt['p']:.4f} n={receipt['n']} "
        f"requiredN={receipt['requiredN']} powered={receipt['powered']}",
        file=sys.stderr,
    )
    if v == "UNDERPOWERED":
        print(
            "[topology-truth-probe] ILLUSTRATIVE ONLY — underpowered seed set cannot "
            "validate OR falsify the axiom. Grow the externally-labeled set to requiredN.",
            file=sys.stderr,
        )
    elif v == "NO-GO":
        print(
            "[topology-truth-probe] AXIOM FALSIFIED at power — ship an empirical "
            "topology->truth calibration layer instead of assuming identity.",
            file=sys.stderr,
        )
    return int(receipt["exitCode"])


if __name__ == "__main__":
    raise SystemExit(main())
