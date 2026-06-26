#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Interpretability M0 — offline SAE core (no torch / GPU / network).

Mirrors tools/run_steering.py and tools/run_rlvr.py: a deterministic ``--mode
mock`` / ``--dry-run`` path that trains a tiny pure-stdlib TopK SAE on planted
synthetic activations, computes the pre-registered SAE metrics, asserts the
offline invariants, and writes a public-aggregate report. This is the CI-green
floor of the interpretability workstream — the real Qwen2.5-7B harvest + SAELens
training (M1/M2) runs on RunPod / the DGX Spark behind requirements-interp.txt.

    python tools/run_interp.py --mode mock          # full offline invariants
    python tools/run_interp.py --dry-run            # alias for mock

Roadmap: docs/06-Roadmap/frontier-readiness/03-interpretability.md
Claim discipline: candidate-only, canClaimAGI=false; a failing invariant exits 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp import hookpoints  # noqa: E402
from interp.sae import metrics  # noqa: E402
from interp.sae.model import TopKSAE  # noqa: E402
from interp.sae.synthetic import planted_activations  # noqa: E402

OUT_JSON = ROOT / "agi-proof" / "interp" / "interp.public-report.json"


def _offline_invariants(seed: int = 0) -> "tuple[bool, dict]":
    """Train a toy SAE on a planted signal; assert the SAE + metric invariants."""
    d_in, n_features, k_true = 12, 6, 2
    d_hidden, k = 16, k_true
    X, _D = planted_activations(160, d_in, n_features, k_true, seed=seed, noise=0.02)

    sae = TopKSAE(d_in, d_hidden, k, seed=seed)
    fvu_init = metrics.fvu(X, sae.reconstruct_batch(X))
    curve = sae.fit(X, steps=600, lr=0.5)
    codes = sae.encode_batch(X)
    recon = [sae.decode(c) for c in codes]
    fvu_final = metrics.fvu(X, recon)
    l0_val = metrics.l0(codes)
    dead = metrics.dead_feature_fraction(codes, d_hidden)
    norms = sae.decoder_norms()

    # Determinism: an identical run reproduces the final FVU bit-for-bit.
    sae2 = TopKSAE(d_in, d_hidden, k, seed=seed)
    sae2.fit(X, steps=600, lr=0.5)
    fvu_repeat = metrics.fvu(X, sae2.reconstruct_batch(X))

    # TopK exactness: with all-positive pre-activations, exactly k features fire.
    probe = TopKSAE(4, 6, 3, seed=1)
    probe.W_enc = [[0.0] * 4 for _ in range(6)]
    probe.b_enc = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a_probe, keep_probe = probe.encode([0.0, 0.0, 0.0, 0.0])

    checks = {
        "metricFvuIdentityZero": metrics.fvu(X, X) == 0.0,
        "reconstructionImproves": fvu_final < fvu_init,
        "reconstructsPlantedSignal": fvu_final < 0.40,
        "topkExactL0": sum(1 for v in a_probe if v != 0.0) == 3 and keep_probe == [3, 4, 5],
        "decoderUnitNorm": max(abs(n - 1.0) for n in norms) < 1e-6,
        "deterministic": fvu_final == fvu_repeat,
        "lossDecreasing": curve[-1] < curve[0],
        "ceRecoveredFormula": abs(metrics.ce_loss_recovered(1.0, 1.2, 3.0) - 0.9) < 1e-9,
    }
    detail = {
        "metrics": {
            "fvuInit": round(fvu_init, 6),
            "fvuFinal": round(fvu_final, 6),
            "explainedVariance": round(1.0 - fvu_final, 6),
            "l0": round(l0_val, 6),
            "k": k,
            "deadFeatureFraction": round(dead, 6),
            "lossInit": round(curve[0], 6),
            "lossFinal": round(curve[-1], 6),
        },
        "checks": checks,
        "config": {"dIn": d_in, "dHidden": d_hidden, "kTrue": k_true, "nFeatures": n_features,
                   "steps": 600, "lr": 0.5, "seed": seed, "samples": len(X)},
        "hookpoint": hookpoints.resolve(hookpoints.default_layer(), "resid_post"),
    }
    return all(checks.values()), detail


def _write_report(detail: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "interp-sae-offline",
        "mode": "mock-offline",
        "visibility": "public-aggregate",
        "claimStatus": "Offline core only — a real SAE capability requires the gated M2 "
                       "metrics bar on harvested Qwen2.5-7B activations (CE-recovered ≥ ~0.9, "
                       "L0 in band, dead-% under ceiling). This artifact asserts the machinery, "
                       "not a capability.",
        "candidateOnly": True,
        "canClaimAGI": False,
        **detail,
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["mock"], default="mock",
                    help="offline core (no torch/GPU); the only mode implemented at M0")
    ap.add_argument("--dry-run", action="store_true", help="alias for --mode mock")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    ok, detail = _offline_invariants(seed=args.seed)
    _write_report(detail, args.out)
    print(f"wrote {args.out}")
    if ok:
        print("INTERP OFFLINE CORE VERIFIED ✓")
        return 0
    failed = [k for k, v in detail["checks"].items() if not v]
    print(f"INTERP OFFLINE CORE FAILED ✗  failing: {failed}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
