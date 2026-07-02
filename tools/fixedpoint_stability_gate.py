#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""O3 — DEQ-style fixed-point stability as a claim-vs-evidence consistency gate.

Inspiration: Deep Equilibrium Models (Bai, Kolter, Koltun, NeurIPS 2019) define an
answer as the FIXED POINT of a weight-tied map and use the residual ||f(z)-z|| as a
stability signal. This tool reframes verification as: does (claim + evidence) settle to
a stable, self-consistent equilibrium?

The map g projects the claim onto the span of its evidence and renormalizes:
  z_{t+1} = normalize( sum_j a_j(z_t) * e_j ),  a_j = softmax_j( <z_t, e_j> / tau )
i.e. re-express the claim as the attention-weighted combination of its evidence
embeddings, iterated to convergence. The KEY design choice (answering the strongest
objection to O3): the map routes through the EVIDENCE set, never the claim's own text.
So the fixed point is 'the closest thing the evidence can reconstruct', and:

  - residual ||claim - z*||  = how far the claim is from what the evidence supports
  - drift    ||z_T - z_0||   = how far the reconstruction moved from the claim
  - converged (small step)   = a stable equilibrium was reached at all

A claim well-supported by its evidence sits at (or near) the evidence-reconstruction
fixed point: small residual, fast convergence. An unsupported claim either can't be
reconstructed (large residual) or the iteration wanders (non-convergence) -> abstain.

This is a soft complement to agent.smt_verifier (which certifies a narrow DECIDABLE band
exactly): O3 gives a graded stability score on the large undecidable remainder. It is a
pure numerical instrument (numpy only), changes no weights, and is fail-closed.

Input JSONL, one record per claim:
  {"claim":"...", "evidence":["src1 text", "src2 text", ...], "supported":true}
  - supported: optional offline audit label (does the evidence actually support the claim?)
Output: per-record residual/drift/converged + a threshold sweep and, if labels present,
an AUROC-style separation of supported vs unsupported by residual.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

_IMPORT_ERR = ""
try:
    import oscillator_core as oc          # reuse the shared hash_embed (documented seam)
    from agent.selective_risk import aurc
    _REPO_OK = True
except Exception as e:  # pragma: no cover
    _REPO_OK = False
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def _env_artifact(reason: str) -> dict[str, Any]:
    return {
        "schema": "sophia.fixedpoint_stability.v1",
        "environmentArtifact": True, "ok": False, "reason": reason,
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
    }


def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = x / max(tau, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def iterate_fixedpoint(claim: str, evidence: list[str], *, dim: int, tau: float = 0.3,
                       steps: int = 50, tol: float = 1e-4) -> dict[str, Any]:
    """Iterate z_{t+1}=normalize(sum_j softmax(<z,e_j>/tau) e_j) from z_0=embed(claim)."""
    c = oc.hash_embed(claim, dim)
    E = np.stack([oc.hash_embed(e, dim) for e in evidence]) if evidence else np.zeros((0, dim))
    if E.shape[0] == 0 or np.linalg.norm(c) == 0:
        # no evidence, or empty claim -> maximally unstable (cannot be reconstructed)
        return {"residual": 1.0, "drift": 1.0, "converged": False, "steps": 0}
    z = c.copy()
    last_step = 1.0
    converged = False
    used = 0
    for t in range(int(steps)):
        sims = E @ z                       # <z, e_j>
        a = _softmax(sims, tau)
        z_new = a @ E
        n = np.linalg.norm(z_new)
        z_new = z_new / n if n > 0 else z_new
        last_step = float(np.linalg.norm(z_new - z))
        z = z_new
        used = t + 1
        if last_step < tol:
            converged = True
            break
    residual = float(np.linalg.norm(c - z))      # claim vs evidence-reconstruction fixed point
    drift = residual
    return {"residual": round(residual, 6), "drift": round(drift, 6),
            "converged": bool(converged), "lastStep": round(last_step, 8), "steps": used}


def run(records: list[dict[str, Any]], *, dim: int = oc.EMBED_DIM_DEFAULT,
        tau: float = 0.3, steps: int = 50, residual_threshold: float = 0.6) -> dict[str, Any]:
    if not _REPO_OK:
        return _env_artifact(f"repo instruments unavailable ({_IMPORT_ERR}); run with PYTHONPATH=.")
    rows = [r for r in records if r.get("claim") is not None]
    if not rows:
        return _env_artifact("no records with a 'claim' (fail-closed)")

    per = []
    for r in rows:
        fp = iterate_fixedpoint(str(r.get("claim", "")), [str(e) for e in r.get("evidence", [])],
                                dim=dim, tau=tau, steps=steps)
        # gate: stable AND low residual -> accept; else abstain (fail-closed default)
        accept = fp["converged"] and fp["residual"] <= residual_threshold
        rec = {"claim": str(r.get("claim", ""))[:80], **fp, "decision": "accept" if accept else "abstain"}
        if "supported" in r:
            rec["supported"] = bool(r["supported"])
        per.append(rec)

    out: dict[str, Any] = {
        "schema": "sophia.fixedpoint_stability.v1",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "n": len(per), "residualThreshold": residual_threshold,
        "hashEmbedSeam": oc.active_embed_backend().startswith("hash"),  # false once a semantic embedder is wired
        "embedBackend": oc.active_embed_backend(),
        "abstainRate": round(sum(1 for p in per if p["decision"] == "abstain") / len(per), 4),
        "records": per,
    }

    # If labels present: does residual separate supported (low) from unsupported (high)?
    labelled = [p for p in per if "supported" in p]
    if labelled and len({p["supported"] for p in labelled}) == 2:
        # AURC over items (confidence=1-residual, fabricated=not supported): lower AURC = residual
        # ranks supported above unsupported.
        items = [(max(0.0, 1.0 - p["residual"]), 0 if p["supported"] else 1) for p in labelled]
        sup_res = [p["residual"] for p in labelled if p["supported"]]
        uns_res = [p["residual"] for p in labelled if not p["supported"]]
        # data-driven threshold: the residual midpoint between the class means. The fixed
        # default is deliberately conservative (abstains under uncertainty); this reports
        # the threshold the labelled data actually supports, plus its accept/abstain split,
        # so the operator calibrates from evidence rather than a hardcoded constant.
        suggested = round((float(np.mean(sup_res)) + float(np.mean(uns_res))) / 2.0, 4)
        acc_sup = sum(1 for r in sup_res if r <= suggested)
        acc_uns = sum(1 for r in uns_res if r <= suggested)
        out["separation"] = {
            "nLabelled": len(labelled),
            "aurcByResidual": round(aurc(items), 6),
            "meanResidualSupported": round(float(np.mean(sup_res)), 4),
            "meanResidualUnsupported": round(float(np.mean(uns_res)), 4),
            "suggestedThreshold": suggested,
            "atSuggested": {
                "supportedAccepted": f"{acc_sup}/{len(sup_res)}",
                "unsupportedAccepted": f"{acc_uns}/{len(uns_res)}",
            },
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
    p.add_argument("--records", required=True, help="JSONL of {claim, evidence:[...], supported?}")
    p.add_argument("--output", default=None)
    p.add_argument("--dim", type=int, default=oc.EMBED_DIM_DEFAULT)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--residual-threshold", type=float, default=0.6)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = _load_jsonl(args.records)
    except Exception as e:
        report = _env_artifact(f"could not read --records ({type(e).__name__}: {e})")
        records = None
    if records is not None:
        report = run(records, dim=args.dim, tau=args.tau, steps=args.steps,
                     residual_threshold=args.residual_threshold)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not report.get("environmentArtifact") else 2


if __name__ == "__main__":
    raise SystemExit(main())