#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""O1 — Kuramoto order-parameter consensus as a self-consistency confidence gate.

Inspiration: unconv-ai/Un-0 reads an answer out of how coherently a coupled-oscillator
population settles. sophia already does self-consistency selective prediction (the
validated SimpleQA result), where confidence = fraction of samples agreeing with the
majority (agent.calibration.self_consistency). This tool proposes a DIFFERENT, training-
free confidence readout over the SAME samples: embed the k sampled answers, couple them
by similarity, integrate Kuramoto dynamics, and take the order parameter r in [0,1] as
confidence. r is a strict generalization of the agreement count — identical answers sync
fully, near-duplicates still reinforce, so r sees soft agreement a hard vote-count misses.

The honest question is whether r actually BEATS the majority-agreement gate. This tool
runs that head-to-head using the repo's own paired AURC bootstrap
(agent.selective_risk.paired_aurc_delta_ci): if r-gating's AURC is lower and the paired
95% CI on (baseline_aurc - r_aurc) excludes 0, r is a real improvement; otherwise it is
NOT kept. A run is not a result: this measures an instrument on whatever labeled samples
you feed it; it changes no weights.

Input: JSONL, one record per query:
  {"samples": ["answer a", "answer a", "answer b"], "correct": true}
  - samples: the k sampled model answers for this query (strings)
  - correct:  whether the MAJORITY answer is correct (bool) — the offline audit label

Output JSON:
  - baselineGate:  calibration report for the majority-agreement confidence
  - consensusGate: calibration report for the Kuramoto r confidence
  - aurcDelta:     {baseline, consensus, delta, ci95, consensusWins}
  - verdict:       "consensus_beats_baseline" | "no_improvement" | "inconclusive"
Fail-closed: missing repo instruments, empty input, or degenerate labels -> environment
artifact (ok:false) and a non-zero exit; never a fabricated win.
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
    from agent.calibration import self_consistency, calibration_report
    from agent.selective_risk import aurc, paired_aurc_delta_ci
    import oscillator_core as oc
    _REPO_OK = True
except Exception as e:  # pragma: no cover - exercised via fail-closed path
    _REPO_OK = False
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def _env_artifact(reason: str) -> dict[str, Any]:
    """Fail-closed environment artifact: no metric is a result on this path."""
    return {
        "schema": "sophia.consensus_gate.v1",
        "environmentArtifact": True,
        "ok": False,
        "reason": reason,
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
    }


def score_record(rec: dict[str, Any], *, dim: int, steps: int, gain: float,
                 seed: int) -> dict[str, Any]:
    """Return {majorityConf, consensusR, correct} for one query's samples."""
    samples = [str(s) for s in rec.get("samples", []) if str(s).strip()]
    correct = bool(rec.get("correct", False))
    _, maj_conf = self_consistency(samples)          # repo's agreement-fraction confidence
    r = oc.consensus_r(samples, dim=dim, steps=steps, gain=gain, seed=seed)
    return {"majorityConf": float(maj_conf), "consensusR": float(r),
            "correct": correct, "nSamples": len(samples)}


def run(records: list[dict[str, Any]], *, dim: int = oc.EMBED_DIM_DEFAULT,
        steps: int = 40, gain: float = 2.0, seed: int = 0, coverage: float = 0.5,
        n_boot: int = 5000) -> dict[str, Any]:
    """Head-to-head: majority-agreement gate vs Kuramoto-r gate on labeled samples."""
    if not _REPO_OK:
        return _env_artifact(f"repo instruments unavailable ({_IMPORT_ERR}); run inside the "
                             f"repo with PYTHONPATH=. so agent.* imports resolve")
    rows = [r for r in records if r.get("samples")]
    if not rows:
        return _env_artifact("no records with samples (fail-closed; nothing to gate)")
    # Order-invariance: AURC breaks confidence ties by position, so a gate comparison must
    # not depend on the order records arrive in. Shuffle once (seeded) before scoring so
    # neither arm gets a spurious tiebreak advantage from input order; the paired bootstrap
    # then sees the same regime as the point estimate.
    import random as _random
    rows = list(rows)
    _random.Random(seed).shuffle(rows)
    scored = [score_record(r, dim=dim, steps=steps, gain=gain, seed=seed) for r in rows]
    correct = [s["correct"] for s in scored]
    if len(set(correct)) < 2:
        return _env_artifact("labels are degenerate (all correct or all wrong); "
                             "risk-coverage is undefined — feed a mixed-outcome set")

    base_conf = [s["majorityConf"] for s in scored]
    cons_conf = [s["consensusR"] for s in scored]

    # AURC items are (confidence, fabricated); fabricated = NOT correct.
    fab = [0 if c else 1 for c in correct]
    base_items = list(zip(base_conf, fab))
    cons_items = list(zip(cons_conf, fab))
    base_aurc, cons_aurc = aurc(base_items), aurc(cons_items)
    # delta = baseline - consensus; POSITIVE delta and a CI excluding 0 => consensus wins
    ci = paired_aurc_delta_ci(base_items, cons_items, n_boot=n_boot, seed=seed)
    delta = base_aurc - cons_aurc
    consensus_wins = ci[0] > 0.0  # lower CI bound of (baseline - consensus) above 0

    if consensus_wins:
        verdict = "consensus_beats_baseline"
    elif ci[1] < 0.0:
        verdict = "no_improvement"       # baseline strictly better
    else:
        verdict = "inconclusive"         # CI straddles 0

    return {
        "schema": "sophia.consensus_gate.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "n": len(scored),
        "hashEmbedSeam": oc.active_embed_backend().startswith("hash"),  # false once a semantic embedder is wired
        "embedBackend": oc.active_embed_backend(),
        "baselineGate": calibration_report(base_conf, correct, coverage=coverage),
        "consensusGate": calibration_report(cons_conf, correct, coverage=coverage),
        "aurcDelta": {
            "baseline": round(base_aurc, 6),
            "consensus": round(cons_aurc, 6),
            "delta": round(delta, 6),
            "ci95": [round(ci[0], 6), round(ci[1], 6)],
            "consensusWins": consensus_wins,
        },
        "verdict": verdict,
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--records", required=True, help="JSONL: {samples:[...], correct:bool}")
    p.add_argument("--output", default=None, help="optional JSON output path")
    p.add_argument("--dim", type=int, default=oc.EMBED_DIM_DEFAULT)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--gain", type=float, default=2.0)
    p.add_argument("--coverage", type=float, default=0.5)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = _load_jsonl(args.records)
    except Exception as e:
        report = _env_artifact(f"could not read --records ({type(e).__name__}: {e})")
        records = None
    if records is not None:
        report = run(records, dim=args.dim, steps=args.steps, gain=args.gain,
                     coverage=args.coverage, n_boot=args.n_boot, seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    # exit 0 only on a real, non-environment report
    return 0 if not report.get("environmentArtifact") else 2


if __name__ == "__main__":
    raise SystemExit(main())