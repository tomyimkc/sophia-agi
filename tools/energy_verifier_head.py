#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""O2 (flagship) — energy-based verifier head: min-energy selection = Best-of-N self-verification.

Inspiration: Energy-Based Transformers (EBT, 2025) train a scalar ENERGY that verifies
context-prediction compatibility; prediction is gradient descent on that energy, and
min-energy selection is Best-of-N self-verification WITHOUT an external reward model.
AKOrN (ICLR 2025 Oral) reports oscillator energy is near-linearly calibrated to
correctness. This tool builds the LLM analogue over sophia's existing pieces:

  energy(answer, evidence) = -logit of a learned compatibility probe

trained on VERIFIER-LABELLED (answer, evidence) pairs (accepted -> compatible/low energy,
rejected/abstain -> incompatible/high energy). Low energy = trust; a high converged energy
is a principled ABSTAIN signal. For a query with k candidate answers, argmin-energy is the
selected answer (Best-of-N by self-verification).

Because this is a LEARNED verifier, its danger is Goodhart / verifier-coverage inheritance.
So the tool is built around two audits, not just a train accuracy:
  1. calibration of energy vs correctness (does low energy actually mean correct?)
  2. HELD-OUT-DOMAIN generalization: train on some domains, test on unseen ones. If the
     energy only works on domains it was trained on, it memorized the verifier's coverage
     rather than learning compatibility — reported as goodhartGap.

Honesty boundary: the learned probe here is a LINEAR energy over the repo's dependency-free
featurize_text (agent.activation_probes) — the SAME documented stand-in as the probe family.
A real energy head is a learned scalar over model hidden states
(build_hidden_state_featurizer, the named seam) and its differentiable gradient gives the
refinement step; this tool implements the selection + calibration + audit machinery and
marks the featurizer seam. It trains a probe but updates NO model weights.

Input JSONL, one record per (answer, evidence) pair used for training/eval:
  {"answer":"...", "evidence":"...", "accepted":true, "domain":"math", "correct":true}
  - accepted: the VERIFIER's verdict on this pair (the energy's supervision label)
  - domain:   used for the held-out-domain split
  - correct:  optional ground-truth for the calibration audit (defaults to `accepted`)

For Best-of-N selection eval, optionally group by "query" with a per-candidate "correct".
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

_IMPORT_ERR = ""
try:
    from agent.activation_probes import (
        train_centroid_probe, LinearProbe, featurize_text, build_hidden_state_featurizer,
    )
    from agent.calibration import calibration_report
    _REPO_OK = True
except Exception as e:  # pragma: no cover
    _REPO_OK = False
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def _env_artifact(reason: str) -> dict[str, Any]:
    return {
        "schema": "sophia.energy_verifier.v1",
        "environmentArtifact": True, "ok": False, "reason": reason,
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
    }


def _pair_text(rec: dict[str, Any]) -> str:
    """The text the compatibility probe reads: answer conditioned on its evidence."""
    return f"{rec.get('answer','')} || evidence: {rec.get('evidence','')}"


def energy_of(probe: "LinearProbe", rec: dict[str, Any]) -> float:
    """Energy = -logit(compatibility). Low energy = compatible/verified; high = abstain.

    score_vector is sigmoid(logit); we invert to recover the logit, then negate so that
    'more compatible' (probe score -> 1) maps to LOWER energy, as in an EBM.
    """
    s = probe.score_vector(featurize_text(_pair_text(rec)))
    s = min(max(s, 1e-6), 1 - 1e-6)
    logit = math.log(s / (1.0 - s))
    return -logit


def _train_energy(rows: list[dict[str, Any]]) -> "LinearProbe":
    """Train the compatibility probe: label = accepted (verifier verdict)."""
    probe_rows = [{"text": _pair_text(r), "label": bool(r.get("accepted"))} for r in rows]
    return train_centroid_probe(probe_rows, name="energy_verifier", threshold=0.5)


def _calibration_from_energy(probe: "LinearProbe", rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Confidence = sigmoid(-energy) = compatibility prob; audit vs `correct`."""
    confidences, correct = [], []
    for r in rows:
        e = energy_of(probe, r)
        confidences.append(1.0 / (1.0 + math.exp(e)))   # sigmoid(-energy)
        correct.append(bool(r.get("correct", r.get("accepted"))))
    if len(set(correct)) < 2:
        return {"degenerate": True, "n": len(correct)}
    return calibration_report(confidences, correct, coverage=0.5)


def _bestof_n(probe: "LinearProbe", groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Min-energy selection accuracy: pick argmin-energy candidate per query."""
    hits = usable = 0
    for _q, cands in groups.items():
        labelled = [c for c in cands if "correct" in c]
        if not labelled:
            continue
        usable += 1
        chosen = min(cands, key=lambda c: energy_of(probe, c))
        hits += int(bool(chosen.get("correct")))
    return {"queries": usable, "selectionAccuracy": round(hits / usable, 4) if usable else None}


def run(records: list[dict[str, Any]], *, seed: int = 0, min_domain_rows: int = 4) -> dict[str, Any]:
    if not _REPO_OK:
        return _env_artifact(f"repo instruments unavailable ({_IMPORT_ERR}); run with PYTHONPATH=.")
    rows = [r for r in records if r.get("answer") is not None]
    if not rows:
        return _env_artifact("no records with an 'answer' (fail-closed)")
    if not any(r.get("accepted") for r in rows) or not any(not r.get("accepted") for r in rows):
        return _env_artifact("training labels degenerate: need both accepted and rejected "
                             "(verifier) pairs to learn a compatibility energy")

    # 1) train on all, in-sample calibration
    probe_all = _train_energy(rows)
    cal_all = _calibration_from_energy(probe_all, rows)

    # 2) held-out-DOMAIN generalization (Goodhart audit)
    domains = sorted({str(r.get("domain", "default")) for r in rows})
    heldout = None
    if len(domains) >= 2:
        rng = random.Random(seed)
        test_dom = rng.choice(domains)
        train_rows = [r for r in rows if str(r.get("domain", "default")) != test_dom]
        test_rows = [r for r in rows if str(r.get("domain", "default")) == test_dom]
        if (len(train_rows) >= min_domain_rows and len(test_rows) >= 2
                and any(r.get("accepted") for r in train_rows)
                and any(not r.get("accepted") for r in train_rows)):
            probe_tr = _train_energy(train_rows)
            cal_in = _calibration_from_energy(probe_tr, train_rows)
            cal_out = _calibration_from_energy(probe_tr, test_rows)
            in_aurc = cal_in.get("aurc") if not cal_in.get("degenerate") else None
            out_aurc = cal_out.get("aurc") if not cal_out.get("degenerate") else None
            gap = (out_aurc - in_aurc) if (in_aurc is not None and out_aurc is not None) else None
            heldout = {
                "heldOutDomain": test_dom, "nTrain": len(train_rows), "nTest": len(test_rows),
                "inDomain": cal_in, "outDomain": cal_out,
                "goodhartGap": round(gap, 4) if gap is not None else None,
                # a large positive gap => energy fails on unseen domains => memorized verifier coverage
                "generalizes": (gap is not None and gap <= 0.15),
            }

    # 3) Best-of-N selection if records carry a query grouping
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("query") is not None:
            groups.setdefault(str(r["query"]), []).append(r)
    bestof = _bestof_n(probe_all, groups) if groups else None

    return {
        "schema": "sophia.energy_verifier.v1",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "n": len(rows), "domains": domains,
        "hiddenStateFeaturizerReady": _featurizer_ready(),  # the real energy-head seam
        "energyProbe": probe_all.to_dict(),
        "inSampleCalibration": cal_all,
        "heldOutDomain": heldout,
        "bestOfN": bestof,
        "note": ("energy = -logit(compatibility) over featurize_text (documented stand-in); "
                 "the real energy head is a learned scalar over hidden states "
                 "(build_hidden_state_featurizer seam). No model weights are updated here."),
    }


def _featurizer_ready() -> bool:
    """True only if the real hidden-state featurizer seam is implemented (it is a stub today)."""
    try:
        fz = build_hidden_state_featurizer(spec="mlx")
        return bool(fz and getattr(fz, "ready", False))
    except Exception:
        return False


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--records", required=True, help="JSONL of (answer, evidence, accepted, domain, correct)")
    p.add_argument("--output", default=None)
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
        report = run(records, seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not report.get("environmentArtifact") else 2


if __name__ == "__main__":
    raise SystemExit(main())