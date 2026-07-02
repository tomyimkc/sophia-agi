#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""R5 — operationalize the one signal that worked as a decode-time ABSTENTION gate.

The capstone / adoption test. R1 showed the internal honesty-probe signal, coupled with the
model's stated confidence, adds discrimination. R5 turns that into a deployable selective-
prediction policy and measures the risk-coverage (AURC) improvement — the metric that matters
if you actually want to *abstain more wisely*.

Two gates over the same answers:
  baseline : answer/abstain by STATED confidence (what the model says about itself)
  R5       : answer/abstain by a cross-validated logistic(stated, honesty-probe) confidence

Gate: paired AURC delta (statedAURC - combinedAURC) 95% CI excluding 0 (the internal-coupled
gate fabricates less at matched coverage), reproduced >=2 seeds. Reuses R1 data; no new
generation. NOTE: full generative activation-steering along the honesty direction is the deeper
version (left as future work); this measures the tractable selection/abstention adoption.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from agent.selective_risk import aurc, paired_aurc_delta_ci
from eval_o2_energy_hidden import make_hidden_featurizer
from reframe_r1_analyze import train_honesty_probe


def cv_conf(cols, y, seed, n_splits=5):
    X = np.array(cols).T; y = np.array(y, int); oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((X[tr]-mu)/sd, y[tr])
        oof[te] = clf.predict_proba((X[te]-mu)/sd)[:, 1]
    return list(oof)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/tmp/reframe-data/r1_combined.jsonl")
    ap.add_argument("--dpo", default="training/local_sophia_7b/dpo_hard_negatives.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-boot", type=int, default=4000); ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(a.pack)]
    rows = [r for r in rows if r.get("stated") is not None]
    y = [1 if r["correct"] else 0 for r in rows]
    print(f"n={len(rows)} correct={sum(y)}", file=sys.stderr)
    fz = make_hidden_featurizer(a.model)
    probe, _ = train_honesty_probe(fz, a.dpo, 250, seed=0)
    stated = [float(r["stated"]) for r in rows]
    hprobe = [probe(r["answer"]) for r in rows]
    fabricated = [0 if v else 1 for v in y]

    report = {"schema": "sophia.reframe_r5.v1", "candidateOnly": True, "canClaimAGI": False,
              "reframe": "R5 internal-coupled abstention gate (risk-coverage)", "n": len(rows), "seeds": {}}
    for seed in (0, 1, 7):
        base_conf = stated  # stated as-is is the baseline gate confidence
        comb_conf = cv_conf([stated, hprobe], y, seed)
        base_items = list(zip(base_conf, fabricated))
        comb_items = list(zip(comb_conf, fabricated))
        base_aurc, comb_aurc = aurc(base_items), aurc(comb_items)
        ci = paired_aurc_delta_ci(base_items, comb_items, n_boot=a.n_boot, seed=seed)
        report["seeds"][seed] = {"statedAURC": round(base_aurc, 4), "combinedAURC": round(comb_aurc, 4),
                                 "delta": round(base_aurc - comb_aurc, 4),
                                 "ci95": [round(ci[0], 4), round(ci[1], 4)], "combinedWins": ci[0] > 0}
    wins = all(report["seeds"][s]["combinedWins"] for s in (0, 1, 7))
    report["internalGateBeatsStated_allSeeds"] = wins
    report["verdict"] = ("internal_coupled_gate_lowers_risk" if wins
                         else "no_significant_risk_coverage_gain")
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
