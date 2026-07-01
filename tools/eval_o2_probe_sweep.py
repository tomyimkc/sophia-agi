#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""O2 deepening: does a PROPER probe + more data/domains recover the hidden-state signal?

The first O2 run used a centroid mean-diff head and n=69/4-domains, giving out-of-fold
AUROC ~0.49-0.50. This gives O2 its fairest honest shot before concluding: a properly
regularized LOGISTIC probe (sklearn, L2) over the same real MLX hidden states, on the
LARGEST available real set (factcheck-full-r1 + fact-check-live, deduped -> more domains),
sweeping BOTH supervision labels (verifier `accepted` and ground-truth `correct`), each
evaluated leave-one-domain-out (no in-sample optimism) with a bootstrap AUROC CI.

If even this does not clear chance out-of-fold, the O2 negative is robust to head+data,
not an artifact of the weak centroid stand-in.
"""
from __future__ import annotations

import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from eval_o2_energy_hidden import transform_cases, pair_text, make_hidden_featurizer, auroc


def load_combined() -> list[dict]:
    rows = transform_cases("agi-proof/external-eval/factcheck-full-r1.json")
    seen = {r["answer"] for r in rows}
    for r in transform_cases("agi-proof/fact-check-live/fact-check-live-eval.LIVE-2026-06-24.json"):
        if r["answer"] not in seen:
            rows.append(r); seen.add(r["answer"])
    return rows


def boot_ci(scores, labels, n_boot=3000, seed=0):
    rng = random.Random(seed); n = len(labels); vals = []
    base = auroc(scores, labels)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [labels[i] for i in idx]
        if len(set(la)) < 2:
            continue
        a = auroc([scores[i] for i in idx], la)
        if a is not None:
            vals.append(a)
    vals.sort()
    return round(base, 4), [round(vals[int(0.025*len(vals))], 4), round(vals[int(0.975*len(vals))], 4)]


def lodo_logistic(feats, rows, label_key, C=1.0):
    """Leave-one-domain-out out-of-fold probabilities from an L2 logistic probe."""
    domains = sorted({r["domain"] for r in rows})
    X = np.asarray(feats)
    oof_p, oof_correct = [], []
    for d in domains:
        tr = [i for i, r in enumerate(rows) if r["domain"] != d]
        te = [i for i, r in enumerate(rows) if r["domain"] == d]
        ytr = [1 if rows[i][label_key] else 0 for i in tr]
        if len(set(ytr)) < 2:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6          # standardize on TRAIN only (no leak)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit((X[tr]-mu)/sd, ytr)
        p = clf.predict_proba((X[te]-mu)/sd)[:, 1]
        for j, i in enumerate(te):
            oof_p.append(float(p[j])); oof_correct.append(bool(rows[i]["correct"]))
    return oof_p, oof_correct


def main() -> int:
    rows = load_combined()
    domains = sorted({r["domain"] for r in rows})
    print(f"combined n={len(rows)} domains={len(domains)} accepted={sum(r['accepted'] for r in rows)} "
          f"correct={sum(r['correct'] for r in rows)}", file=sys.stderr)
    print("loading MLX featurizer ...", file=sys.stderr)
    fz = make_hidden_featurizer("Qwen/Qwen2.5-3B-Instruct")
    feats = [fz(pair_text(r)) for r in rows]

    report = {"schema": "sophia.o2_probe_sweep.v1", "candidateOnly": True, "canClaimAGI": False,
              "n": len(rows), "domains": domains, "featurizer": "Qwen2.5-3B MLX hidden states (real)",
              "arms": {}}
    for label_key in ("accepted", "correct"):
        for C in (0.1, 1.0):
            oof_p, oof_c = lodo_logistic(feats, rows, label_key, C=C)
            a, ci = boot_ci(oof_p, oof_c)
            key = f"logistic(C={C}) supervised_on={label_key}"
            beats = ci[0] > 0.5
            report["arms"][key] = {"oofAUROC": a, "ci95": ci, "beatsChance(CI>0.5)": beats, "nOOF": len(oof_c)}
            print(f"  {key}: OOF AUROC={a} CI={ci} beatsChance={beats}", file=sys.stderr)

    any_beats = any(v["beatsChance(CI>0.5)"] for v in report["arms"].values())
    report["verdict"] = ("hidden_signal_recovers" if any_beats
                         else "gate_not_met_robust — proper probe + more data/domains still does not clear chance out-of-fold")
    Path("/tmp/osc-data/o2_probe_sweep.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
