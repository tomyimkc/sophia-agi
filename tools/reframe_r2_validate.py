#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""R2 validation on balanced multi-domain MMLU + the paired head-to-head vs the confidence margin.

Setup: one domain-specialist probe per MMLU subject (label=correct, both-class). For a row in
subject S, the OOD ensemble = probes from subjects != S (genuinely out-of-domain). Signals:
  disagreement = std of the OOD probes' P(correct)
  margin       = |mean(P) - 0.5|   (the ensemble's own confidence — the baseline to beat)
  ensembleError = round(mean(P)) != correct   (what an abstain signal should predict)

Two gates:
  A (usable):   AUROC(disagreement -> ensembleError) > 0.5, bootstrap CI excluding chance, >=2 seeds.
  B (additive): does adding disagreement to margin beat MARGIN ALONE at predicting error? CV-OOF
     logistic(margin, disagreement) vs logistic(margin), paired-bootstrap AUROC delta CI excluding 0.
     This is the head-to-head R2 had not yet won. Also reports disagreement-alone vs margin-alone.
No weights updated. candidateOnly.
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
from eval_o2_energy_hidden import make_hidden_featurizer, auroc


def boot_vs_chance(scores, labels, n_boot, seed):
    rng = random.Random(seed); n = len(labels); v = []
    base = auroc(scores, labels)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [labels[i] for i in idx]
        if len(set(la)) < 2:
            continue
        a = auroc([scores[i] for i in idx], la)
        if a is not None:
            v.append(a)
    v.sort()
    return {"auroc": round(base, 4), "ci95": [round(v[int(.025*len(v))], 4), round(v[int(.975*len(v))], 4)]}


def paired_delta(a, b, labels, n_boot, seed):
    rng = random.Random(seed); n = len(labels); d = []
    ba, bb = auroc(a, labels), auroc(b, labels)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [labels[i] for i in idx]
        if len(set(la)) < 2:
            continue
        xa, xb = auroc([a[i] for i in idx], la), auroc([b[i] for i in idx], la)
        if xa is not None and xb is not None:
            d.append(xa - xb)
    d.sort()
    return {"aurocA": round(ba, 4), "aurocB": round(bb, 4), "delta": round(ba - bb, 4),
            "ci95": [round(d[int(.025*len(d))], 4), round(d[int(.975*len(d))], 4)]}


def cv_oof(cols, y, seed, n_splits=5):
    X = np.array(cols).T; y = np.array(y, int); oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((X[tr]-mu)/sd, y[tr])
        oof[te] = clf.predict_proba((X[te]-mu)/sd)[:, 1]
    return list(oof)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/tmp/r2-data/mmlu_pack.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-boot", type=int, default=5000); ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(a.pack)]
    subjects = sorted({r["subject"] for r in rows})
    print(f"n={len(rows)} subjects={len(subjects)} correct={sum(r['correct'] for r in rows)}", file=sys.stderr)
    fz = make_hidden_featurizer(a.model)
    X = np.array([fz(r["answerText"]) for r in rows])
    subj = [r["subject"] for r in rows]
    y = [1 if r["correct"] else 0 for r in rows]

    probes = {}
    for s in subjects:
        idx = [i for i in range(len(rows)) if subj[i] == s]
        ys = [y[i] for i in idx]
        if len(set(ys)) < 2:
            continue
        mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-6
        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit((X[idx]-mu)/sd, ys)
        probes[s] = (clf, mu, sd)
    print(f"trained {len(probes)}/{len(subjects)} domain probes", file=sys.stderr)

    disagreement, margin, err = [], [], []
    for i in range(len(rows)):
        ood = [s for s in probes if s != subj[i]]
        if len(ood) < 2:
            continue
        preds = [float(probes[s][0].predict_proba(((X[i]-probes[s][1])/probes[s][2]).reshape(1, -1))[0, 1]) for s in ood]
        mean = sum(preds) / len(preds)
        disagreement.append(float(np.std(preds))); margin.append(abs(mean - 0.5))
        err.append(0 if round(mean) == y[i] else 1)
    err_lab = [bool(e) for e in err]
    print(f"scored {len(err)} rows | OOD ensemble error rate={sum(err)/len(err):.3f}", file=sys.stderr)

    report = {"schema": "sophia.reframe_r2_validate.v1", "candidateOnly": True, "canClaimAGI": False,
              "dataset": "MMLU (6 subjects, exact labels)", "n": len(err), "nDomainProbes": len(probes),
              "subjects": sorted(probes), "oodEnsembleErrorRate": round(sum(err)/len(err), 4),
              "gateA_usable": {}, "gateB_additive": {}, "headToHead_disagreement_vs_margin": {}}
    neg_margin = [-m for m in margin]   # low margin -> more error
    for seed in (0, 1, 7):
        report["gateA_usable"][seed] = boot_vs_chance(disagreement, err_lab, a.n_boot, seed)
        comb = cv_oof([margin, disagreement], y=err, seed=seed)   # predict error from margin+disagreement
        base = cv_oof([margin], y=err, seed=seed)
        report["gateB_additive"][seed] = paired_delta(comb, base, err_lab, a.n_boot, seed)
        report["headToHead_disagreement_vs_margin"][seed] = paired_delta(disagreement, neg_margin, err_lab, a.n_boot, seed)

    A = all(report["gateA_usable"][s]["ci95"][0] > 0.5 for s in (0, 1, 7))
    B = all(report["gateB_additive"][s]["ci95"][0] > 0 for s in (0, 1, 7))
    report["gateA_passed_allSeeds"] = A
    report["gateB_passed_allSeeds"] = B
    report["verdict"] = ("R2_VALIDATED: disagreement is usable AND adds error-prediction value beyond the confidence margin"
                         if (A and B) else
                         "R2_usable_but_not_additive: disagreement beats chance but does not add beyond the margin"
                         if A else "R2_not_usable_on_this_data")
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
