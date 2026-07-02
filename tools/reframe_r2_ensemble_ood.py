#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""R2 — ensemble disagreement as an OOD / abstain signal.

Turns the cross-domain-transfer failure (O2/W1) into a feature. Train one domain-specialist
probe per domain over Qwen hidden states. For an input, the OTHER-domain probes are seeing it
out-of-domain; if they DISAGREE, the ensemble is unreliable there -> abstain.

Gate: disagreement (std of the out-of-domain probes' P(correct)) predicts the out-of-domain
ENSEMBLE's error better than chance — AUROC(disagreement, ensembleError) > 0.5, paired-bootstrap
CI excluding 0.5, reproduced >=2 seeds. If it clears chance, disagreement is a usable abstain
signal even though no single probe generalizes. No weights updated.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from eval_o2_energy_hidden import make_hidden_featurizer, transform_cases, pair_text, auroc


def load_combined():
    rows = transform_cases("agi-proof/external-eval/factcheck-full-r1.json")
    seen = {r["answer"] for r in rows}
    for r in transform_cases("agi-proof/fact-check-live/fact-check-live-eval.LIVE-2026-06-24.json"):
        if r["answer"] not in seen:
            rows.append(r); seen.add(r["answer"])
    return rows


def boot_auroc_vs_chance(scores, labels, n_boot, seed):
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--min-domain", type=int, default=8)
    ap.add_argument("--label", choices=["correct", "accepted"], default="accepted",
                    help="per-domain probe target; 'correct' is near-degenerate on factcheck (use 'accepted')")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    rows = load_combined()
    from collections import Counter
    dc = Counter(r["domain"] for r in rows)
    domains = [d for d, c in dc.items() if c >= a.min_domain]
    rows = [r for r in rows if r["domain"] in domains]
    print(f"n={len(rows)} domains(>= {a.min_domain} rows)={domains}", file=sys.stderr)

    print("featurizing (Qwen hidden states) ...", file=sys.stderr)
    fz = make_hidden_featurizer(a.model)
    X = np.array([fz(pair_text(r)) for r in rows])
    dom = [r["domain"] for r in rows]
    y = [1 if r[a.label] else 0 for r in rows]   # probe target (default 'accepted')

    # one probe per domain, trained on that domain's rows (label=correct)
    probes = {}
    for d in domains:
        idx = [i for i in range(len(rows)) if dom[i] == d]
        yd = [y[i] for i in idx]
        if len(set(yd)) < 2:
            continue
        mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-6
        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit((X[idx] - mu) / sd, yd)
        probes[d] = (clf, mu, sd)
    print(f"trained {len(probes)} domain probes: {sorted(probes)}", file=sys.stderr)

    disagreement, ensemble_err, single_conf = [], [], []
    for i in range(len(rows)):
        ood = [d for d in probes if d != dom[i]]
        if len(ood) < 2:
            continue
        preds = [float(probes[d][0].predict_proba(((X[i] - probes[d][1]) / probes[d][2]).reshape(1, -1))[0, 1]) for d in ood]
        ens = sum(preds) / len(preds)
        disagreement.append(float(np.std(preds)))
        ensemble_err.append(0 if (round(ens) == y[i]) else 1)     # 1 = OOD ensemble WRONG
        single_conf.append(abs(ens - 0.5))                          # baseline: ensemble margin
    err_lab = [bool(e) for e in ensemble_err]
    err_rate = (sum(ensemble_err) / len(ensemble_err)) if ensemble_err else 0.0
    print(f"scored {len(disagreement)} rows | OOD ensemble error rate={err_rate:.3f}", file=sys.stderr)

    report = {"schema": "sophia.reframe_r2.v1", "candidateOnly": True, "canClaimAGI": False,
              "reframe": "R2 ensemble disagreement as OOD/abstain signal", "probeLabel": a.label,
              "n": len(disagreement), "nDomainProbes": len(probes), "domains": sorted(probes),
              "oodEnsembleErrorRate": round(err_rate, 4), "seeds": {}}

    if len(probes) < 3 or len(set(err_lab)) < 2 or len(disagreement) < 20:
        report["verdict"] = ("underpowered_on_available_data — the factcheck domains are too few/"
                             "imbalanced to train >=3 both-class probes and produce OOD ensemble errors; "
                             "R2 needs a balanced multi-domain correctness set")
        report["blocked"] = {"trainableDomainProbes": len(probes), "errorLabelClasses": len(set(err_lab)),
                             "scoredRows": len(disagreement)}
        txt = json.dumps(report, indent=2)
        if a.output:
            Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
        print(txt); return 0
    for seed in (0, 1, 7):
        dis = boot_auroc_vs_chance(disagreement, err_lab, a.n_boot, seed)
        base = boot_auroc_vs_chance([-c for c in single_conf], err_lab, a.n_boot, seed)  # low margin ~ error (baseline)
        report["seeds"][seed] = {"disagreementAUROCvsError": dis, "ensembleMarginBaselineAUROC": base}
    wins = all(report["seeds"][s]["disagreementAUROCvsError"]["ci95"][0] > 0.5 for s in (0, 1, 7))
    report["verdict"] = ("disagreement_is_usable_abstain_signal" if wins
                         else "disagreement_no_better_than_chance")
    report["disagreementBeatsChance_allSeeds"] = wins
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
