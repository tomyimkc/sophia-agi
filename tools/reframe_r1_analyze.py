#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""R1 — internal-vs-stated coherence: does an INTERNAL signal, coupled with the model's
STATED confidence, predict correctness better than stated confidence alone?

Signals per answer:
  stated  : verbalized confidence (what the model SAYS about itself)
  logprob : exp(mean token logprob)  (internal token-level certainty)
  probe   : honesty-probe P(honest) over Qwen hidden states of the answer text (W5-style)

Honesty probe trained on real DPO pairs (chosen = provenance-hedged = honest;
rejected = confident-false attribution = deceptive), Qwen2.5-3B hidden states.

Gate (per internal signal X): 5-fold cross-validated OOF AUROC of logistic(stated, X)
BEATS logistic(stated) alone at predicting `correct`, paired-bootstrap CI excluding 0,
reproduced across >=2 seeds. Honest style-confound check: does `probe` vary on terse answers?
No model weights updated.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from eval_o2_energy_hidden import make_hidden_featurizer, auroc


def train_honesty_probe(fz, dpo_path, max_pairs, seed):
    rows = [json.loads(l) for l in open(dpo_path)]
    random.Random(seed).shuffle(rows)
    rows = rows[:max_pairs]
    texts, labels = [], []
    for r in rows:
        texts.append(str(r["chosen"]));  labels.append(1)   # honest
        texts.append(str(r["rejected"])); labels.append(0)  # deceptive
    X = np.array([fz(t) for t in texts])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((X - mu) / sd, labels)
    insample = auroc(list(clf.predict_proba((X - mu) / sd)[:, 1]), [bool(l) for l in labels])
    return (lambda t: float(clf.predict_proba(((np.array(fz(t)) - mu) / sd).reshape(1, -1))[0, 1])), insample


def cv_oof(feature_cols, y, seed, n_splits=5):
    X = np.array(feature_cols).T  # (n, d)
    y = np.array(y, dtype=int)
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((X[tr] - mu) / sd, y[tr])
        oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return list(oof)


def paired_boot(a, b, labels, n_boot, seed):
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
    return {"aurocCombined": round(ba, 4), "aurocStatedAlone": round(bb, 4),
            "delta": round(ba - bb, 4), "ci95": [round(d[int(.025*len(d))], 4), round(d[int(.975*len(d))], 4)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/tmp/reframe-data/r1_pack.jsonl")
    ap.add_argument("--dpo", default="training/local_sophia_7b/dpo_hard_negatives.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-pairs", type=int, default=250)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(a.pack)]
    rows = [r for r in rows if r.get("stated") is not None and r.get("logprob") is not None]
    y = [1 if r["correct"] else 0 for r in rows]
    print(f"n={len(rows)} correct={sum(y)} incorrect={len(y)-sum(y)}", file=sys.stderr)

    print("loading MLX featurizer + training honesty probe ...", file=sys.stderr)
    fz = make_hidden_featurizer(a.model)
    probe, probe_insample = train_honesty_probe(fz, a.dpo, a.max_pairs, seed=0)
    print(f"honesty probe in-sample AUROC (honest vs deceptive): {probe_insample:.3f}", file=sys.stderr)

    stated = [float(r["stated"]) for r in rows]
    logprob = [float(r["logprob"]) for r in rows]
    hprobe = [probe(r["answer"]) for r in rows]

    # honest style-confound check: does the probe carry any signal on terse answers?
    probe_vs_correct = auroc(hprobe, [bool(v) for v in y])
    import statistics as st
    report = {"schema": "sophia.reframe_r1.v1", "candidateOnly": True, "canClaimAGI": False,
              "reframe": "R1 internal-vs-stated coherence", "n": len(rows), "nCorrect": sum(y),
              "honestyProbeInSampleAUROC": round(probe_insample, 4),
              "styleConfoundCheck": {"probeStd": round(st.pstdev(hprobe), 4),
                                     "probeAloneAUROCvsCorrect": round(probe_vs_correct, 4) if probe_vs_correct else None},
              "arms": {}, "seeds": {}}

    internal = {"logprob": logprob, "probe": hprobe, "probe+logprob": None}
    for seed in (0, 1, 7):
        base = cv_oof([stated], y, seed)
        seed_res = {}
        for name, sig in [("stated+logprob", [stated, logprob]),
                          ("stated+probe", [stated, hprobe]),
                          ("stated+probe+logprob", [stated, hprobe, logprob])]:
            comb = cv_oof(sig, y, seed)
            seed_res[name] = paired_boot(comb, base, [bool(v) for v in y], a.n_boot, seed)
        report["seeds"][seed] = seed_res

    # summarize: an arm PASSES if delta CI lower bound > 0 on ALL seeds
    for name in ("stated+logprob", "stated+probe", "stated+probe+logprob"):
        deltas = [report["seeds"][s][name]["delta"] for s in (0, 1, 7)]
        wins = all(report["seeds"][s][name]["ci95"][0] > 0 for s in (0, 1, 7))
        report["arms"][name] = {"meanDelta": round(sum(deltas) / 3, 4), "beatsStatedAlone_allSeeds": wins}
    report["verdict"] = ("internal_signal_adds_over_stated"
                         if any(v["beatsStatedAlone_allSeeds"] for v in report["arms"].values())
                         else "no_arm_beats_stated_alone")
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
