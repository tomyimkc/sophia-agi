#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""O2 flagship benchmark: hidden-state energy verifier vs the linear-stub, on real data.

Closes the two gaps the shipped tools/energy_verifier_head.py leaves open for the
`o2-energy-verifier-linear-stub-not-hidden-state` gate:

  1. REAL hidden-state featurizer — loads a local MLX model and calls the shipped
     agent.activation_probes.build_hidden_state_featurizer seam (no stub), so the
     energy is a learned scalar over Qwen hidden states, not featurize_text.
  2. The gate METRIC the tool never wired — AUROC (discrimination of energy vs
     ground-truth `correct`) with a PAIRED bootstrap CI, plus ECE and a held-out-
     domain generalization audit.

Honest evaluation protocol (no in-sample optimism):
  * Leave-One-Domain-Out CV — train the energy head on 3 domains, predict the 4th;
    pool the out-of-fold predictions. This is the generalization number AND the
    Goodhart audit in one (in-fold vs out-fold AURC = goodhartGap).
  * Paired bootstrap over the pooled out-of-fold rows: does AUROC(hidden) beat
    AUROC(linear-stub)? Does AUROC(hidden) beat chance (0.5)? CI excluding 0 = pass.

Data: real verifier-labelled (claim, evidence, verdict, correct) cases from
agi-proof/external-eval/factcheck-full-r1.json. verdict->accepted (accepted=True,
held/rejected=False) is the energy's supervision; `correct` is ground truth for the
audit; `type` is the domain. The brief's named source (agent.verified_trace_rlvr ->
verified_traces.jsonl) does not exist on disk and carries no evidence field, so the
factcheck pack is the real substitute — recorded honestly.

No model weights are updated (a light centroid-probe head fit over frozen hidden
states); this does NOT hit the --run-train human-escalation lane.
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

from agent.activation_probes import build_hidden_state_featurizer, featurize_text, train_centroid_probe
from agent.calibration import calibration_report


def transform_cases(path: str) -> list[dict[str, Any]]:
    cases = json.load(open(path))["cases"]
    rows = []
    for c in cases:
        ev_parts = []
        for cl in (c.get("claims") or []):
            for l in (cl.get("layers") or []):
                for e in (l.get("evidence") or []):
                    seg = " ".join(str(e.get(k, "")) for k in ("title", "publisher", "relation") if e.get(k))
                    if seg.strip():
                        ev_parts.append(seg.strip())
        evidence = " ; ".join(ev_parts) if ev_parts else str(c.get("reason", ""))
        rows.append({
            "id": c["id"], "answer": c.get("claim", ""), "evidence": evidence,
            "accepted": (c.get("verdict") == "accepted"),
            "domain": c.get("type", "default"),
            "correct": bool(c.get("correct")),
        })
    return rows


def pair_text(r: dict) -> str:
    return f"{r.get('answer','')} || evidence: {r.get('evidence','')}"


# ---- featurizers -----------------------------------------------------------
def make_hidden_featurizer(model_id: str):
    from mlx_lm import load
    model, tokenizer = load(model_id)
    fz = build_hidden_state_featurizer(spec="mlx", model=model, tokenizer=tokenizer)
    return fz


def zscore(mat: list[list[float]]) -> list[list[float]]:
    n = len(mat); d = len(mat[0])
    means = [sum(row[j] for row in mat) / n for j in range(d)]
    stds = [max(1e-6, (sum((row[j] - means[j]) ** 2 for row in mat) / n) ** 0.5) for j in range(d)]
    return [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in mat]


# ---- probe / energy over arbitrary feature vectors -------------------------
def train_energy_vecs(feats: list[list[float]], accepted: list[bool]):
    rows = [{"text": None, "label": a} for a in accepted]
    # replicate train_centroid_probe over precomputed vectors (mean-diff of pos/neg)
    pos = [f for f, a in zip(feats, accepted) if a]
    neg = [f for f, a in zip(feats, accepted) if not a]
    D = len(feats[0])
    if not pos or not neg:
        return [0.0] * D, 0.0
    mean = lambda vs: [sum(v[i] for v in vs) / len(vs) for i in range(D)]
    mp, mn = mean(pos), mean(neg)
    w = [a - b for a, b in zip(mp, mn)]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    bias = -0.5 * (dot(w, mp) + dot(w, mn))
    return w, bias


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def energy_of_vec(w, bias, f) -> float:
    dot = sum(x * y for x, y in zip(w, f))
    s = min(max(_sigmoid(dot + bias), 1e-6), 1 - 1e-6)
    return -math.log(s / (1.0 - s))   # low energy = compatible


def _conf_from_energy(e: float) -> float:
    return _sigmoid(-e)


# ---- metrics ---------------------------------------------------------------
def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """P(score[pos] > score[neg]); scores rank higher for the positive class."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    n = len(confidences)
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        tot += (len(idx) / n) * abs(acc - conf)
    return tot


def aurc(confidences: list[float], fabricated: list[int]) -> float:
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    run = 0.0; tot = 0.0
    for k, i in enumerate(order, 1):
        run += fabricated[i]
        tot += run / k
    return tot / len(order) if order else 0.0


def lodo_oof(rows, feat_fn):
    """Leave-one-domain-out: return pooled out-of-fold (energy, confidence, correct),
    plus mean in-fold and out-fold AURC (the goodhart audit)."""
    domains = sorted({r["domain"] for r in rows})
    feats_all = [feat_fn(pair_text(r)) for r in rows]
    feats_all = zscore(feats_all)
    by = {r["id"]: f for r, f in zip(rows, feats_all)}
    oof_energy, oof_conf, oof_correct = [], [], []
    in_aurcs, out_aurcs = [], []
    for d in domains:
        tr = [r for r in rows if r["domain"] != d]
        te = [r for r in rows if r["domain"] == d]
        if not any(r["accepted"] for r in tr) or not any(not r["accepted"] for r in tr):
            continue
        w, bias = train_energy_vecs([by[r["id"]] for r in tr], [r["accepted"] for r in tr])
        # out-of-fold predictions on the held-out domain
        for r in te:
            e = energy_of_vec(w, bias, by[r["id"]])
            oof_energy.append(e); oof_conf.append(_conf_from_energy(e)); oof_correct.append(r["correct"])
        # in-fold vs out-fold AURC for the goodhart gap
        tr_conf = [_conf_from_energy(energy_of_vec(w, bias, by[r["id"]])) for r in tr]
        te_conf = [_conf_from_energy(energy_of_vec(w, bias, by[r["id"]])) for r in te]
        if len(set(r["correct"] for r in tr)) == 2:
            in_aurcs.append(aurc(tr_conf, [0 if r["correct"] else 1 for r in tr]))
        if len(set(r["correct"] for r in te)) == 2:
            out_aurcs.append(aurc(te_conf, [0 if r["correct"] else 1 for r in te]))
    gap = (sum(out_aurcs) / len(out_aurcs) - sum(in_aurcs) / len(in_aurcs)) if in_aurcs and out_aurcs else None
    return oof_energy, oof_conf, oof_correct, gap


def paired_bootstrap_delta(scores_a, scores_b, labels, n_boot=5000, seed=0):
    """CI of AUROC(a)-AUROC(b) over the SAME resampled rows."""
    rng = random.Random(seed)
    n = len(labels); deltas = []
    base_a, base_b = auroc(scores_a, labels), auroc(scores_b, labels)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [labels[i] for i in idx]
        if len(set(la)) < 2:
            continue
        a = auroc([scores_a[i] for i in idx], la)
        b = auroc([scores_b[i] for i in idx], la)
        if a is not None and b is not None:
            deltas.append(a - b)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]; hi = deltas[int(0.975 * len(deltas))]
    return {"aurocA": round(base_a, 4), "aurocB": round(base_b, 4),
            "delta": round(base_a - base_b, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def bootstrap_vs_chance(scores, labels, n_boot=5000, seed=0):
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
    return {"auroc": round(base, 4), "ci95": [round(vals[int(0.025*len(vals))], 4), round(vals[int(0.975*len(vals))], 4)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="agi-proof/external-eval/factcheck-full-r1.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    rows = transform_cases(args.data)
    domains = sorted({r["domain"] for r in rows})
    n_acc = sum(r["accepted"] for r in rows); n_cor = sum(r["correct"] for r in rows)
    print(f"n={len(rows)} accepted={n_acc} correct={n_cor} domains={domains}", file=sys.stderr)

    # hidden-state featurizer (real MLX) + linear-stub baseline
    print("loading MLX model ...", file=sys.stderr)
    hidden_fz = make_hidden_featurizer(args.model)
    lin_fz = lambda txt: featurize_text(txt)

    he, hc, hcorr, hgap = lodo_oof(rows, hidden_fz)
    le, lc, lcorr, lgap = lodo_oof(rows, lin_fz)
    # align labels (both use same row order/domain loop -> same correctness sequence)
    assert hcorr == lcorr, "fold alignment mismatch"
    labels = hcorr

    hidden_vs_chance = bootstrap_vs_chance([-e for e in he], labels, args.n_boot, args.seed)
    linear_vs_chance = bootstrap_vs_chance([-e for e in le], labels, args.n_boot, args.seed)
    paired = paired_bootstrap_delta([-e for e in he], [-e for e in le], labels, args.n_boot, args.seed)

    ece_hidden = ece(hc, labels)
    goodhart_ok = (hgap is not None and hgap <= 0.15)
    hidden_beats_chance = hidden_vs_chance["ci95"][0] > 0.5
    hidden_beats_linear = paired["ci95"][0] > 0.0

    report = {
        "schema": "sophia.o2_energy_hidden.v1",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "hiddenStateFeaturizerReady": True, "featurizerModel": args.model,
        "embedBackend": "mlx-hidden-state:" + args.model,
        "data": args.data, "n": len(rows), "domains": domains,
        "nAccepted": n_acc, "nCorrect": n_cor,
        "evaluation": "leave-one-domain-out out-of-fold (no in-sample optimism)",
        "hiddenAUROCvsChance": hidden_vs_chance,
        "linearStubAUROCvsChance": linear_vs_chance,
        "pairedHiddenMinusLinear": paired,
        "eceHidden": round(ece_hidden, 4),
        "goodhartGap": round(hgap, 4) if hgap is not None else None,
        "gates": {
            "hiddenBeatsChance(CI>0.5)": hidden_beats_chance,
            "hiddenBeatsLinearStub(deltaCI>0)": hidden_beats_linear,
            "goodhartGap<=0.15": goodhart_ok,
        },
        "verdict": ("energy_head_meets_gate" if (hidden_beats_chance and goodhart_ok)
                    else "gate_not_met"),
        "note": ("Real MLX hidden-state energy head (build_hidden_state_featurizer seam, no stub); "
                 "trained on verifier `accepted`, audited vs ground-truth `correct` via LODO out-of-fold. "
                 "verified_traces.jsonl (brief's named source) absent -> factcheck pack used. "
                 "No model weights updated."),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
