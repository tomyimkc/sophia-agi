#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibration-verifier scaling evaluation (deterministic machinery; PENDING result).

Turns the T3 measurement plan into a runnable, gated instrument — the same pattern as
tools/run_dikaiosyne_eval.py.

The claim under test (agi-proof/benchmark-results/calibration-verifier/measurement_spec.json):
a separate, trace-feature VERIFIER (semantic-entropy + neighbour-corroboration +
author/source-confidence, with NO access to the gold answer) predicts answer correctness
BETTER (delta AUROC >= +0.05, CI excluding 0) than the existing provenance-grounded
confidence prior, WITHOUT worsening calibration (delta ECE <= 0).

Arms (all scored on the SAME traces):
  * provenance-confidence-baseline — the cheap source-quality prior (the current weak,
    ~0.52-balanced-accuracy, non-monotonic predictor per the failure ledger).
  * trace-feature-verifier — a tiny deterministic combiner FIT on a disjoint train split
    over the three trace features, applied to the held-out test split (the candidate).
  * raw-model-logprob-baseline — an overconfident, scale-degrading control.

Feature extraction runs the REAL feature code (agent/semantic_entropy.semantic_entropy and
agent/corroboration.corroborated_confidence) over each synthetic trace, so the harness
exercises the production feature path — only the traces and their correctness labels are
synthetic in --mock mode.

Modes (all offline):
  * --mock: build a deterministic synthetic trace set, fit the verifier on the train split,
    score all three arms on the test split, print per-arm AUROC/ECE + the paired delta AUROC
    with a bootstrap 95% CI. Exercises the AUROC/ECE/delta math in CI; NOT evidence (a
    synthetic trace is not a real labelled answer; labels are not >= 2 judge families).
  * --emit-pending: write the committed not-run / NO-GO artifact.
  * --model <spec>: refuse rather than fabricate; result stays PENDING.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.corroboration import Evidence, corroborated_confidence  # noqa: E402
from agent.semantic_entropy import semantic_entropy  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "calibration-verifier"
SPEC_PATH = RESULTS_DIR / "measurement_spec.json"
PENDING_PATH = RESULTS_DIR / "calibration-verifier-scaling.PENDING.public-report.json"

DELTA_AUROC_THRESHOLD = 0.05  # GO floor (mirrors the measurement_spec magnitudeRule)


# --------------------------------------------------------------------------- #
# Metrics (deterministic, stdlib). AUROC via Mann–Whitney; ECE via equal-width bins.
# --------------------------------------------------------------------------- #
def auroc(scores: list[float], labels: list[int]) -> float | None:
    """P(score(correct) > score(incorrect)), ties counted as 0.5. None if a class is absent.

    labels: 1 = answer correct (positive), 0 = incorrect. This is the standard rank-AUROC,
    equivalent to the normalised Mann–Whitney U statistic."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return round(wins / (len(pos) * len(neg)), 6)


def ece(probs: list[float], labels: list[int], *, bins: int = 10) -> float | None:
    """Expected calibration error: weighted |confidence - accuracy| over equal-width bins."""
    if not probs:
        return None
    n = len(probs)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        # last bin is closed on the right so prob == 1.0 lands somewhere
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(conf - acc)
    return round(total, 6)


def bootstrap_ci_auroc_delta(
    scores_a: list[float], scores_b: list[float], labels: list[int],
    *, iters: int = 2000, alpha: float = 0.05, seed: int = 0,
) -> list[float]:
    """Percentile CI for AUROC(a) - AUROC(b) via PAIRED trace resampling (same resampled
    index set scored by both arms each draw, so the CI reflects the within-trace pairing)."""
    rng = random.Random(seed)
    n = len(labels)
    deltas: list[float] = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        ya = [labels[i] for i in idx]
        a = auroc([scores_a[i] for i in idx], ya)
        b = auroc([scores_b[i] for i in idx], ya)
        if a is not None and b is not None:
            deltas.append(a - b)
    if not deltas:
        return [None, None]
    deltas.sort()
    lo = deltas[int((alpha / 2) * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int((1 - alpha / 2) * len(deltas)))]
    return [round(lo, 6), round(hi, 6)]


# --------------------------------------------------------------------------- #
# Synthetic traces (mock only) + REAL feature extraction.
# --------------------------------------------------------------------------- #
def _synthetic_traces(n: int, *, seed: int = 0) -> list[dict]:
    """A deterministic trace set with a latent correctness bit and features that are
    INFORMATIVE-but-noisy about it (so a fit verifier can beat a noisier single prior).

    Each trace carries:
      - samples: paraphrase set for semantic entropy (more agreement when correct)
      - evidence: (confidence) list for corroboration (stronger support when correct)
      - authorConfidence: the source's self-rated confidence (weak, miscalibrated prior)
    """
    rng = random.Random(seed)
    traces: list[dict] = []
    for i in range(n):
        correct = 1 if rng.random() < 0.5 else 0
        # When correct, samples cluster (low entropy); when wrong, they scatter.
        base = f"the answer is option {rng.choice('ABCD')}"
        if correct:
            samples = [base, base, base, f"{base} indeed"]
        else:
            samples = [f"the answer is option {rng.choice('ABCD')}" for _ in range(4)]
        # Corroboration evidence: higher support prob when correct, with noise.
        k = 3
        confs = [min(0.99, max(0.01, (0.65 if correct else 0.45) + rng.uniform(-0.2, 0.2))) for _ in range(k)]
        evidence = [Evidence(source_id=f"s{i}_{j}", confidence=c) for j, c in enumerate(confs)]
        # Author confidence: deliberately weak/miscalibrated (overconfident on wrong too).
        author = min(0.99, max(0.01, 0.7 + rng.uniform(-0.25, 0.25)))
        traces.append({"id": f"tr-{i}", "correct": correct, "samples": samples,
                       "evidence": evidence, "authorConfidence": round(author, 4)})
    return traces


def _as_evidence(evidence: list) -> list:
    """Accept evidence as Evidence objects (synthetic traces) OR plain confidence floats
    (generated traces from tools/run_calibration_traces.py). Returns a list[Evidence]."""
    out = []
    for i, e in enumerate(evidence or []):
        if isinstance(e, Evidence):
            out.append(e)
        else:
            try:
                out.append(Evidence(source_id=f"ev{i}", confidence=float(e)))
            except (TypeError, ValueError):
                continue
    return out


def extract_features(trace: dict) -> dict:
    """Run the REAL feature code over a trace. Uses NO access to the correctness label.

    Evidence may be Evidence objects (synthetic) or confidence floats (generated traces);
    an empty/absent evidence list falls back to the 0.5 corroboration prior (honest)."""
    se = semantic_entropy(trace["samples"], mode="lexical")["entropy"]  # 0=certain, 1=scattered
    ev = _as_evidence(trace.get("evidence", []))
    corr = corroborated_confidence(ev, method="logodds") if ev else 0.5
    return {
        "agreement": round(1.0 - se, 6),          # high when paraphrases agree
        "corroboration": round(corr, 6),
        "authorConfidence": float(trace["authorConfidence"]),
    }


def _fit_verifier(train: list[dict]) -> dict:
    """Fit a tiny deterministic linear combiner on the train split (no gold-answer feature).

    Returns per-feature weights = the feature's correlation sign-and-strength with
    correctness on the train split. Deterministic; no external deps."""
    feats = [extract_features(t) for t in train]
    ys = [t["correct"] for t in train]
    keys = ("agreement", "corroboration", "authorConfidence")
    weights: dict[str, float] = {}
    for kkey in keys:
        xs = [f[kkey] for f in feats]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
        var = sum((x - mx) ** 2 for x in xs) / len(xs) or 1e-9
        weights[kkey] = cov / var
    return {"weights": weights}


def _verifier_score(trace: dict, model: dict) -> float:
    f = extract_features(trace)
    raw = sum(model["weights"][k] * f[k] for k in model["weights"])
    return 1.0 / (1.0 + pow(2.718281828, -raw))  # squash to [0,1]


def _provenance_score(trace: dict) -> float:
    """The cheap prior arm: source confidence only (corroboration of the evidence)."""
    return corroborated_confidence(trace["evidence"], method="logodds")


def _raw_logprob_score(trace: dict) -> float:
    """Overconfident control: the author's self-rated confidence, pushed toward 1."""
    return min(0.999, trace["authorConfidence"] ** 0.5)


# --------------------------------------------------------------------------- #
# Run + verdict.
# --------------------------------------------------------------------------- #
def run_mock(*, n: int = 400, seed: int = 0) -> dict:
    traces = _synthetic_traces(n, seed=seed)
    split = len(traces) // 2
    train, test = traces[:split], traces[split:]
    model = _fit_verifier(train)
    labels = [t["correct"] for t in test]

    verifier_scores = [_verifier_score(t, model) for t in test]
    prov_scores = [_provenance_score(t) for t in test]
    raw_scores = [_raw_logprob_score(t) for t in test]

    arms = {
        "trace-feature-verifier": {"auroc": auroc(verifier_scores, labels), "ece": ece(verifier_scores, labels)},
        "provenance-confidence-baseline": {"auroc": auroc(prov_scores, labels), "ece": ece(prov_scores, labels)},
        "raw-model-logprob-baseline": {"auroc": auroc(raw_scores, labels), "ece": ece(raw_scores, labels)},
    }
    delta_auroc = round((arms["trace-feature-verifier"]["auroc"] or 0) -
                        (arms["provenance-confidence-baseline"]["auroc"] or 0), 6)
    delta_ece = round((arms["trace-feature-verifier"]["ece"] or 0) -
                      (arms["provenance-confidence-baseline"]["ece"] or 0), 6)
    ci = bootstrap_ci_auroc_delta(verifier_scores, prov_scores, labels, seed=seed)
    # The no-answer-leakage audit is a REAL, runnable guardrail (tools/audit_feature_leakage.py):
    # run it on the actual extractor so the leakage pillar reflects the audit, not a promise.
    from tools.audit_feature_leakage import audit_t3_extractor
    leakage = audit_t3_extractor(n=min(len(traces), 100), seed=seed)
    verdict = gate_verdict(real_corpus=False, judge_families=1, leakage_audited=leakage["passed"],
                           delta_auroc=delta_auroc, delta_auroc_ci=ci, delta_ece=delta_ece,
                           base_sizes=1)
    return {
        "mode": "mock",
        "n_test": len(test),
        "arms": arms,
        "deltaAUROC": delta_auroc,
        "deltaAUROCCI95": ci,
        "deltaECE": delta_ece,
        "leakageAudit": {"passed": leakage["passed"], "verdict": leakage["verdict"]},
        "verdict": verdict["verdict"],
        "criticalFailures": verdict["criticalFailures"],
        "boundary": "synthetic traces + synthetic labels — machinery proof of the AUROC/ECE/delta "
                    "math, NOT evidence about real answer correctness.",
    }


def _provenance_score_features(f: dict) -> float:
    """Provenance prior arm over generated traces: corroboration feature only (the cheap,
    source-quality prior — what agent/grounded_confidence.py pools)."""
    return f["corroboration"]


def run_real(traces_path: Path, *, seed: int = 0, judge_families: int = 1) -> dict:
    """Score a REAL generated trace corpus (tools/run_calibration_traces.py output).

    Splits questions into disjoint train/test, fits the verifier on train (all models), and
    reports per-model AUROC/ECE on the test split (the >=3-size scaling axis) plus the delta vs
    the provenance prior. Correctness labels come from the traces (gold-match = family 1); the
    >=2-family requirement and the real-corpus pillar are enforced by gate_verdict downstream.
    """
    traces = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not traces:
        raise ValueError(f"no traces in {traces_path}")
    models = sorted({t["model"] for t in traces})
    qids = sorted({t["id"] for t in traces})
    # Deterministic held-out split on QUESTION id (so a question never spans train and test).
    rng = random.Random(seed)
    shuffled = qids[:]
    rng.shuffle(shuffled)
    cut = max(1, len(shuffled) // 2)
    train_qids, test_qids = set(shuffled[:cut]), set(shuffled[cut:])

    feats = {id(t): extract_features(t) for t in traces}
    train = [t for t in traces if t["id"] in train_qids]
    train_model = _fit_verifier_from_features([(feats[id(t)], t["correct"]) for t in train])

    per_model: dict[str, dict] = {}
    for m in models:
        test = [t for t in traces if t["model"] == m and t["id"] in test_qids]
        if not test:
            continue
        labels = [t["correct"] for t in test]
        v_scores = [_score_from_features(feats[id(t)], train_model) for t in test]
        p_scores = [_provenance_score_features(feats[id(t)]) for t in test]
        per_model[m] = {
            "n": len(test),
            "verifier": {"auroc": auroc(v_scores, labels), "ece": ece(v_scores, labels)},
            "provenance": {"auroc": auroc(p_scores, labels), "ece": ece(p_scores, labels)},
        }

    # Scaling: ECE trend of the verifier across models (in given order = size order by convention).
    eces = [(m, per_model[m]["verifier"]["ece"]) for m in models if m in per_model and per_model[m]["verifier"]["ece"] is not None]
    ece_slope = None
    if len(eces) >= 2:
        ys = [e for _, e in eces]
        ece_slope = round((ys[-1] - ys[0]) / max(1, len(ys) - 1), 6)  # >0 = calibration worsens with size

    # Pooled delta over the full test split (all models).
    test_all = [t for t in traces if t["id"] in test_qids]
    labels_all = [t["correct"] for t in test_all]
    v_all = [_score_from_features(feats[id(t)], train_model) for t in test_all]
    p_all = [_provenance_score_features(feats[id(t)]) for t in test_all]
    delta_auroc = round((auroc(v_all, labels_all) or 0) - (auroc(p_all, labels_all) or 0), 6)
    delta_ece = round((ece(v_all, labels_all) or 0) - (ece(p_all, labels_all) or 0), 6)
    ci = bootstrap_ci_auroc_delta(v_all, p_all, labels_all, seed=seed)

    from tools.audit_feature_leakage import audit
    leak = audit(test_all, extract_features)
    verdict = gate_verdict(real_corpus=False, judge_families=judge_families, leakage_audited=leak["passed"],
                           delta_auroc=delta_auroc, delta_auroc_ci=ci, delta_ece=delta_ece,
                           base_sizes=len(per_model))
    return {
        "mode": "real-traces",
        "tracesPath": str(traces_path),
        "models": models,
        "nTrain": len(train), "nTest": len(test_all),
        "perModel": per_model,
        "eceSlopeAcrossModels": ece_slope,
        "deltaAUROC": delta_auroc,
        "deltaAUROCCI95": ci,
        "deltaECE": delta_ece,
        "leakageAudit": {"passed": leak["passed"], "verdict": leak["verdict"]},
        "verdict": verdict["verdict"],
        "go": verdict["go"],
        "criticalFailures": verdict["criticalFailures"],
        "canClaimAGI": False,
        "boundary": "real generated traces with gold-match labels (family 1). The >=2-judge-family "
                    "and real-corpus pillars remain until a judge panel + decontaminated external "
                    "corpus are supplied; gate stays NO-GO accordingly. canClaimAGI:false.",
    }


def _fit_verifier_from_features(rows: list) -> dict:
    """Fit the linear combiner from (features, label) rows (shared by mock + real paths)."""
    keys = ("agreement", "corroboration", "authorConfidence")
    ys = [y for _, y in rows]
    my = sum(ys) / len(ys) if ys else 0.0
    weights: dict[str, float] = {}
    for kkey in keys:
        xs = [f[kkey] for f, _ in rows]
        mx = sum(xs) / len(xs) if xs else 0.0
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs) if xs else 0.0
        var = (sum((x - mx) ** 2 for x in xs) / len(xs)) if xs else 0.0
        weights[kkey] = cov / (var or 1e-9)
    return {"weights": weights}


def _score_from_features(f: dict, model: dict) -> float:
    raw = sum(model["weights"][k] * f[k] for k in model["weights"])
    return 1.0 / (1.0 + pow(2.718281828, -raw))


def gate_verdict(*, real_corpus: bool, judge_families: int, leakage_audited: bool,
                 delta_auroc: float | None, delta_auroc_ci: list | None,
                 delta_ece: float | None, base_sizes: int) -> dict:
    """GO/NO-GO over the pre-registered pillars. Offline this is always NO-GO."""
    failures: list[str] = []
    if not real_corpus:
        failures.append("no_real_corpus: requires a decontaminated, gold-labelled trace corpus (synthetic traces are not evidence)")
    if judge_families < 2:
        failures.append("labels_not_2family: correctness labels are not >= 2 independent judge families (kappa >= 0.40)")
    if not leakage_audited:
        failures.append("no_leakage_audit: the no-answer-leakage feature audit has not been run")
    ci = delta_auroc_ci or [None, None]
    excludes_zero = ci[0] is not None and ci[0] > 0
    if not (excludes_zero and (delta_auroc or 0) >= DELTA_AUROC_THRESHOLD):
        failures.append(f"no_effect: delta AUROC must be >= {DELTA_AUROC_THRESHOLD} with a 95% CI excluding 0")
    if delta_ece is not None and delta_ece > 0:
        failures.append("calibration_guardrail: delta ECE > 0 (the verifier worsened calibration)")
    if base_sizes < 3:
        failures.append("scaling_not_3sizes: the scaling-monotonicity headline needs >= 3 base sizes on identical traces")
    return {
        "verdict": "NO-GO" if failures else "GO",
        "go": not failures,
        "criticalFailures": failures,
        "boundary": (
            "Calibration-verifier is candidate infrastructure. GO requires a real decontaminated "
            "trace corpus, >= 2 independent judge families for correctness labels, a no-leakage "
            "feature audit, delta AUROC >= 0.05 with a CI excluding 0, delta ECE <= 0, and (for the "
            "scaling headline) >= 3 base sizes. canClaimAGI:false."
        ),
    }


def build_pending_artifact() -> dict:
    verdict = gate_verdict(real_corpus=False, judge_families=1, leakage_audited=False,
                           delta_auroc=None, delta_auroc_ci=None, delta_ece=None, base_sizes=1)
    return {
        "experimentId": "calibration-verifier-scaling",
        "status": "not_run",
        "verdict": verdict["verdict"],
        "go": False,
        "canClaimAGI": False,
        "claimCeiling": "candidate_only; canClaimAGI:false",
        "headline": "PENDING — machinery only; no real labelled-trace run has been performed",
        "harness": "tools/run_calibration_verifier_eval.py",
        "preregistration": "agi-proof/benchmark-results/calibration-verifier/measurement_spec.json",
        "arms": {
            "trace-feature-verifier": {"status": "not_run", "reason": "needs a real labelled trace corpus + a fit on a disjoint split"},
            "provenance-confidence-baseline": {"status": "not_run", "reason": "needs the live grounded path on the same corpus"},
            "raw-model-logprob-baseline": {"status": "not_run", "reason": "needs base-model logprobs on the same corpus"},
        },
        "delta": None,
        "criticalFailures": verdict["criticalFailures"],
        "note": (
            "Intentionally PENDING. The --mock mode exercises the AUROC/ECE/delta+CI math over "
            "synthetic traces (tests/test_calibration_verifier_eval.py), but synthetic traces are "
            "not evidence: no claim about real answer correctness is made. Promotion needs a "
            "decontaminated gold-labelled trace corpus, >= 2 independent judge families, a "
            "no-answer-leakage feature audit, and (for the scaling headline) >= 3 base sizes — see "
            "the measurement_spec and the calibration-verifier-scaling row in agi-proof/failure-ledger.md."
        ),
    }


def emit_pending() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = build_pending_artifact()
    PENDING_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return PENDING_PATH


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock", action="store_true", help="run the deterministic synthetic-trace machinery check")
    ap.add_argument("--traces", type=Path, default=None, help="score a REAL generated trace corpus (run_calibration_traces.py output)")
    ap.add_argument("--judge-families", type=int, default=1, help="number of independent correctness-judge families behind the labels")
    ap.add_argument("--out", type=Path, default=None, help="write the report JSON here (for --traces)")
    ap.add_argument("--emit-pending", action="store_true", help="write the committed not-run / NO-GO artifact")
    ap.add_argument("--model", default=None, help="real-model spec (refused offline; result stays PENDING)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.traces:
        report = run_real(args.traces, seed=args.seed, judge_families=args.judge_families)
        out = json.dumps(report, indent=2, ensure_ascii=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(out + "\n", encoding="utf-8")
            print(f"wrote {args.out} (verdict={report['verdict']})")
        else:
            print(out)
        return 0
    if args.model:
        print(json.dumps({
            "status": "refused",
            "reason": "real-model run needs a decontaminated gold-labelled trace corpus + >= 2 judge "
                      "families + a no-leakage audit; not fabricated here. Result stays PENDING / NO-GO.",
            "verdict": "NO-GO", "canClaimAGI": False,
        }, indent=2))
        return 0
    if args.emit_pending:
        path = emit_pending()
        print(f"wrote {path}")
        return 0
    if args.mock:
        print(json.dumps(run_mock(n=args.n, seed=args.seed), indent=2, ensure_ascii=False))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
