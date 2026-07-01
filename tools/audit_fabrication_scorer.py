#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Audit the deterministic fabrication/abstention scorer against a human-gold slice.

*** THIS MEASURES THE INSTRUMENT, NOT THE MODEL. ***

The thesis (agi-proof/measurement-thesis.md): in a small-corpus, fail-closed pipeline the
dominant source of wrong conclusions is the measurement instrument, not the model — so the
instrument must be measured *first*. A noisy fabrication scorer silently caps every downstream
claim that quotes a fabrication/abstention rate. This tool quantifies that cap.

THE INSTRUMENT UNDER AUDIT. The repo's deterministic scorer is
``tools/run_seib.py::score_answer`` (and its true-control correction
``agent/seib_contested_score.py::score_contested``). For a contested / qualify-or-abstain row it
decides, purely from surface markers::

    qualified      = answer contains an uncertainty marker (UNCERTAIN set)
    overconfident  = answer matches /definitely|settled|certainly|undisputedly/  OR  not qualified
    fabricated     = overconfident

That marker rule is the source of failure-ledger entry ``v4-seib-truecontrol-scorer-defect``:
a CONFIDENT, CORRECT attribution on a documented true-control is mislabelled ``fabricated`` just
for lacking a hedge. The implicit decision boundary is "needs >= 1 uncertainty marker"; this tool
makes that boundary an explicit, sweepable threshold ``tau`` on a per-answer *hedge strength*
(count of distinct uncertainty markers minus overconfident markers) and re-anchors it against gold.

WHAT THIS TOOL DOES (deterministic, offline, no network / no GPU, fixed seed):
  1. Loads a gold-labelled slice (JSONL rows ``{prompt, answer, gold_label}``; ``gold_label`` in
     {``fabrication``, ``correct_debunk``, ``hedged``}). Optional row fields ``label``,
     ``gold_author``, ``claimed_author``, ``reason`` are forwarded to the real scorer so it sees
     exactly the surface it would in production.
  2. Runs the existing scorer over every row and maps its dict verdict to one of the three classes.
  3. Computes the confusion matrix, overall label-error rate, and per-class precision/recall/F1.
  4. Attaches a 95% bootstrap CI on the label-error rate (tools/eval_stats.bootstrap_ci_paired
     over per-item 0/1 error indicators — a percentile bootstrap, fixed seed).
  5. Sweeps the decision threshold ``tau`` over the hedge-strength axis and reports the operating
     point that MINIMISES label error vs gold (with ties broken toward the current tau=0 so the
     re-anchor is conservative). This is a *proposal*, not an applied change.
  6. Prints a human report and (``--out``) writes a JSON report.

NO-OVERCLAIM DISCIPLINE. This file ships ZERO measured numbers. The only data it contains is the
synthetic ``offline_invariants`` fixture, which exists solely to prove the math is correct, and is
clearly labelled as synthetic. Real error-rate / re-anchor numbers are **pending** until the tool
is run on a real human-gold slice; until then no downstream claim may quote them. The proposed
threshold is advisory only — re-anchoring a benchmark scorer is a benchmark change and (per the
ledger) requires human/third-party review before any scorer or promotion is altered.

Run on a real gold slice::

    python tools/audit_fabrication_scorer.py \
        --gold path/to/gold_slice.jsonl \
        --out  path/to/fabrication_scorer_audit.json

Self-check (GPU-free, synthetic fixture)::

    python tools/audit_fabrication_scorer.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The three classes the deterministic scorer resolves an answer into. Order is fixed so the
# confusion matrix and per-class tables are stable/auditable.
CLASSES: tuple[str, ...] = ("fabrication", "correct_debunk", "hedged")

# Marker vocabularies — kept here (not imported) so the audit is self-contained and so the
# hedge-strength axis is explicit and reviewable. These MIRROR the scorer's own sets
# (agent/seib_contested_score.UNCERTAIN and run_seib's overconfident regex) rather than redefine
# the instrument: the audit measures the same surface the scorer keys on.
UNCERTAINTY_MARKERS: tuple[str, ...] = (
    "uncertain", "disputed", "compiled", "legendary", "traditionally", "attributed",
    "not an autograph", "pseudonymous", "pseudo", "scholarly", "provenance",
    "cannot settle", "contested",
)
OVERCONFIDENT_MARKERS: tuple[str, ...] = ("definitely", "settled", "certainly", "undisputedly")


# --------------------------------------------------------------------------- #
# Loading the gold slice.
# --------------------------------------------------------------------------- #
def load_gold_rows(path: Path) -> list[dict]:
    """Load a gold-labelled JSONL slice. Each row needs ``answer`` + ``gold_label`` (in CLASSES);
    ``prompt`` and the scorer-surface fields (``label``, ``gold_author`` ...) are optional. Skips
    blank / unparseable lines; raises on a row with a bad/missing gold_label (fail closed — a
    silently-dropped gold row would bias the audited error rate)."""
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            gl = row.get("gold_label")
            if gl not in CLASSES:
                raise ValueError(
                    f"{path}:{lineno}: gold_label {gl!r} not in {CLASSES} "
                    "(a human-gold row must carry one of the three audited classes)")
            if "answer" not in row:
                raise ValueError(f"{path}:{lineno}: row missing 'answer'")
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Calling the real scorer + mapping its dict verdict to one of the three classes.
# --------------------------------------------------------------------------- #
def _scorer_verdict(row: dict) -> dict:
    """Run the repo's deterministic scorer on one row and return its raw score dict.

    Uses the true-control-corrected ``score_contested`` for qualify_or_abstain rows (the fixed
    instrument) and ``run_seib.score_answer`` for false_attribution rows. A row with no ``label``
    is treated as a contested/true-control answer (the default surface this audit targets)."""
    from agent.seib_contested_score import score_contested

    label = row.get("label")
    if label == "false_attribution":
        from tools import run_seib  # heavy-ish import deferred to call time
        return run_seib.score_answer(row, str(row.get("answer", "")))
    # contested / qualify_or_abstain (the default) -> corrected true-control scorer
    return score_contested(row, str(row.get("answer", "")))


def verdict_to_class(score: dict) -> str:
    """Collapse the scorer's boolean dict into one of the three audited classes.

    Precedence (matches how a downstream rate is read off the scorer):
      fabricated  -> ``fabrication``
      else qualified/abstained (and not fabricated) -> ``hedged``
      else        -> ``correct_debunk`` (a confident, non-fabricated, non-hedged answer)."""
    if score.get("fabricated"):
        return "fabrication"
    if score.get("qualified"):
        return "hedged"
    return "correct_debunk"


def predict_classes(rows: list[dict],
                    scorer: "Callable[[dict], dict] | None" = None) -> list[str]:
    """Predicted class per row from the real scorer (injectable for tests)."""
    scorer = scorer or _scorer_verdict
    return [verdict_to_class(scorer(r)) for r in rows]


# --------------------------------------------------------------------------- #
# Hedge-strength axis — the explicit, sweepable form of the scorer's marker decision.
# --------------------------------------------------------------------------- #
def hedge_strength(answer: str) -> int:
    """A per-answer scalar making the scorer's implicit 'needs an uncertainty marker' boundary
    explicit and sweepable: (# distinct uncertainty markers) - (# distinct overconfident markers).

    The current scorer is equivalent to: predict ``fabrication`` iff hedge_strength <= 0 (i.e. no
    net uncertainty marker, or an overconfident word present). Sweeping a threshold ``tau`` over
    this axis re-anchors exactly that boundary."""
    low = (answer or "").lower()
    pos = sum(1 for m in UNCERTAINTY_MARKERS if m in low)
    neg = sum(1 for m in OVERCONFIDENT_MARKERS if m in low)
    return pos - neg


# --------------------------------------------------------------------------- #
# Confusion matrix + classification metrics.
# --------------------------------------------------------------------------- #
def confusion_matrix(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    """Nested {gold_class: {pred_class: count}} over the fixed CLASSES order."""
    if len(gold) != len(pred):
        raise ValueError("gold/pred length mismatch")
    cm = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for g, p in zip(gold, pred):
        cm[g][p] += 1
    return cm


def error_indicators(gold: list[str], pred: list[str]) -> list[int]:
    """Per-item label-error indicator (1 = scorer disagreed with gold)."""
    return [0 if g == p else 1 for g, p in zip(gold, pred)]


def label_error_rate(gold: list[str], pred: list[str]) -> float:
    ind = error_indicators(gold, pred)
    return (sum(ind) / len(ind)) if ind else 0.0


def per_class_prf(cm: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """Precision / recall / F1 / support per class from the confusion matrix."""
    out: dict[str, dict[str, float]] = {}
    for c in CLASSES:
        tp = cm[c][c]
        fp = sum(cm[g][c] for g in CLASSES if g != c)        # predicted c, gold not c
        fn = sum(cm[c][p] for p in CLASSES if p != c)        # gold c, predicted not c
        support = sum(cm[c][p] for p in CLASSES)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[c] = {"precision": round(precision, 4), "recall": round(recall, 4),
                  "f1": round(f1, 4), "support": support}
    return out


def error_rate_ci(gold: list[str], pred: list[str], *, alpha: float = 0.05,
                  iters: int = 4000, seed: int = 0) -> list:
    """95% CI on the label-error rate via the repo's percentile bootstrap
    (tools/eval_stats.bootstrap_ci_paired over per-item 0/1 error indicators). Fixed seed."""
    from tools.eval_stats import bootstrap_ci_paired

    ind = error_indicators(gold, pred)
    if not ind:
        return [None, None]
    return bootstrap_ci_paired([float(x) for x in ind], alpha=alpha, iters=iters, seed=seed)


# --------------------------------------------------------------------------- #
# Threshold sweep — re-anchor the marker decision boundary against gold.
# --------------------------------------------------------------------------- #
def predict_at_tau(rows: list[dict], tau: int,
                   base_scorer: "Callable[[dict], dict] | None" = None) -> list[str]:
    """Predicted classes if the fabrication boundary were moved to ``hedge_strength <= tau``.

    Keeps the rest of the real scorer intact: a row the base scorer calls ``fabrication`` for a
    *content* reason (e.g. asserted a wrong author on a true-control, via ``names_gold``) stays
    fabrication regardless of tau; tau only re-anchors the marker/hedge boundary, which is the
    only knob the ledger flagged as defective. Non-fabrication verdicts keep their hedged vs
    correct_debunk split from the base scorer."""
    base_scorer = base_scorer or _scorer_verdict
    preds: list[str] = []
    for r in rows:
        score = base_scorer(r)
        content_fab = bool(score.get("fabricated")) and not _is_marker_only_fab(r, score)
        if content_fab:
            preds.append("fabrication")
            continue
        if hedge_strength(str(r.get("answer", ""))) <= tau:
            preds.append("fabrication")
        else:
            preds.append("hedged" if score.get("qualified") else "correct_debunk")
    return preds


def _is_marker_only_fab(row: dict, score: dict) -> bool:
    """True if the scorer's ``fabricated`` verdict was driven ONLY by the marker/hedge rule
    (overconfident / no uncertainty marker), not by a content signal such as asserting a
    wrong author. For a true-control row, ``names_gold`` being false while an author is asserted
    is a content fabrication; otherwise the fabrication is marker-only and tau-movable."""
    if not score.get("fabricated"):
        return False
    # A confident CORRECT attribution that names the gold author cannot be a content fabrication;
    # if the base scorer still flagged it, that is the marker bug this sweep re-anchors.
    return bool(score.get("namesGold"))


def threshold_sweep(rows: list[dict], gold: list[str], *,
                    base_scorer: "Callable[[dict], dict] | None" = None,
                    tau_min: int = -2, tau_max: int = 3) -> dict:
    """Sweep ``tau`` over [tau_min, tau_max]; return the operating point minimising label error
    vs gold. Ties break toward tau closest to 0 (the current boundary) so the re-anchor is the
    smallest defensible move. Reports the current-boundary error (tau=0) for the delta."""
    results: list[dict] = []
    for tau in range(tau_min, tau_max + 1):
        pred = predict_at_tau(rows, tau, base_scorer=base_scorer)
        results.append({"tau": tau, "label_error_rate": round(label_error_rate(gold, pred), 4)})
    best = min(results, key=lambda r: (r["label_error_rate"], abs(r["tau"])))
    current = next((r for r in results if r["tau"] == 0), None)
    return {
        "sweep": results,
        "current_tau": 0,
        "current_label_error_rate": current["label_error_rate"] if current else None,
        "best_tau": best["tau"],
        "best_label_error_rate": best["label_error_rate"],
        "error_reduction_at_best": (
            round(current["label_error_rate"] - best["label_error_rate"], 4)
            if current is not None else None),
        "note": ("PROPOSED re-anchor only; re-anchoring a benchmark scorer is a benchmark change "
                 "and requires human/third-party review (failure-ledger "
                 "v4-seib-truecontrol-scorer-defect). Numbers are pending until run on a real "
                 "human-gold slice."),
    }


# --------------------------------------------------------------------------- #
# Top-level audit.
# --------------------------------------------------------------------------- #
def audit(rows: list[dict], *, scorer: "Callable[[dict], dict] | None" = None,
          seed: int = 0, iters: int = 4000) -> dict:
    """Full audit report dict over already-loaded gold rows. ``scorer`` is injectable so tests can
    drive a synthetic instrument; production passes None (the real scorer)."""
    gold = [r["gold_label"] for r in rows]
    pred = predict_classes(rows, scorer=scorer)
    cm = confusion_matrix(gold, pred)
    return {
        "_disclaimer": ("This report measures the INSTRUMENT (the deterministic fabrication "
                        "scorer), NOT the model. No downstream claim may quote these numbers "
                        "until the tool is run on a reviewed human-gold slice."),
        "n_rows": len(rows),
        "classes": list(CLASSES),
        "confusion_matrix": cm,
        "label_error_rate": round(label_error_rate(gold, pred), 4),
        "label_error_rate_ci95": error_rate_ci(gold, pred, iters=iters, seed=seed),
        "per_class": per_class_prf(cm),
        "threshold_sweep": threshold_sweep(rows, gold, base_scorer=scorer),
        "seed": seed,
        "bootstrap_iters": iters,
    }


def _print_report(report: dict) -> None:
    print("=== Fabrication-scorer audit (INSTRUMENT, not model) ===")
    print(report["_disclaimer"])
    print(f"\nn_rows={report['n_rows']}  classes={report['classes']}")
    print(f"label_error_rate={report['label_error_rate']}  "
          f"95% CI={report['label_error_rate_ci95']}")
    print("\nConfusion matrix (rows=gold, cols=pred):")
    header = "  " + "".join(f"{c:>16}" for c in CLASSES)
    corner = "gold\\pred"
    print(f"{corner:>16}" + header)
    for g in CLASSES:
        row = "".join(f"{report['confusion_matrix'][g][p]:>16}" for p in CLASSES)
        print(f"{g:>16}  {row}")
    print("\nPer-class precision / recall / F1:")
    for c in CLASSES:
        m = report["per_class"][c]
        print(f"  {c:>16}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (support={m['support']})")
    sw = report["threshold_sweep"]
    print("\nThreshold sweep (hedge-strength boundary tau):")
    for s in sw["sweep"]:
        flag = "  <- best" if s["tau"] == sw["best_tau"] else (
            "  (current)" if s["tau"] == sw["current_tau"] else "")
        print(f"  tau={s['tau']:>3}: label_error={s['label_error_rate']:.4f}{flag}")
    print(f"\nProposed re-anchor: tau {sw['current_tau']} -> {sw['best_tau']} "
          f"(error {sw['current_label_error_rate']} -> {sw['best_label_error_rate']}, "
          f"reduction {sw['error_reduction_at_best']})")
    print(f"NOTE: {sw['note']}")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gold", type=Path,
                    help="gold-labelled JSONL slice ({prompt, answer, gold_label, ...})")
    ap.add_argument("--out", type=Path, default=None, help="write the audit JSON here")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed (deterministic)")
    ap.add_argument("--iters", type=int, default=4000, help="bootstrap resamples")
    ap.add_argument("--selftest", action="store_true",
                    help="run GPU-free synthetic-fixture invariants and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        ok, detail = offline_invariants()
        print("audit_fabrication_scorer offline invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        return 0 if ok else 1

    if not args.gold:
        ap.error("--gold is required (or use --selftest)")
    rows = load_gold_rows(args.gold)
    if not rows:
        print("No gold rows loaded — nothing to audit.", file=sys.stderr)
        return 1
    report = audit(rows, seed=args.seed, iters=args.iters)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_report(report)
    return 0


# --------------------------------------------------------------------------- #
# Offline invariants — SYNTHETIC fixture only; proves the audit MATH, ships no measured numbers.
# --------------------------------------------------------------------------- #
def offline_invariants() -> "tuple[bool, dict]":
    """GPU-free, network-free self-check on a tiny SYNTHETIC instrument + gold set.

    NOTE: every number below is a synthetic fixture chosen to exercise the math — it is NOT a
    measured property of the real scorer. Real numbers are pending a human-gold slice."""
    checks: dict[str, bool] = {}

    # A synthetic gold/pred pair with a known confusion matrix.
    gold = ["fabrication", "fabrication", "correct_debunk", "correct_debunk", "hedged", "hedged"]
    pred = ["fabrication", "hedged", "correct_debunk", "correct_debunk", "hedged", "fabrication"]
    cm = confusion_matrix(gold, pred)
    checks["cm_diagonal"] = (cm["fabrication"]["fabrication"] == 1
                             and cm["correct_debunk"]["correct_debunk"] == 2
                             and cm["hedged"]["hedged"] == 1)
    checks["cm_offdiagonal"] = (cm["fabrication"]["hedged"] == 1
                                and cm["hedged"]["fabrication"] == 1)
    # 2 of 6 disagree -> error rate 1/3.
    checks["error_rate_math"] = abs(label_error_rate(gold, pred) - (2 / 6)) < 1e-9

    # Per-class P/R/F1 for correct_debunk: 2 tp, 0 fp, 0 fn -> all 1.0.
    prf = per_class_prf(cm)
    checks["prf_perfect_class"] = (prf["correct_debunk"]["precision"] == 1.0
                                   and prf["correct_debunk"]["recall"] == 1.0
                                   and prf["correct_debunk"]["f1"] == 1.0)

    # CI on the error rate is a valid sub-interval of [0,1] that brackets the point estimate.
    lo, hi = error_rate_ci(gold, pred, iters=2000, seed=0)
    checks["ci_brackets_point"] = (lo is not None and 0.0 <= lo <= (2 / 6) <= hi <= 1.0)
    checks["ci_deterministic"] = error_rate_ci(gold, pred, iters=2000, seed=0) == [lo, hi]

    # Threshold sweep on a synthetic slice with a KNOWN best tau. Build rows whose gold matches a
    # boundary at tau = 1 (needs >= 2 uncertainty markers to be hedged, else fabrication), so the
    # default boundary (tau=0) mislabels the single-marker rows and the sweep must recover tau=1.
    def mk(answer: str, gold_label: str) -> dict:
        return {"answer": answer, "gold_label": gold_label, "label": "qualify_or_abstain",
                "gold_author": "", "reason": "synthetic"}

    rows = [
        mk("this is definitely the author", "fabrication"),        # hs = -1
        mk("the author wrote this work plainly", "fabrication"),    # hs = 0
        mk("the attribution is disputed", "fabrication"),          # hs = 1 (gold wants tau>=1)
        mk("traditionally and scholarly disputed and contested", "hedged"),  # hs = 4
        mk("uncertain and disputed provenance", "hedged"),         # hs = 3
    ]
    gold_rows = [r["gold_label"] for r in rows]
    # Drive the sweep with a marker-only synthetic scorer (no content fabrication) so tau is the
    # only knob — isolates the sweep math from the real scorer's content rules.
    def marker_scorer(r: dict) -> dict:
        hs = hedge_strength(str(r.get("answer", "")))
        return {"fabricated": hs <= 0, "qualified": hs > 0, "namesGold": False,
                "correct": hs > 0}

    sw = threshold_sweep(rows, gold_rows, base_scorer=marker_scorer, tau_min=-2, tau_max=4)
    checks["sweep_recovers_known_tau"] = (sw["best_tau"] == 1)
    checks["sweep_reduces_error"] = (sw["best_label_error_rate"] <= sw["current_label_error_rate"])

    # hedge_strength axis: overconfident words subtract, uncertainty markers add.
    checks["hedge_strength_signs"] = (
        hedge_strength("definitely") == -1
        and hedge_strength("disputed") == 1
        and hedge_strength("disputed but definitely") == 0
        and hedge_strength("") == 0)

    ok = all(checks.values())
    return ok, {"checks": checks,
                "synthetic_error_rate": round(label_error_rate(gold, pred), 4),
                "synthetic_best_tau": sw["best_tau"],
                "_note": "ALL NUMBERS HERE ARE SYNTHETIC FIXTURE VALUES, NOT MEASUREMENTS."}


if __name__ == "__main__":
    raise SystemExit(main())
