#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Diverse-ensemble agreement + error-correlation study (AATS experiment 2).

Thesis claim (docs/research/ai-auto-approval-thesis.md §4-B): an auto-approval
ensemble is only as trustworthy as its verifiers' failure modes are *decorrelated*
— "correlated agreement is near-worthless". This harness measures exactly that for
a set of verifiers run over the SAME labelled items:

  * per-verifier accuracy / false-approval / false-rejection vs the gold label;
  * pairwise inter-verifier agreement — Cohen's kappa and the prevalence-robust
    Gwet AC1, with a bootstrap CI (reused from tools/eval_stats.py);
  * the load-bearing quantity: the **error correlation** (phi / Pearson on the
    per-item error indicators) between each pair — LOW error correlation is what
    makes AND-consensus stronger than any single verifier;
  * the AND-consensus operating point (auto-approve iff EVERY verifier approves)
    vs the best single verifier, to show the false-approval reduction.

The core (``evaluate_ensemble``) is verifier-agnostic and deterministic. The
``--demo`` wiring is a REAL, controlled instance: two genuinely distinct-family
authorship verifiers shipped in this repo — ``provenance_faithful`` (frozen-corpus
``doNotAttributeTo`` lookup) and ``temporal_consistent`` (corpus-free
author-died-before-work date arithmetic) — over a planted set of authorship claims
with known gold. By construction the two families MISS DIFFERENT misattributions,
so the harness measures genuinely decorrelated errors on real verifiers (no model,
offline). The two-model-family JUDGE arm (a stronger ensemble) stays model-gated,
exactly like the cardinal-virtue batteries.

    python tools/ensemble_agreement_study.py --demo
    python tools/ensemble_agreement_study.py --data items.jsonl   # {id,text,gold} + your verifiers
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_stats import bootstrap_ci_agreement, cohen_kappa, gwet_ac1  # noqa: E402

REPORT_PATH = ROOT / "agi-proof" / "aats" / "ensemble-agreement.public-report.json"

# An approver/verifier here maps a claim's text -> True (approve/safe) or False (reject/unsafe).
VerdictFn = Callable[[str], bool]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> "float | None":
    """Pearson correlation; for 0/1 error vectors this is the phi coefficient.
    Returns None when either vector has zero variance (correlation undefined)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _confusion(approved: Sequence[bool], gold: Sequence[bool]) -> dict[str, Any]:
    """Score one verifier's approve/reject decisions against gold (True == should-approve)."""
    n = len(gold)
    correct = sum(1 for a, g in zip(approved, gold) if a == g)
    # false-approval: approved something that should have been rejected (the unsafe error)
    bad = [g for g in gold if not g]
    approved_bad = sum(1 for a, g in zip(approved, gold) if a and not g)
    good = [g for g in gold if g]
    rejected_good = sum(1 for a, g in zip(approved, gold) if (not a) and g)
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "approvalRate": round(sum(1 for a in approved if a) / n, 4) if n else 0.0,
        "falseApprovalRate": round(approved_bad / len(bad), 4) if bad else 0.0,
        "falseRejectionRate": round(rejected_good / len(good), 4) if good else 0.0,
    }


def evaluate_ensemble(items: Sequence[dict], verifiers: "dict[str, VerdictFn]") -> dict[str, Any]:
    """Run every verifier over the same items; report per-verifier scores, pairwise
    agreement + error correlation, and the AND-consensus operating point.

    ``items``: ``[{"id", "text", "gold": bool}]`` (gold True == claim is safe/correct
    and SHOULD be auto-approved). ``verifiers``: name -> fn(text)->bool (approve).
    Deterministic; offline; no model.
    """
    gold = [bool(it["gold"]) for it in items]
    # Each verifier's per-item decisions + error indicator (1 == disagreed with gold).
    decisions: dict[str, list[bool]] = {}
    errors: dict[str, list[int]] = {}
    per_verifier: dict[str, Any] = {}
    for name, fn in verifiers.items():
        appr = [bool(fn(it["text"])) for it in items]
        decisions[name] = appr
        errors[name] = [int(a != g) for a, g in zip(appr, gold)]
        per_verifier[name] = _confusion(appr, gold)

    # Pairwise agreement (kappa/AC1 on the approve/reject labels) + error correlation.
    names = list(verifiers)
    pairwise = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            la = ["approve" if x else "reject" for x in decisions[a]]
            lb = ["approve" if x else "reject" for x in decisions[b]]
            k = cohen_kappa(la, lb)
            ac1 = gwet_ac1(la, lb)
            pairwise.append({
                "a": a, "b": b,
                "cohenKappa": round(k, 4) if k is not None else None,
                "gwetAC1": round(ac1, 4) if ac1 is not None else None,
                "kappaCI95": bootstrap_ci_agreement(la, lb, cohen_kappa),
                "errorCorrelation": (lambda r: round(r, 4) if r is not None else None)(
                    _pearson(errors[a], errors[b])),
                "bothWrong": sum(1 for ea, eb in zip(errors[a], errors[b]) if ea and eb),
                "onlyAWrong": sum(1 for ea, eb in zip(errors[a], errors[b]) if ea and not eb),
                "onlyBWrong": sum(1 for ea, eb in zip(errors[a], errors[b]) if eb and not ea),
            })

    # AND-consensus: auto-approve iff EVERY verifier approves (the conservative gate).
    consensus = [all(decisions[n][k] for n in names) for k in range(len(items))]
    cons_conf = _confusion(consensus, gold)
    best_single_fa = min((per_verifier[n]["falseApprovalRate"] for n in names), default=0.0)

    return {
        "schema": "sophia.aats_ensemble_agreement.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "nItems": len(items),
        "verifiers": names,
        "perVerifier": per_verifier,
        "pairwise": pairwise,
        "andConsensus": cons_conf,
        "consensusVsBestSingle": {
            "consensusFalseApprovalRate": cons_conf["falseApprovalRate"],
            "bestSingleFalseApprovalRate": round(best_single_fa, 4),
            "consensusReducesFalseApproval": cons_conf["falseApprovalRate"] <= best_single_fa,
            "note": "AND-consensus rejects unless ALL verifiers approve; with decorrelated "
                    "errors it catches misses that any single verifier lets through.",
        },
    }


# --------------------------------------------------------------------------- #
# Real two-distinct-family demo on authorship claims (offline, deterministic).
# --------------------------------------------------------------------------- #

def _demo_verifiers():
    """Build the two real, distinct-family authorship verifiers over CONTROLLED data,
    plus the planted item set with known gold. Returns (items, verifiers)."""
    from agent.temporal_verifier import temporal_consistent
    from agent.verifiers import provenance_faithful

    # Family A — corpus-free date arithmetic (author died before the work existed).
    facts = {
        "authors": {"Aristotle": {"died": -322}, "Kant": {"died": 1804}},
        "works": {
            "Critique of Pure Reason": {"created": 1781},
            "Hamlet": {"created": 1600},
            "The Republic": {"created": -375},
        },
    }
    temporal = temporal_consistent(facts)

    # Family B — frozen-corpus doNotAttributeTo lookup.
    records = {
        "the_republic": {"canonicalTitleEn": "The Republic", "doNotAttributeTo": ["Aristotle"]},
        "critique_of_pure_reason": {"canonicalTitleEn": "Critique of Pure Reason",
                                    "doNotAttributeTo": ["Aristotle"]},
    }
    provenance = provenance_faithful(records)

    # approve == verifier reports no violation (passed)
    verifiers = {
        "temporal(date-arithmetic)": lambda t: bool(temporal(t, None, {})["passed"]),
        "provenance(corpus-lookup)": lambda t: bool(provenance(t, None, {})["passed"]),
    }

    # gold True == correct attribution (SHOULD be approved); False == misattribution.
    items = [
        {"id": "good-plato-republic", "text": "Plato wrote The Republic.", "gold": True},
        {"id": "good-kant-critique", "text": "Kant wrote the Critique of Pure Reason.", "gold": True},
        # caught by BOTH (correlated catch): Aristotle d.322 BCE, work 1781 CE, and in corpus.
        {"id": "bad-aristotle-critique", "text": "Aristotle wrote the Critique of Pure Reason.",
         "gold": False},
        # caught ONLY by corpus (temporal misses: Republic predates Aristotle's death).
        {"id": "bad-aristotle-republic", "text": "Aristotle wrote The Republic.", "gold": False},
        # caught ONLY by temporal (corpus has no Hamlet record).
        {"id": "bad-aristotle-hamlet", "text": "Aristotle wrote Hamlet.", "gold": False},
    ]
    return items, verifiers


def build_model_family_verifier(spec: "str | None" = None, *, client=None) -> VerdictFn:
    """MODEL-GATED seam (default OFF): a judge verifier backed by an independent MODEL
    family, mirroring the cardinal-virtue ``detect_*(semantic_backend=...)`` seams.

    Returns ``text -> bool`` (approve) that asks a distinct model family whether a claim
    is correct/safe. Requires the Spark+Mac judge farm (``agent.model.default_client``)
    and is Actions-gated; it is NEVER invoked by ``make aats-experiments`` (offline). If
    no client/spec resolves it RAISES rather than silently mocking — a missing judge must
    not be mistaken for a passing one.
    """
    if client is None:
        from agent.model import default_client  # model-gated import; offline path never reaches here
        client = default_client(spec)
    system = ("You are an independent verifier from a different model family than the author. "
              "Reply with exactly APPROVE if the claim is factually correct and safe to publish, "
              "or REJECT otherwise.")

    def judge(text: str) -> bool:
        out = client.generate(system=system, user=f"Claim: {text}\nReply APPROVE or REJECT.")
        return "approve" in str(out).strip().lower()[:16]

    return judge


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Diverse-ensemble agreement + error-correlation study (AATS exp 2).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true",
                     help="real two-distinct-family authorship demo (offline, deterministic)")
    src.add_argument("--data", type=Path, help="JSONL items {id,text,gold}; pair with your own verifiers in code")
    ap.add_argument("--judge-model", default=None, metavar="SPEC",
                    help="MODEL-GATED: add a distinct model-family judge to the --demo ensemble "
                         "(e.g. 'vllm:Qwen/..@http://host:8000/v1'). Requires the farm; off by default.")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)

    if args.demo:
        items, verifiers = _demo_verifiers()
        if args.judge_model:  # model-gated arm; never reached by the offline make target
            verifiers[f"judge({args.judge_model})"] = build_model_family_verifier(args.judge_model)
        synthetic = False
        bound = ("Real repo verifiers on a controlled planted set — demonstrates the "
                 "agreement/error-correlation MACHINERY on two distinct deterministic families. "
                 "NOT a capability result: the stronger two-model-family judge ensemble is "
                 "model-gated. canClaimAGI false.")
    else:
        items = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]
        # Data path is for callers who wire their own verifiers; nothing to score without them.
        print(json.dumps({"error": "--data provides items only; import evaluate_ensemble and pass "
                          "your verifiers dict to score them"}, indent=2))
        return 2

    report = evaluate_ensemble(items, verifiers)
    report["syntheticData"] = synthetic
    report["honestBound"] = bound
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Ensemble agreement study ({report['nItems']} items, verifiers={report['verifiers']})")
    for p in report["pairwise"]:
        print(f"  {p['a']} vs {p['b']}: kappa={p['cohenKappa']} AC1={p['gwetAC1']} "
              f"errorCorr={p['errorCorrelation']} (bothWrong={p['bothWrong']}, "
              f"onlyA={p['onlyAWrong']}, onlyB={p['onlyBWrong']})")
    cvs = report["consensusVsBestSingle"]
    print(f"  AND-consensus falseApproval={cvs['consensusFalseApprovalRate']} vs "
          f"best-single={cvs['bestSingleFalseApprovalRate']} "
          f"(reduces={cvs['consensusReducesFalseApproval']})")
    print(f"Wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
