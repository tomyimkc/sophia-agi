#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Benchmark the four AATS auto-approval harnesses at scale, with CIs (AATS benchmark).

The harnesses (docs/research/ai-auto-approval-thesis.md §5) ship with tiny demos. This
benchmark stress-tests their LOAD-BEARING claims on larger, controlled, deterministic
batteries and attaches uncertainty (bootstrap CIs + McNemar) from tools/eval_stats.py:

  exp 1  census        — the auto-approval envelope is stable and every member is
                         deterministic + offline + independent.
  exp 2  ensemble      — on a controlled authorship battery spanning all four
                         catch-quadrants (both / temporal-only / provenance-only /
                         neither), AND-consensus catches significantly MORE bad items
                         than the best single verifier (McNemar), AND it honestly does
                         NOT catch the 'neither' quadrant (consensus is not magic).
  exp 3  conformal     — the split-conformal coverage guarantee holds across separation
                         regimes and seeds; the escalation price rises as the target
                         false-approval rate tightens.
  exp 4  breaker       — an armed breaker trips iff a known-bad canary is approved;
                         it detects every false-approval its canary set COVERS, and is
                         blind to failure modes the canary set omits (honest bound).

Everything is synthetic/controlled and deterministic (fixed seeds). It characterises the
MACHINERY, not a model — candidateOnly, canClaimAGI false.

    python tools/aats_benchmark.py            # run + print summary + write report
    python tools/aats_benchmark.py --check     # exit 1 if a load-bearing claim regresses
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.aats_conformal_calibration import calibration_curve, choose_operating_point  # noqa: E402
from tools.ensemble_agreement_study import evaluate_ensemble  # noqa: E402
from tools.eval_stats import bootstrap_ci_paired, fixed_n_ci_mean, mcnemar  # noqa: E402
from tools.fit_conformal_policy import synthetic_rows  # noqa: E402
from tools.verify_generate_census import census  # noqa: E402

REPORT_PATH = ROOT / "agi-proof" / "aats" / "aats-benchmark.public-report.json"

# --------------------------------------------------------------------------- #
# Exp 2 — controlled authorship battery (real verifiers, all catch-quadrants)
# --------------------------------------------------------------------------- #

# Real authors (death year; negative = BCE) and works (creation year, true author).
_AUTHORS = {
    "Aristotle": -322, "Plato": -348, "Kant": 1804, "Nietzsche": 1900, "Descartes": 1650,
    "Shakespeare": 1616, "Dante": 1321, "Aquinas": 1274, "Hume": 1776, "Locke": 1704,
    "Marx": 1883, "Newton": 1727, "Spinoza": 1677, "Hobbes": 1679, "Rousseau": 1778,
}
_WORKS = {
    "The Republic": (-375, "Plato"), "Critique of Pure Reason": (1781, "Kant"),
    "Hamlet": (1600, "Shakespeare"), "Thus Spoke Zarathustra": (1883, "Nietzsche"),
    "Meditations on First Philosophy": (1641, "Descartes"), "The Divine Comedy": (1320, "Dante"),
    "Summa Theologica": (1265, "Aquinas"), "Principia Mathematica": (1687, "Newton"),
    "Das Kapital": (1867, "Marx"), "Leviathan": (1651, "Hobbes"),
    "The Social Contract": (1762, "Rousseau"), "Ethics": (1677, "Spinoza"),
}
_GRACE = 50  # temporal verifier's posthumous grace window


def _temporal_catches(author: str, work: str) -> bool:
    created, _ = _WORKS[work]
    return (created - _AUTHORS[author]) > _GRACE


def _build_authorship_battery():
    """Construct correct attributions + the FULL cross-product of misattributions, spanning
    all four catch-quadrants, plus the temporal facts and provenance records that make each
    quadrant real. Deterministic. Provenance coverage uses an even/odd rule over the sorted
    wrong-author list per work, so each work contributes BOTH covered (provenance catches)
    and uncovered (provenance blind) misattributions — guaranteeing all quadrants populate.
    Returns (items, facts, records, intended) where `intended` maps id -> intended quadrant.
    """
    facts = {"authors": {a: {"died": y} for a, y in _AUTHORS.items()},
             "works": {w: {"created": c} for w, (c, _) in _WORKS.items()}}
    items, intended = [], {}
    prov_catch: dict[str, list[str]] = {}  # work -> [wrong authors provenance must flag]

    for w, (_c, true_author) in sorted(_WORKS.items()):
        # correct attribution (gold True — must be approved by both)
        cid = f"correct::{true_author}::{w}"
        items.append({"id": cid, "text": f"{true_author} wrote {w}.", "gold": True})
        intended[cid] = "correct"
        # every wrong author -> a misattribution (gold False)
        wrong = [a for a in sorted(_AUTHORS) if a != true_author]
        for idx, a in enumerate(wrong):
            tc = _temporal_catches(a, w)
            cover = (idx % 2 == 0)          # half of each work's wrong authors get a record
            quad = ("both" if tc and cover else "temporal-only" if tc and not cover
                    else "prov-only" if (not tc) and cover else "neither")
            cid = f"bad::{a}::{w}"
            items.append({"id": cid, "text": f"{a} wrote {w}.", "gold": False})
            intended[cid] = quad
            if cover:
                prov_catch.setdefault(w, []).append(a)

    records = {
        w.lower().replace(" ", "_"): {"canonicalTitleEn": w, "doNotAttributeTo": authors}
        for w, authors in prov_catch.items()
    }
    return items, facts, records, intended


def benchmark_ensemble() -> dict:
    from agent.temporal_verifier import temporal_consistent
    from agent.verifiers import provenance_faithful

    items, facts, records, intended = _build_authorship_battery()
    temporal = temporal_consistent(facts)
    provenance = provenance_faithful(records)
    verifiers = {
        "temporal": lambda t: bool(temporal(t, None, {})["passed"]),
        "provenance": lambda t: bool(provenance(t, None, {})["passed"]),
    }
    rep = evaluate_ensemble(items, verifiers)

    # Quadrant breakdown on the known-bad items: who actually caught each (approve==False==caught).
    bad = [it for it in items if not it["gold"]]
    t_app = {it["id"]: verifiers["temporal"](it["text"]) for it in bad}
    p_app = {it["id"]: verifiers["provenance"](it["text"]) for it in bad}
    quad_actual = {"both": 0, "temporal-only": 0, "provenance-only": 0, "neither": 0}
    for it in bad:
        caught_t = not t_app[it["id"]]
        caught_p = not p_app[it["id"]]
        if caught_t and caught_p:
            quad_actual["both"] += 1
        elif caught_t:
            quad_actual["temporal-only"] += 1
        elif caught_p:
            quad_actual["provenance-only"] += 1
        else:
            quad_actual["neither"] += 1

    # McNemar: does AND-consensus catch significantly MORE bad items than the best single
    # verifier? base = best-single "caught", adapter = consensus "caught", over known-bad.
    best_single = min(verifiers, key=lambda n: rep["perVerifier"][n]["falseApprovalRate"])
    single_caught = [0 if (verifiers[best_single](it["text"])) else 1 for it in bad]  # 1==caught
    cons_caught = [0 if (t_app[it["id"]] and p_app[it["id"]]) else 1 for it in bad]    # consensus approves iff both approve
    mc = mcnemar(single_caught, cons_caught)

    # CI on consensus false-approval rate over the known-bad items (approve==miss==1).
    cons_miss = [1.0 if (t_app[it["id"]] and p_app[it["id"]]) else 0.0 for it in bad]
    return {
        "nItems": len(items), "nBad": len(bad),
        "perVerifier": rep["perVerifier"],
        "pairwise": rep["pairwise"],
        "andConsensus": rep["andConsensus"],
        "bestSingle": best_single,
        "quadrantsCaught": quad_actual,
        "intendedQuadrants": {q: sum(1 for v in intended.values() if v == q)
                              for q in ("correct", "both", "temporal-only", "prov-only", "neither")},
        "consensusFalseApprovalCI95": fixed_n_ci_mean(cons_miss),
        "mcnemarConsensusVsBestSingle": mc,
        "consensusCatchesMore": mc["c"] > mc["b"],
        "neitherQuadrantIsHonestBound": quad_actual["neither"] > 0,
    }


# --------------------------------------------------------------------------- #
# Exp 3 — conformal validity across regimes + seeds
# --------------------------------------------------------------------------- #

def benchmark_conformal(*, regimes=(4.0, 8.0, 16.0), seeds=(1, 2, 3, 4, 5), n=1000) -> dict:
    """Across separation regimes x seeds: (a) WITHIN-bucket split-conformal validity (the
    statistically correct usage — fit + measure coverage per risk bucket via fit_and_validate);
    (b) the auto-approval price curve over the whole stream + the NAIVE single-threshold
    cross-bucket coverage, reported as an honest finding (a single normal-bucket threshold
    under-covers high-risk correct items)."""
    from tools.fit_conformal_policy import fit_and_validate

    cells = []
    within_hits = within_total = 0
    naive_hits = naive_total = 0
    for k in regimes:
        for s in seeds:
            rows = synthetic_rows(n, seed=1000 + s, sep=k)
            # (a) within-bucket validity at a small alpha sweep
            for a in (0.05, 0.1, 0.2):
                fv = fit_and_validate(rows, alpha=a)
                within_total += 1
                within_hits += int(bool(fv["validityHolds"]))
            # (b) whole-stream price curve + naive cross-bucket coverage
            curve = calibration_curve(rows)
            for p in curve:
                naive_total += 1
                naive_hits += int(bool(p["validityHolds"]))
            chosen = choose_operating_point(curve, target_false_approve=0.05)
            cells.append({"separation": k, "seed": s,
                          "lenientEscalation": curve[0]["escalationRate"],
                          "strictEscalation": curve[-1]["escalationRate"],
                          "lenientFalseApprove": curve[0]["falseApprovalRate"],
                          "strictFalseApprove": curve[-1]["falseApprovalRate"],
                          "meets005": chosen is not None,
                          "escalationToMeet005": (chosen or {}).get("escalationRate")})
    monotone = all(c["strictEscalation"] >= c["lenientEscalation"] for c in cells)
    cheaper_false_approve = all(c["strictFalseApprove"] <= c["lenientFalseApprove"] for c in cells)
    return {
        "regimes": list(regimes), "seeds": list(seeds), "nPerCell": n, "cells": cells,
        "withinBucketValidityHeldRate": round(within_hits / within_total, 4) if within_total else 0.0,
        "naiveCrossBucketValidityHeldRate": round(naive_hits / naive_total, 4) if naive_total else 0.0,
        "priceMonotone": monotone,
        "stricterIsSaferEverywhere": cheaper_false_approve,
        "finding": ("split-conformal coverage holds WITHIN a risk bucket; a single normal-bucket "
                    "threshold applied across a mixed-risk stream under-covers high-risk correct "
                    "items — motivating risk-bucketed conformal policies for auto-approval."),
    }


# --------------------------------------------------------------------------- #
# Exp 4 — breaker robustness across adversarial approvers
# --------------------------------------------------------------------------- #

def benchmark_breaker() -> dict:
    from agent.auto_approval_breaker import CanaryItem, CircuitBreaker

    items, _facts, _records, _intended = _build_authorship_battery()
    canaries = [CanaryItem(id=it["id"], text=it["text"], expect_approve=it["gold"],
                           kind="authorship") for it in items]
    bad_ids = {it["id"] for it in items if not it["gold"]}

    # deterministic 'partial leak': approve a fixed fraction of the known-bad canaries.
    def partial_leak(rate: float):
        leak = sorted(bad_ids)[: max(1, int(round(rate * len(bad_ids))))] if rate > 0 else []
        leakset = set(leak)
        # approve good items + the leaked bad ones (by id lookup via text->id)
        id_by_text = {it["text"]: it["id"] for it in items}
        good_text = {it["text"] for it in items if it["gold"]}
        return lambda t: (t in good_text) or (id_by_text.get(t) in leakset)

    scenarios = {
        "perfect": (lambda t: t in {it["text"] for it in items if it["gold"]}, False),
        "leaky-all": (lambda _t: True, True),
        "partial-10": (partial_leak(0.10), True),
        "partial-50": (partial_leak(0.50), True),
    }
    results = []
    detection_complete = True
    for name, (approver, expect_trip) in scenarios.items():
        br = CircuitBreaker()
        rep = br.check_canaries(approver, canaries)
        approved_bad = [it.id for it in canaries if (not it.expect_approve) and approver(it.text)]
        tripped = br.tripped
        # the breaker must trip iff at least one known-bad canary was approved
        should_trip = len(approved_bad) > 0
        ok = (tripped == should_trip)
        if not ok:
            detection_complete = False
        results.append({"scenario": name, "approvedBadCount": len(approved_bad),
                        "tripped": tripped, "shouldTrip": should_trip, "correct": ok})
    return {"scenarios": results, "detectionComplete": detection_complete,
            "honestBound": "the breaker only catches failure modes its canary set COVERS; "
                           "a bad artifact unlike any canary is invisible to it."}


# --------------------------------------------------------------------------- #

def run() -> dict:
    cen = census()
    ens = benchmark_ensemble()
    con = benchmark_conformal()
    brk = benchmark_breaker()
    checks = {
        "censusEnvelopeStable": cen["autoApprovalEnvelope"] == [
            "arithmetic", "authorship.temporal", "authorship.provenance", "legal", "code"],
        "consensusCatchesMore": ens["consensusCatchesMore"],
        "consensusHonestlyMissesNeither": ens["neitherQuadrantIsHonestBound"],
        "conformalWithinBucketValidityHeld": con["withinBucketValidityHeldRate"] >= 0.95,
        "conformalPriceMonotone": con["priceMonotone"],
        "breakerDetectionComplete": brk["detectionComplete"],
    }
    return {
        "schema": "sophia.aats_benchmark.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "syntheticData": True,
        "validated": False,
        "census": {"envelope": cen["autoApprovalEnvelope"], "excluded": cen["excludedFromEnvelope"]},
        "ensemble": ens,
        "conformal": con,
        "breaker": brk,
        "loadBearingChecks": checks,
        "allChecksPass": all(checks.values()),
        "honestBound": ("Synthetic/controlled batteries characterise the MACHINERY (offline, "
                        "deterministic), NOT a model capability. The model-gated arms "
                        "(measurement_spec.json) remain the path to a validated claim. "
                        "canClaimAGI false."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark the four AATS harnesses with CIs.")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    ap.add_argument("--check", action="store_true", help="exit 1 if a load-bearing claim regresses")
    args = ap.parse_args(argv)

    rep = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    e = rep["ensemble"]
    print("AATS benchmark")
    print(f"  census envelope: {rep['census']['envelope']}")
    print(f"  ensemble: {e['nItems']} items ({e['nBad']} bad); quadrants caught={e['quadrantsCaught']}")
    print(f"    per-verifier falseApproval: " + ", ".join(
        f"{n}={v['falseApprovalRate']}" for n, v in e["perVerifier"].items()))
    print(f"    AND-consensus falseApproval={e['andConsensus']['falseApprovalRate']} "
          f"CI95={e['consensusFalseApprovalCI95']} | McNemar b={e['mcnemarConsensusVsBestSingle']['b']} "
          f"c={e['mcnemarConsensusVsBestSingle']['c']} p={e['mcnemarConsensusVsBestSingle']['p']}")
    c = rep["conformal"]
    print(f"  conformal: within-bucket validity {c['withinBucketValidityHeldRate']} vs naive "
          f"cross-bucket {c['naiveCrossBucketValidityHeldRate']} across {len(c['cells'])} cells; "
          f"price monotone={c['priceMonotone']}")
    b = rep["breaker"]
    print(f"  breaker: detectionComplete={b['detectionComplete']} over {len(b['scenarios'])} scenarios")
    print(f"  load-bearing checks: {rep['loadBearingChecks']}")
    print(f"Wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")

    if args.check:
        if not rep["allChecksPass"]:
            failed = [k for k, v in rep["loadBearingChecks"].items() if not v]
            print(f"FAIL: load-bearing checks regressed: {failed}")
            return 1
        print("OK: all load-bearing benchmark checks pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
