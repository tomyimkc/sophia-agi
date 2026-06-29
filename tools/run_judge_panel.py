#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Multi-judge labeling panel — turns single-source labels into >= 2-judge-family ground truth.

Both the G10R risk-awareness bank and the T3 correctness corpus carry a hard requirement before
any number may be cited: labels must come from >= 2 INDEPENDENT judge families with measured
agreement (kappa >= 0.40 OR Gwet AC1 >= 0.40, with a CI). This harness makes that requirement a
runnable instrument: given per-judge label vectors over the SAME items, it (1) computes the
consensus label by majority vote (ties -> unresolved, fail-closed), and (2) computes pairwise
Cohen's kappa and Gwet's AC1 (+ bootstrap CIs) across every judge pair, then GO/NO-GOs the panel
on the >= 0.40 floor and a low unresolved fraction.

The agreement primitives are the audited ones in tools/eval_stats.py (cohen_kappa, gwet_ac1,
bootstrap_ci_agreement); this tool adds the panel layer (pairwise sweep + majority consensus +
the gate).

Modes (all offline):
  * --mock-bank <jsonl>: simulate a panel over a labelled bank (e.g. the G10R bank). Synthetic
    judges agree with each item's reference label at a controllable rate; a high-agreement panel
    PASSES, a coin-flip judge DRAGS kappa below the floor. Exercises the panel + gate math in CI;
    NOT evidence (synthetic judges are not independent expert families).
  * --model <spec>: refuse rather than fabricate; the panel stays PENDING.

Output: a labeled set with provenance (consensus label per item + the agreement stats + the
gate verdict), the artifact the G10R/T3 gates consume as ground truth.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_stats import bootstrap_ci_agreement, cohen_kappa, gwet_ac1  # noqa: E402

AGREEMENT_FLOOR = 0.40           # kappa OR AC1 must clear this (mirrors the measurement specs)
MAX_UNRESOLVED_FRACTION = 0.05   # too many majority-vote ties -> the panel cannot label


def majority_consensus(judge_labels: list[list[str]]) -> tuple[list[str | None], int]:
    """Per-item majority vote across judges; a tie (no strict majority) -> None (unresolved)."""
    n_items = len(judge_labels[0])
    consensus: list[str | None] = []
    unresolved = 0
    for i in range(n_items):
        votes: dict[str, int] = {}
        for judge in judge_labels:
            votes[judge[i]] = votes.get(judge[i], 0) + 1
        top = max(votes.values())
        winners = [lab for lab, c in votes.items() if c == top]
        if len(winners) == 1:
            consensus.append(winners[0])
        else:
            consensus.append(None)
            unresolved += 1
    return consensus, unresolved


def pairwise_agreement(judges: dict[str, list[str]], *, seed: int = 0) -> list[dict]:
    """Cohen kappa + Gwet AC1 (+ bootstrap CIs) for every judge pair."""
    out: list[dict] = []
    for a, b in combinations(sorted(judges), 2):
        ka = cohen_kappa(judges[a], judges[b])
        ac = gwet_ac1(judges[a], judges[b])
        out.append({
            "pair": [a, b],
            "kappa": round(ka, 4) if ka is not None else None,
            "ac1": round(ac, 4) if ac is not None else None,
            "kappaCI95": bootstrap_ci_agreement(judges[a], judges[b], cohen_kappa, seed=seed),
            "ac1CI95": bootstrap_ci_agreement(judges[a], judges[b], gwet_ac1, seed=seed),
        })
    return out


def panel_verdict(judges: dict[str, list[str]], pairwise: list[dict], unresolved_frac: float) -> dict:
    """GO/NO-GO over the panel pillars: >= 2 judges, min pairwise agreement >= floor, low unresolved."""
    failures: list[str] = []
    if len(judges) < 2:
        failures.append("not_2_families: a panel needs >= 2 independent judge families")
    # A pair clears the bar if EITHER kappa or AC1 >= floor (AC1 rescues kappa under skew).
    def pair_ok(p: dict) -> bool:
        return (p["kappa"] is not None and p["kappa"] >= AGREEMENT_FLOOR) or \
               (p["ac1"] is not None and p["ac1"] >= AGREEMENT_FLOOR)
    weak = [p["pair"] for p in pairwise if not pair_ok(p)]
    if weak:
        failures.append(f"low_agreement: judge pairs below {AGREEMENT_FLOOR} on both kappa and AC1: {weak}")
    if unresolved_frac > MAX_UNRESOLVED_FRACTION:
        failures.append(f"too_many_ties: unresolved fraction {unresolved_frac:.3f} > {MAX_UNRESOLVED_FRACTION}")
    return {
        "verdict": "GO" if not failures else "NO-GO",
        "go": not failures,
        "criticalFailures": failures,
        "agreementFloor": AGREEMENT_FLOOR,
        "boundary": (
            "Panel ground truth is admissible only when >= 2 independent families agree at "
            "kappa/AC1 >= 0.40 with the unresolved fraction low. Synthetic judges are not "
            "independent families — --mock-bank is machinery proof, not evidence. canClaimAGI:false."
        ),
    }


def run_panel(items: list[dict], judges: dict[str, list[str]], *, seed: int = 0) -> dict:
    """Score a real panel: per-item consensus + pairwise agreement + the gate."""
    lengths = {len(v) for v in judges.values()} | {len(items)}
    if len(lengths) != 1:
        raise ValueError(f"items and every judge vector must align; got lengths {lengths}")
    judge_labels = [judges[name] for name in sorted(judges)]
    consensus, unresolved = majority_consensus(judge_labels)
    unresolved_frac = unresolved / len(items) if items else 1.0
    pairwise = pairwise_agreement(judges, seed=seed)
    verdict = panel_verdict(judges, pairwise, unresolved_frac)
    labeled = [{"id": it.get("id"), "consensusLabel": c}
               for it, c in zip(items, consensus)]
    return {
        "schema": "sophia.judge_panel.v1",
        "nItems": len(items),
        "judges": sorted(judges),
        "unresolved": unresolved,
        "unresolvedFraction": round(unresolved_frac, 4),
        "pairwiseAgreement": pairwise,
        "verdict": verdict["verdict"],
        "go": verdict["go"],
        "criticalFailures": verdict["criticalFailures"],
        "labeledSet": labeled,
        "agreementFloor": AGREEMENT_FLOOR,
        "candidateOnly": True,
        "canClaimAGI": False,
        "boundary": verdict["boundary"],
    }


# --------------------------------------------------------------------------- #
# Mock judges (machinery check only).
# --------------------------------------------------------------------------- #
def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _to_label(v: Any) -> str:
    return "unsafe" if v in (1, "1", "unsafe", True) else "safe"


def _simulate_judges(items: list[dict], *, agree_rates: dict[str, float], seed: int = 0) -> dict[str, list[str]]:
    """Synthetic judges: each agrees with the item's reference label at its agree_rate, else flips.

    A judge at rate ~1.0 is a near-expert; a judge at ~0.5 is a coin-flip that destroys kappa."""
    rng = random.Random(seed)
    refs = [_to_label(it.get("label")) for it in items]
    judges: dict[str, list[str]] = {}
    for name, rate in agree_rates.items():
        labels: list[str] = []
        for ref in refs:
            if rng.random() < rate:
                labels.append(ref)
            else:
                labels.append("safe" if ref == "unsafe" else "unsafe")
        judges[name] = labels
    return judges


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock-bank", default=None, help="JSONL bank to simulate a panel over (e.g. the G10R bank)")
    ap.add_argument("--coin-flip", action="store_true", help="add a coin-flip judge to demonstrate a NO-GO panel")
    ap.add_argument("--model", default=None, help="real-judge model spec (refused offline; panel stays PENDING)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.model:
        print(json.dumps({
            "status": "refused",
            "reason": "a real panel needs >= 2 independent expert/model judge families; not fabricated here. "
                      "Panel stays PENDING / NO-GO until real judge labels are supplied.",
            "verdict": "NO-GO", "canClaimAGI": False,
        }, indent=2))
        return 0
    if args.mock_bank:
        items = _load_jsonl(Path(args.mock_bank))
        # Odd panel (3 judges) so a binary majority always resolves — no ties. --coin-flip
        # swaps one expert for a coin-flip judge to demonstrate the low-agreement NO-GO.
        if args.coin_flip:
            rates = {"judgeA": 0.97, "judgeB": 0.95, "judgeCoin": 0.5}
        else:
            rates = {"judgeA": 0.97, "judgeB": 0.95, "judgeC": 0.93}
        judges = _simulate_judges(items, agree_rates=rates, seed=args.seed)
        print(json.dumps(run_panel(items, judges, seed=args.seed), indent=2, ensure_ascii=False))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
