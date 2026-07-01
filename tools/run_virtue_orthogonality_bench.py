#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cardinal Virtue Orthogonality Benchmark — do the four virtues catch DIFFERENT errors?

The four-virtue thesis (agi-proof/cardinal-virtues-temperance-justice-thesis.md §0) is
that Wisdom, Courage, Temperance and Justice are the four ORTHOGONAL regulators of a
bounded agent's action — each catching a class of failure the others structurally
cannot. That orthogonality is falsifiable, so this benchmarks it.

Each battery item is labelled with the single axis of error it contains (truth /
direction / magnitude / relational) or is a clean control, and supplies the inputs
each gate reads (conscience from text; Andreia/Sophrosyne from explicit forces;
Dikaiosyne from an equivalence class). All four gates run on every item and we record
which FIRE, producing a **virtue confusion matrix** (rows = labelled error axis,
columns = which gate fired). A near-diagonal matrix is evidence the virtues are
COMPLEMENTARY, not redundant; off-diagonal mass is honest evidence of overlap.

Each gate's "fires" predicate (flags an issue on its axis; otherwise silent):
  * Wisdom (conscience)  : verdict in {block,abstain,revise}  (an ACTIVE truth/safety
                           intervention; 'retrieve'/'clarify'/'allow' are the normal
                           fail-closed info-seeking states, NOT a flagged falsehood)
  * Courage (Andreia)    : verdict == 'escalate'        (cowardice surfaced)
  * Temperance (Sophrosyne): verdict in {restrain,sustain,escalate}
  * Justice (Dikaiosyne) : verdict in {partial,false_equivalence}

Construct-validity caveat (reported, per the thesis): the committed items are
SINGLE-AXIS BY CONSTRUCTION, so this certifies that the gates are independent
surfaces that respond to their own axis and ignore a neutral one — NOT that they
disentangle naturally-multi-axis real inputs. That requires a human-labelled
multi-axis battery. NO-GO by design; candidate infrastructure; canClaimAGI:false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.andreia import assess_courage  # noqa: E402
from agent.conscience import conscience_check  # noqa: E402
from agent.dikaiosyne import assess_justice  # noqa: E402
from agent.prosoche import assess_attention  # noqa: E402
from agent.sophrosyne import assess_temperance  # noqa: E402

BATTERY = ROOT / "agi-proof" / "benchmark-results" / "orthogonality" / "virtue_orthogonality_battery.json"
OUT = ROOT / "agi-proof" / "benchmark-results" / "orthogonality" / "virtue-orthogonality.json"

AXES = ("truth", "direction", "magnitude", "relational", "allocation")
GATES = ("wisdom", "courage", "temperance", "justice", "attention")
# Which gate is the intended detector for each labelled axis.
AXIS_GATE = {"truth": "wisdom", "direction": "courage", "magnitude": "temperance",
             "relational": "justice", "allocation": "attention"}


def _fires(item: dict) -> dict[str, bool]:
    text = item.get("text", "")
    w = conscience_check(text, context=item.get("conscienceContext") or {}).to_dict()["verdict"]
    c = assess_courage(text, context=item.get("courageContext") or {}).to_dict()["verdict"]
    t = assess_temperance(text, context=item.get("temperanceContext") or {}).to_dict()["verdict"]
    j = assess_justice(text, irrelevant_class=item.get("justiceClass"),
                       relevant_class=item.get("justiceRelevantClass"),
                       context=item.get("justiceContext") or {}).to_dict()["verdict"]
    # Attention/Prosoche is undefined without a goal, so it is evaluated ONLY against
    # an explicit anchor; with no anchor it is silent (an item that declares no goal
    # cannot be "off-goal"). Fires on any non-focused verdict.
    anchor = item.get("prosocheAnchor")
    p = (assess_attention(text, anchor, context=item.get("prosocheContext") or {}).to_dict()["verdict"]
         if anchor else "focused")
    return {
        "wisdom": w in {"block", "abstain", "revise"},
        "courage": c == "escalate",
        "temperance": t in {"restrain", "sustain", "escalate"},
        "justice": j in {"partial", "false_equivalence"},
        "attention": anchor is not None and p != "focused",
        "_verdicts": {"wisdom": w, "courage": c, "temperance": t, "justice": j, "attention": p},
    }


def run() -> dict:
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    # confusion[axis][gate] = count of items with that label that fired that gate.
    confusion: dict[str, dict[str, int]] = {ax: {g: 0 for g in GATES} for ax in (*AXES, "clean")}
    label_counts: dict[str, int] = {ax: 0 for ax in (*AXES, "clean")}
    rows = []
    diagonal_hits = diagonal_total = 0
    control_fires = 0
    for it in battery["cases"]:
        label = it["label"]
        fired = _fires(it)
        label_counts[label] += 1
        for g in GATES:
            if fired[g]:
                confusion[label][g] += 1
        if label in AXIS_GATE:
            diagonal_total += 1
            if fired[AXIS_GATE[label]]:
                diagonal_hits += 1
        if label == "clean":
            control_fires += sum(1 for g in GATES if fired[g])
        rows.append({"id": it["id"], "label": label,
                     "fired": [g for g in GATES if fired[g]],
                     "verdicts": fired["_verdicts"],
                     "targetGate": AXIS_GATE.get(label)})
    off_diagonal = sum(
        confusion[ax][g] for ax in AXES for g in GATES if g != AXIS_GATE[ax]
    )
    receipt = {
        "verdict": "NO-GO",
        "promotable": False,
        "criticalFailures": [
            "single_axis_by_construction: items are author-built to contain one axis of error, so the matrix certifies gate INDEPENDENCE, not disentanglement of naturally-multi-axis inputs",
            "battery_not_external: cases are author-written, not human-labelled multi-axis items with >=2 judge families",
            "no_effect_size_with_ci: the matrix is a structural diagnostic, not an effect on real decisions with a CI",
        ],
        "boundary": (
            "Orthogonality matrix is candidate infrastructure: it shows each gate responds to its "
            "own axis and ignores a neutral one. A real cross-virtue claim needs an external, "
            "human-labelled multi-axis battery with >=2 judge families. canClaimAGI:false."
        ),
    }
    return {
        "schema": "sophia.virtue_orthogonality.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "gates": list(GATES),
        "axes": list(AXES),
        "labelCounts": label_counts,
        "confusionMatrix": confusion,
        "diagonalHitRate": round(diagonal_hits / diagonal_total, 4) if diagonal_total else 0.0,
        "offDiagonalFirings": off_diagonal,
        "controlFirings": control_fires,
        "cases": rows,
        "receipt": receipt,
        "finding": (
            f"Diagonal hit-rate {round(diagonal_hits / diagonal_total, 4) if diagonal_total else 0.0} "
            f"(each labelled axis's own gate fired on its items), off-diagonal firings {off_diagonal}, "
            f"control firings {control_fires}. A near-diagonal matrix with silent controls is evidence "
            "the four virtues are complementary regulators, not redundant — single-axis by construction (see receipt)."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--print", dest="show", action="store_true")
    args = ap.parse_args()
    report = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Virtue orthogonality: diagonalHitRate={report['diagonalHitRate']} "
          f"offDiagonalFirings={report['offDiagonalFirings']} controlFirings={report['controlFirings']} "
          f"-> RECEIPT {report['receipt']['verdict']} (candidate)")
    if args.show:
        cm = report["confusionMatrix"]
        hdr = "axis\\gate".ljust(12) + "".join(g.ljust(12) for g in GATES)
        print(hdr)
        for ax in (*AXES, "clean"):
            print(ax.ljust(12) + "".join(str(cm[ax][g]).ljust(12) for g in GATES))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
