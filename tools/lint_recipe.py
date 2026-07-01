#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Lint the master training recipe spec so the RECEIPT itself cannot overclaim.

The recipe (docs/06-Roadmap/recipe_spec.json) is the single machine-checkable source of truth for
"what goes into the trained model, and what proof licenses it". This linter enforces the repo's
no-overclaim discipline on the recipe:

  ADOPTION RULE (the load-bearing gate):
    an ingredient may be marked  "adopted": true  (i.e. folded into the model as HARD training
    signal) ONLY IF  proofStatus == "validated"  AND  ablationDelta is a number  AND  a gate is
    named. An idea with no measured ablation delta, or that has not passed a gate, MUST stay
    adopted:false (candidate/auxiliary) — you cannot ship an unproven idea into the recipe.

  Plus structural validity: known layers, unique ids, valid proofStatus, canClaimAGI == false.

Deterministic / offline / stdlib-only. Exit 0 = OK, 1 = violation. Mirrors tools/lint_claims.py so
it can join `make claim-check`. `canClaimAGI` stays false.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "docs" / "06-Roadmap" / "recipe_spec.json"

VALID_STATUS = {"validated", "candidate", "open"}
REQUIRED_TOP = {"version", "canClaimAGI", "claimBoundary", "layers", "ingredients", "adoptionRule"}
REQUIRED_ING = {"id", "layer", "module", "description", "proofStatus", "gate", "ablationDelta", "adopted"}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def lint(spec: dict) -> "tuple[bool, list[str], dict]":
    errs: list[str] = []
    warns: list[str] = []

    missing_top = REQUIRED_TOP - set(spec)
    if missing_top:
        errs.append(f"missing top-level keys: {sorted(missing_top)}")
        return False, errs, {}

    # no-overclaim invariant
    if spec.get("canClaimAGI") is not False:
        errs.append("canClaimAGI MUST be false in the recipe spec")

    layers = spec.get("layers") or []
    if not isinstance(layers, list) or not layers:
        errs.append("layers must be a non-empty list")
    layer_set = set(layers)

    ings = spec.get("ingredients") or []
    seen_ids: set[str] = set()
    counts = {"validated": 0, "candidate": 0, "open": 0, "adopted": 0, "total": 0}

    for i, ing in enumerate(ings):
        tag = ing.get("id", f"#{i}")
        miss = REQUIRED_ING - set(ing)
        if miss:
            errs.append(f"[{tag}] missing fields: {sorted(miss)}")
            continue
        counts["total"] += 1
        if ing["id"] in seen_ids:
            errs.append(f"[{tag}] duplicate id")
        seen_ids.add(ing["id"])
        if ing["proofStatus"] not in VALID_STATUS:
            errs.append(f"[{tag}] proofStatus '{ing['proofStatus']}' not in {sorted(VALID_STATUS)}")
        else:
            counts[ing["proofStatus"]] += 1
        if ing["layer"] not in layer_set:
            errs.append(f"[{tag}] layer '{ing['layer']}' not in declared layers {layers}")
        if not isinstance(ing["adopted"], bool):
            errs.append(f"[{tag}] adopted must be a bool")
            continue

        # ---- THE ADOPTION RULE ----
        if ing["adopted"]:
            counts["adopted"] += 1
            if ing["proofStatus"] != "validated":
                errs.append(f"[{tag}] adopted:true but proofStatus is '{ing['proofStatus']}' "
                            "(must be 'validated' to fold in as hard signal)")
            if not _is_number(ing["ablationDelta"]):
                errs.append(f"[{tag}] adopted:true but ablationDelta is not a number "
                            f"({ing['ablationDelta']!r}) — no measured on/off value, cannot adopt")
            if not (ing["gate"] and str(ing["gate"]).strip()):
                errs.append(f"[{tag}] adopted:true but no gate named")
        else:
            # candidate/open ingredients are fine unadopted; nudge if a validated one is left out.
            if ing["proofStatus"] == "validated" and _is_number(ing["ablationDelta"]):
                warns.append(f"[{tag}] validated + has ablationDelta but adopted:false "
                             "(eligible to adopt — confirm intentional)")

    return (len(errs) == 0), errs, {"counts": counts, "warnings": warns}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--spec", default=str(DEFAULT_SPEC), help="path to recipe_spec.json")
    ap.add_argument("--check", action="store_true", help="(default) validate and exit non-zero on error")
    args = ap.parse_args(argv)

    path = Path(args.spec)
    if not path.exists():
        print(f"RECIPE LINTER: FAIL — spec not found at {path}")
        return 1
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"RECIPE LINTER: FAIL — {path} is not valid JSON: {exc}")
        return 1

    ok, errs, info = lint(spec)
    counts = info.get("counts", {})
    for w in info.get("warnings", []):
        print(f"  [warn] {w}")
    if ok:
        print(f"RECIPE LINTER: OK — {counts.get('total', 0)} ingredient(s): "
              f"{counts.get('validated', 0)} validated / {counts.get('candidate', 0)} candidate / "
              f"{counts.get('open', 0)} open; {counts.get('adopted', 0)} adopted, all proof-gated. "
              "canClaimAGI false.")
        return 0
    print("RECIPE LINTER: FAIL")
    for e in errs:
        print(f"  [XX] {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
