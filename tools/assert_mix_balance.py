#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Mix-balance regression gate — fail-closed on domain-mix DRIFT (Phase 4).

The corpus mix is skewed today (settled_fact over-represented; hk_bilingual /
moral_gate / tool_mcp starved — see failure-ledger ``mix-balance-gate-absent`` and
``sophia-wisdom-4b-m2-volume-below-target``). An *absolute* gate would fail CI
immediately, so — exactly like ``pipeline/quality_regression.py`` — this is a
RATCHET: it pins the current per-family distance-from-target as a baseline and fails
only if the mix gets WORSE (overall L1, or any single family, regresses beyond a
tolerance). As the mix improves, re-baseline with ``--update`` to lock the gain in.

Distance is |actual_fraction - target_midpoint| per family; L1 is their sum (0 =
perfect, larger = more skewed). Deterministic, stdlib, no timestamps.

    python tools/assert_mix_balance.py            # gate: exit 1 on regression
    python tools/assert_mix_balance.py --json      # machine-readable
    python tools/assert_mix_balance.py --update     # re-baseline (after an improvement)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "training" / "local_sophia_v3" / "manifest.json"
BASELINE = ROOT / "agi-proof" / "data-health" / "mix-baseline.json"
SCHEMA = "sophia.mix_baseline.v1"
DEFAULT_TOLERANCE = 0.02     # allowed worsening before the gate trips


def current_mix(manifest_path: Path = MANIFEST) -> dict:
    """{l1, perFamily:{fam: distance-from-target-midpoint}} from the committed manifest."""
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_family = doc.get("byFamily") or {}
    total = sum(int(v.get("rows", 0)) for v in by_family.values()) or 1
    per_family: dict[str, float] = {}
    for fam in sorted(by_family):
        info = by_family[fam]
        actual = int(info.get("rows", 0)) / total
        band = info.get("targetPct") or [0.0, 0.0]
        target_mid = (float(band[0]) + float(band[1])) / 2.0 / 100.0
        per_family[fam] = round(abs(actual - target_mid), 6)
    return {"l1Distance": round(sum(per_family.values()), 6), "perFamily": per_family}


def _baseline_doc(mix: dict, tolerance: float) -> dict:
    return {
        "schema": SCHEMA,
        "note": "Mix-balance ratchet baseline (tools/assert_mix_balance.py). "
                "Lower is better; --update only after a measured improvement.",
        "tolerance": tolerance,
        "l1Distance": mix["l1Distance"],
        "perFamily": mix["perFamily"],
    }


def serialize(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def check(mix: dict, baseline: dict) -> tuple[bool, list[str]]:
    """Return (ok, regressions). A regression = drift beyond baseline + tolerance."""
    tol = float(baseline.get("tolerance", DEFAULT_TOLERANCE))
    problems: list[str] = []
    if mix["l1Distance"] > baseline["l1Distance"] + tol:
        problems.append(
            f"overall L1 {mix['l1Distance']:.4f} > baseline {baseline['l1Distance']:.4f} + tol {tol}")
    base_fam = baseline.get("perFamily") or {}
    for fam, dist in sorted(mix["perFamily"].items()):
        ref = base_fam.get(fam)
        if ref is None:
            continue   # a new family is not a regression by itself
        if dist > ref + tol:
            problems.append(f"family '{fam}' distance {dist:.4f} > baseline {ref:.4f} + tol {tol}")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="re-baseline to the current mix (only after an improvement)")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    mix = current_mix()

    if args.update:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(serialize(_baseline_doc(mix, args.tolerance)), encoding="utf-8")
        print(f"MIX BALANCE: baseline updated — L1={mix['l1Distance']:.4f}")
        return 0

    if not BASELINE.exists():
        print(f"MIX BALANCE: FAIL — no baseline at {BASELINE.relative_to(ROOT)}; run --update once.")
        return 1

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ok, problems = check(mix, baseline)
    worst = sorted(mix["perFamily"].items(), key=lambda kv: kv[1], reverse=True)[:3]

    if args.json:
        print(json.dumps({"ok": ok, "current": mix, "baselineL1": baseline["l1Distance"],
                          "regressions": problems}, indent=2, ensure_ascii=False))
    else:
        print(f"MIX BALANCE: L1={mix['l1Distance']:.4f} (baseline {baseline['l1Distance']:.4f}) "
              f"| worst: " + ", ".join(f"{f}={d:.3f}" for f, d in worst))
        for p in problems:
            print(f"  REGRESSION: {p}")
        print("OK — no mix regression." if ok else "FAIL — mix skew regressed beyond tolerance.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
