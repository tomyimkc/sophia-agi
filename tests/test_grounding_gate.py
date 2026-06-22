#!/usr/bin/env python3
"""Tests for provenance_bench.grounded — the grounding gate that closes the
cross-entity gap at low false-positive cost. Deterministic, offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.grounded import (  # noqa: E402
    ABSTAIN,
    MISATTRIBUTION,
    TRUE,
    build_kb,
    ground,
    run_grounded,
)

KB = build_kb([
    {"work": "Dao De Jing", "gold_author": "Laozi"},
    {"work": "Analects", "gold_author": "Confucius (compiled by his disciples)"},
])


def test_true_attribution_is_recognized() -> None:
    assert ground("Laozi", "Dao De Jing", KB) == TRUE
    # parenthetical qualifier on the gold must not break the match
    assert ground("Confucius", "Analects", KB) == TRUE


def test_contradiction_is_flagged() -> None:
    assert ground("Confucius", "Dao De Jing", KB) == MISATTRIBUTION
    assert ground("Plato", "Analects", KB) == MISATTRIBUTION


def test_off_kb_abstains_failclosed() -> None:
    assert ground("Anyone", "Some Unknown Work", KB) == ABSTAIN


def test_grounded_beats_structural_on_real_data() -> None:
    import json
    from provenance_bench.cross_entity import _structural_fp     # fast regex baseline
    from provenance_bench.dataset import DATA_DIR

    mis = json.loads((DATA_DIR / "misattributions.json").read_text("utf-8"))["misattributions"]
    pairs = [{"claimed": m["claimed_author"], "work": m["work"]} for m in mis]
    true = json.loads((DATA_DIR / "wikidata_snapshot.json").read_text("utf-8"))["attributions"]
    controls = [{"gold": t["gold_author"], "work": t["work"]} for t in true
                if "(" not in t["gold_author"] and "anonymous" not in t["gold_author"].lower()
                and " and " not in t["gold_author"]]
    structural_fp = _structural_fp(controls)
    g = run_grounded(pairs, controls, build_kb(true), seed=0)

    assert structural_fp >= 0.5                                   # structural flags everything
    assert g["groundedFalsePositive"] <= 0.1                      # grounding cuts FP
    assert g["groundedFalsePositive"] < structural_fp
    assert g["groundedRecall_covered"] >= 0.8                     # transfers on covered entities
    assert g["abstainRate"] > 0.0                                 # fail-closed off-KB


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_grounding_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
