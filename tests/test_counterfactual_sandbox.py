#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Counterfactual sandbox — structural containment + promotion choke-point. Offline.

The generator is a fabrication engine; safety is structural:
  - it emits only bulk-only, tier-tagged, non-promotable nodes;
  - the promotion choke-point bars counterfactual/synthetic tiers even when a human
    approved the row;
  - the no_synthetic_promotion invariant detects a tainted committed row.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.counterfactual_philosophy import audit_counterfactual, generate_counterfactual, is_promotable_tier  # noqa: E402
from agent.invariant_suite import no_synthetic_promotion  # noqa: E402
from okf import promotion_loop  # noqa: E402


def test_generator_is_structurally_contained() -> None:
    bulk = generate_counterfactual(figure="Aristotle", source_concept="eudaimonia", counterpart_concept="wu wei")
    audit = audit_counterfactual(bulk)
    assert audit["structurallyContained"] is True
    assert audit["nodeCount"] >= 1
    assert is_promotable_tier("counterfactual") is False
    assert is_promotable_tier("attributed") is True


def test_commit_bars_counterfactual_even_when_approved() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pending = Path(tmp) / "pending.jsonl"
        row = {
            "schema": "sophia.projection_candidate.v1", "nodeId": "cf_node",
            "meta": {"pageType": "concept", "sourceTier": "counterfactual"},
            "bodyPreview": "fabricated", "checks": {}, "source": "counterfactual",
            "candidateOnly": True, "promoted": True, "approved": True, "committed": False,
            "submittedAt": "2026-06-27T00:00:00+00:00",
        }
        pending.write_text(json.dumps(row) + "\n", encoding="utf-8")
        result = promotion_loop.commit_approved_candidate("cf_node", path=pending)
        assert result["ok"] is False
        assert result.get("nonPromotableTier") is True
        # the row must remain uncommitted on disk
        after = [json.loads(l) for l in pending.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert after[0].get("committed") in (False, None)


def test_invariant_detects_tainted_commit() -> None:
    clean = [{"nodeId": "ok", "committed": True, "meta": {"sourceTier": "attributed"}}]
    tainted = [{"nodeId": "bad", "committed": True, "meta": {"sourceTier": "counterfactual"}}]
    assert no_synthetic_promotion(clean)["verdict"] == "accepted"
    assert no_synthetic_promotion(tainted)["verdict"] == "rejected"
    assert no_synthetic_promotion(None)["verdict"] == "held"


def main() -> int:
    test_generator_is_structurally_contained()
    test_commit_bars_counterfactual_even_when_approved()
    test_invariant_detects_tainted_commit()
    print("test_counterfactual_sandbox: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
