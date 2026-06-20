#!/usr/bin/env python3
"""Tests for the graph-driven hard-negative DPO miner (tools/mine_hard_negatives.py).

CPU-only data generation. The core invariant: every emitted pair is self-validated
through provenance_faithful — the `rejected` MUST trip the gate (a real lineage
merge) and the `chosen` MUST pass it. Offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.verifiers import provenance_faithful  # noqa: E402
from tools import mine_hard_negatives as mhn  # noqa: E402

RECORDS = {
    "dao_de_jing": {
        "canonicalTitleEn": "Dao De Jing", "attributedAuthor": "laozi",
        "authorConfidence": "legendary", "doNotAttributeTo": ["confucius", "plato"], "tradition": "daoist",
    },
    "analects": {
        "canonicalTitleEn": "Analects", "attributedAuthor": "confucius",
        "authorConfidence": "compiled", "doNotAttributeTo": ["laozi"], "tradition": "confucian",
    },
}


def test_every_pair_self_validates() -> None:
    out = mhn.mine(RECORDS)
    pairs = out["dpo"]
    assert pairs, "expected some mined pairs"
    verify = provenance_faithful(RECORDS)
    for p in pairs:
        assert set(p) >= {"prompt", "chosen", "rejected", "metadata"}
        # the honesty invariant:
        assert verify(p["rejected"], None, {})["passed"] is False, f"rejected must trip gate: {p['rejected']}"
        assert verify(p["chosen"], None, {})["passed"] is True, f"chosen must pass gate: {p['chosen']}"


def test_all_forbidden_authors_covered() -> None:
    out = mhn.mine(RECORDS)
    authors = {p["metadata"]["forbiddenAuthor"] for p in out["dpo"]}
    assert {"confucius", "plato", "laozi"} <= authors


def test_alias_negatives_present() -> None:
    out = mhn.mine(RECORDS)
    # an alias of confucius ("kongzi") should appear in some rejected
    assert any("kongzi" in p["rejected"].lower() for p in out["dpo"])
    assert any(p["metadata"].get("negativeType") == "alias" for p in out["dpo"])


def test_laundering_and_sibling_types_present() -> None:
    out = mhn.mine(RECORDS)
    types = {p["metadata"].get("negativeType") for p in out["dpo"]}
    assert "laundering" in types  # indirect phrasing (possessive/passive/work-by)
    assert "sibling" in types     # confucius authors the Analects yet is forbidden on the Dao De Jing


def test_no_duplicate_rejecteds_per_prompt() -> None:
    out = mhn.mine(RECORDS)
    seen = set()
    for p in out["dpo"]:
        key = (p["prompt"], p["rejected"])
        assert key not in seen, f"duplicate pair: {key}"
        seen.add(key)


def test_real_corpus_is_all_honest() -> None:
    # over the seeded corpus, the self-validation invariant must hold for every pair
    out = mhn.mine()  # default: load seeded + user records
    verify = provenance_faithful(mhn._records())
    assert out["dpo"]
    assert all(verify(p["rejected"], None, {})["passed"] is False for p in out["dpo"])


def main() -> int:
    test_every_pair_self_validates()
    test_all_forbidden_authors_covered()
    test_alias_negatives_present()
    test_laundering_and_sibling_types_present()
    test_no_duplicate_rejecteds_per_prompt()
    test_real_corpus_is_all_honest()
    print("test_mine_hard_negatives: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
