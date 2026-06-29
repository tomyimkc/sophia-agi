#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/eval_okf_multihop_qa.py — the gated third-party recall harness.

These validate the HARNESS WIRING on the committed synthetic fixture (loaders, entity
extraction, both retrieval arms, the recall metric). They do NOT assert a capability
result — whether graph multi-hop beats vector-only on real HotpotQA/2Wiki/MuSiQue is the
gated empirical question, run on the farm.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_okf_multihop_qa",
                                               ROOT / "tools" / "eval_okf_multihop_qa.py")
mhq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mhq)


def _fixture_items():
    raw = mhq._load_jsonl(mhq.FIXTURE)
    return mhq.normalize_fixture(raw)


def test_entity_extraction_finds_bridge() -> None:
    # The bell/edinburgh item: "Edinburgh" must be extracted from BOTH gold paragraphs so
    # the entity index can bridge them (the whole point of graph recall).
    ents_a = mhq.extract_entities("Alexander Graham Bell",
                                  "Alexander Graham Bell was born in Edinburgh in 1847.")
    ents_b = mhq.extract_entities("Edinburgh", "Edinburgh has an elevation of 47 meters.")
    assert "edinburgh" in ents_a and "edinburgh" in ents_b


def test_hotpot_normalizer_marks_gold() -> None:
    raw = [{
        "_id": "x", "question": "q?", "answer": "a",
        "supporting_facts": [["Gold One", 0], ["Gold Two", 1]],
        "context": [["Gold One", ["s0.", "s1."]], ["Distractor", ["d."]], ["Gold Two", ["t0.", "t1."]]],
    }]
    items = mhq.normalize_hotpot(raw)
    golds = {p["title"] for p in items[0]["paragraphs"] if p["gold"]}
    assert golds == {"Gold One", "Gold Two"}
    assert len(items[0]["paragraphs"]) == 3


def test_musique_normalizer_marks_gold() -> None:
    raw = [{
        "id": "m", "question": "q?", "answer": "a",
        "paragraphs": [
            {"title": "A", "paragraph_text": "supporting", "is_supporting": True},
            {"title": "B", "paragraph_text": "distractor", "is_supporting": False},
        ],
    }]
    items = mhq.normalize_musique(raw)
    assert [p["gold"] for p in items[0]["paragraphs"]] == [True, False]


def test_both_arms_produce_valid_recall() -> None:
    items = _fixture_items()
    result = mhq.evaluate(items, ks=(2, 5))
    assert result["items"] == 3
    for arm in ("vector_only", "graph_multihop"):
        for k in (2, 5):
            v = result["arms"][arm][f"recall@{k}"]
            assert 0.0 <= v <= 1.0


def test_decontam_contract() -> None:
    items = _fixture_items()
    dc = mhq.check_decontam(items)
    # the synthetic questions never leak; the only way `clean` is False is a vacuous scan
    assert dc["exactLeaks"] == [] and dc["nearLeaks"] == []
    assert dc["vacuous"] == (dc["trainPromptsScanned"] == 0)
    # honest invariant: a real (non-vacuous) clean scan == no leaks; vacuous is NOT clean
    assert dc["clean"] == (not dc["exactLeaks"] and not dc["nearLeaks"] and not dc["vacuous"])


def test_ner_backend_seam_dispatches() -> None:
    # Register a mock entity backend and confirm the harness routes through it (proves the
    # --ner-backend seam works without needing a real model in CI).
    calls = {"n": 0}

    def _mock(title, text):
        calls["n"] += 1
        return ["shared_bridge_entity"]  # force every paragraph to share one entity

    mhq._ENTITY_BACKENDS["__mock__"] = _mock
    try:
        item = _fixture_items()[0]
        events = mhq._events_for_item(item, ner_backend="__mock__")
        assert calls["n"] == len(item["paragraphs"])
        assert all(e.entities == ("shared_bridge_entity",) for e in events)
        # the harness honors the backend end-to-end via evaluate()
        res = mhq.evaluate(_fixture_items(), ks=(2,), ner_backend="__mock__")
        assert "graph_multihop" in res["arms"]
    finally:
        del mhq._ENTITY_BACKENDS["__mock__"]


def test_unknown_backend_is_treated_as_llm_model() -> None:
    # An unknown backend name must NOT be silently treated as deterministic — it routes to
    # the LLM path, which is fail-closed without an API key. We assert it raises rather than
    # quietly returning floor entities (so a typo'd model id can't fake a "model" run).
    import os
    saved = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")}
    try:
        raised = False
        try:
            mhq.extract_entities("T", "Some text.", backend="definitely-not-a-real-model")
        except Exception:
            raised = True
        assert raised, "unknown backend must hit the fail-closed LLM path, not the floor"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_graph_rank_is_total_order() -> None:
    # _graph_rank must return every paragraph index exactly once (deterministic, complete).
    items = _fixture_items()
    for item in items:
        order = mhq._graph_rank(item)
        assert sorted(order) == list(range(len(item["paragraphs"])))


def main() -> int:
    test_entity_extraction_finds_bridge()
    test_hotpot_normalizer_marks_gold()
    test_musique_normalizer_marks_gold()
    test_both_arms_produce_valid_recall()
    test_decontam_contract()
    test_ner_backend_seam_dispatches()
    test_unknown_backend_is_treated_as_llm_model()
    test_graph_rank_is_total_order()
    print("test_okf_multihop_qa: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
