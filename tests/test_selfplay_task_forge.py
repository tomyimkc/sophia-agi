# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/selfplay_task_forge.py (A5 acceptance contract + determinism)."""
from __future__ import annotations

from tools.selfplay_task_forge import decontaminate, forge_tasks, load_graph

GRAPH = {
    "analects": {"recordType": "text", "textId": "analects", "canonicalTitleEn": "Analects",
                 "tradition": "confucian", "attributedAuthor": "confucius",
                 "authorConfidence": "compiled", "doNotAttributeTo": ["laozi"]},
    "spring_autumn": {"recordType": "text", "textId": "spring_autumn",
                      "canonicalTitleEn": "Spring and Autumn Annals",
                      "tradition": "confucian", "attributedAuthor": "confucius",
                      "authorConfidence": "attributed"},
    "mencius": {"recordType": "text", "textId": "mencius", "canonicalTitleEn": "Mencius",
                "tradition": "confucian", "attributedAuthor": "mencius",
                "authorConfidence": "attributed"},
    "republic": {"recordType": "text", "textId": "republic", "canonicalTitleEn": "Republic",
                 "tradition": "platonic", "attributedAuthor": "plato",
                 "authorConfidence": "attributed"},
}


def test_forge_emits_all_task_types_with_acceptance_stamps():
    result = forge_tasks(GRAPH, seed=0, limit=50)
    assert result["ok"]
    types = {t["taskType"] for t in result["tasks"]}
    assert {"trap_uncertain", "trap_forbidden", "hop2_tradition"} <= types
    for t in result["tasks"]:
        assert t["hops"] >= 2 and t["requiredEvidence"] and t["verifier"]
        assert t["candidateOnly"] is True
        assert t["shortcutScreened"] in ("lexical", "lexical+model")


def test_uncertain_confidence_never_yields_flat_author_claim():
    result = forge_tasks(GRAPH, seed=0)
    analects = [t for t in result["tasks"] if "analects" in t["id"]]
    assert analects, "compiled-confidence text must generate tasks"
    assert all(t["taskType"] in ("trap_uncertain", "trap_forbidden") for t in analects), \
        "a compiled text must only produce hedge/reject tasks, never an author-claim task"
    hedge = [t for t in analects if t["taskType"] == "trap_uncertain"][0]
    assert hedge["goldAction"] == "hedge" and "compiled" in hedge["gold"]


def test_shortcut_lexical_screen_and_determinism():
    r1 = forge_tasks(GRAPH, seed=7)
    r2 = forge_tasks(GRAPH, seed=7)
    assert r1["tasks"] == r2["tasks"], "same seed must be byte-deterministic"
    for t in r1["tasks"]:
        if t["taskType"].startswith("hop2"):
            assert str(t["gold"]).lower() not in t["question"].lower()


def test_model_screen_drops_always_solved_tasks():
    always_right = lambda q: "confucian"  # noqa: E731 - solves every tradition task
    r = forge_tasks(GRAPH, seed=0, solver_fn=always_right)
    tradition_golds = [t for t in r["tasks"]
                       if t["taskType"] == "hop2_tradition" and t["gold"] == "confucian"]
    assert not tradition_golds, "raw-solver-always-right tasks must be dropped (no-shortcut)"
    assert r["drops"]["solver_always_right"] >= 1


def test_real_graph_loads_and_decontaminates():
    graph = load_graph()
    assert len(graph) >= 5
    result = forge_tasks(graph, seed=0, limit=10)
    assert result["ok"] and len(result["tasks"]) <= 10
    kept, dropped = decontaminate(result["tasks"])
    assert dropped >= 0, "guard must be available in-repo (fail-visible -1 otherwise)"
    assert len(kept) + dropped == len(result["tasks"])
