# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The SEIB generalization split is deterministic/disjoint and the leakage audit flags
training examples that would teach a held-out contested entity (anti teaching-to-test)."""

from __future__ import annotations

from tools.seib_generalization_split import (
    audit_leakage,
    author_core,
    corpus_partition,
    entity_of,
    load_contested,
    split_contested,
)


def test_author_core_strips_parenthetical():
    assert author_core("Confucius (compiled by his disciples)") == "confucius"
    assert author_core("Laozi") == "laozi"
    assert author_core(None) == ""


def test_split_is_deterministic_disjoint_and_complete():
    rows = load_contested()
    assert len(rows) >= 2
    a = split_contested(rows)
    b = split_contested(rows)
    assert a == b  # deterministic (hash-based, no RNG)
    train_ids = {e["id"] for e in a["train"]}
    held_ids = {e["id"] for e in a["heldout"]}
    assert train_ids.isdisjoint(held_ids)
    assert len(train_ids) + len(held_ids) == len(rows)


def test_entity_extraction():
    row = {"work": "Dao De Jing", "gold_author": "Laozi"}
    assert entity_of(row) == ("dao de jing", "laozi")


def test_leakage_audit_flags_heldout_entity_mention():
    heldout = [("crime and punishment", "dostoevsky")]
    leaky = ("training/examples/999-x.json", {
        "messages": [{"role": "user", "content": "Did Dostoevsky write Crime and Punishment?"},
                     {"role": "assistant", "content": "Qualify: Dostoevsky is the documented author."}],
        "metadata": {},
    })
    clean = ("training/examples/998-y.json", {
        "messages": [{"role": "user", "content": "Who wrote the Analects?"}], "metadata": {},
    })
    findings = audit_leakage(heldout, [leaky, clean])
    assert len(findings) == 1
    assert findings[0]["file"].endswith("999-x.json")


def test_leakage_audit_clean_when_disjoint():
    heldout = [("the second sex", "simone de beauvoir")]
    examples = [("training/examples/001.json",
                 {"messages": [{"role": "user", "content": "Who wrote the Dao De Jing?"}], "metadata": {}})]
    assert audit_leakage(heldout, examples) == []


def test_corpus_partition_separates_taught_from_clean():
    rows = [
        {"id": "c1", "work": "Analects", "gold_author": "Confucius"},
        {"id": "c2", "work": "The Obscure Unseen Treatise", "gold_author": "Nobody Documented"},
    ]
    examples = [("training/examples/x.json",
                 {"messages": [{"role": "assistant",
                                "content": "The Analects was compiled by Confucius's disciples."}],
                  "metadata": {}})]
    part = corpus_partition(rows, examples)
    taught_ids = {e["id"] for e in part["corpusTaught"]}
    clean_ids = {e["id"] for e in part["corpusClean"]}
    assert "c1" in taught_ids and part["corpusTaught"][0]["corpusExamples"] >= 1
    assert "c2" in clean_ids
    assert part["nCorpusClean"] + part["nCorpusTaught"] == 2
