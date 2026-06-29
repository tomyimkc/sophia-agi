# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for entity-level decontamination (Phase 2)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import assert_entity_decontam as aed  # noqa: E402


def test_vocab_has_known_contested_entities() -> None:
    vocab = aed.build_entity_vocab()
    assert vocab
    for e in ("confucius", "plato", "dao de jing", "analects"):
        assert e in vocab, f"expected {e!r} in entity vocab"


def test_matching_is_whole_word_not_substring() -> None:
    vocab = aed.build_entity_vocab()
    # 'plato' must NOT match inside 'platonist' (the substring trap shingles can't catch)
    assert "plato" not in aed.entities_in("a platonist reading of forms", vocab)
    assert "plato" in aed.entities_in("plato wrote the republic", vocab)
    assert "dao de jing" in aed.entities_in("who wrote the dao de jing?", vocab)


def test_audit_is_deterministic() -> None:
    a = aed.audit()
    b = aed.audit()
    assert a == b


def test_audit_surfaces_known_seib_contamination() -> None:
    # The repo's eval/train share contested entities by construction (SEIB ledger item);
    # the entity layer must SEE it where the shingle layer cannot.
    rep = aed.audit()
    assert rep["nSharedEntities"] > 0
    assert rep["nEvalPromptsFullyCovered"] > 0


def test_threshold_gate_behaviour() -> None:
    assert aed.main(["--fail-covered", "0"]) == 1        # contamination present → fails
    assert aed.main(["--fail-covered", "100000"]) == 0   # generous bound → passes
    assert aed.main([]) == 0                              # default is report-only
