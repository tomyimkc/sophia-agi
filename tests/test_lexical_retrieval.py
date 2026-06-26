#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the deterministic lexical-vector retriever (numpy-free)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.lexical_embed import DIM, cosine, embed, rank  # noqa: E402


def test_embedding_is_l2_normalised_and_bounded() -> None:
    vec = embed("provenance-aware verifiable reasoning")
    assert len(vec) == DIM
    norm = sum(value * value for value in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-9
    # self-cosine is 1; orthogonal-ish unrelated text is well below 1.
    assert abs(cosine(vec, vec) - 1.0) < 1e-9
    assert cosine(vec, embed("xyzzy plugh frobnicate")) < 0.5


def test_embedding_is_deterministic_across_processes() -> None:
    # Reproducibility hinges on hashlib (stable), NOT the salted built-in hash().
    in_process = embed("philosophical foundations")
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from agent.lexical_embed import embed;"
        "print(','.join('%%.10f' %% v for v in embed('philosophical foundations')))" % str(ROOT)
    )
    out = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=30, check=True)
    other = [float(x) for x in out.stdout.strip().split(",")]
    assert other == [round(v, 10) or 0.0 for v in [float("%.10f" % v) for v in in_process]]


def test_vector_retrieval_beats_keyword_on_subword_match() -> None:
    # The whole point over keyword overlap: sub-word recall. "philosophy foundation"
    # shares NO exact tokens with "philosophical foundations" (so keyword scores it 0),
    # but the n-gram vectors are close.
    from agent.retrieval import _score, _tokenize

    query = "philosophical foundations"
    morph = ("morph", "the philosophy foundation of ethics")
    unrelated = ("noise", "quantum chromodynamics and gluon confinement")

    q_tokens = _tokenize(query)
    assert _score(q_tokens, morph[1]) == 0.0  # keyword overlap is blind to the morphological variant

    ranked = rank(query, [morph, unrelated], top_k=2)
    assert ranked[0][0] == "morph"  # vector retrieval recovers it
    assert ranked[0][1] > 0.0
    # and it ranks the morphological match strictly above unrelated text
    scores = dict(ranked)
    assert scores["morph"] > scores.get("noise", 0.0)


def test_retrieve_vector_mode_runs_offline() -> None:
    # Smoke: the live path's "vector" tier returns SourceChunks with no model/network.
    from agent.retrieval import SourceChunk, retrieve

    results = retrieve("provenance verifiable reasoning", top_k=3, mode="vector")
    assert isinstance(results, list)
    assert all(isinstance(chunk, SourceChunk) for chunk in results)


def main() -> int:
    test_embedding_is_l2_normalised_and_bounded()
    test_embedding_is_deterministic_across_processes()
    test_vector_retrieval_beats_keyword_on_subword_match()
    test_retrieve_vector_mode_runs_offline()
    print("test_lexical_retrieval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
