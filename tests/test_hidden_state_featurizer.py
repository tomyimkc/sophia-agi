# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the R3/W5 seam: agent.activation_probes.build_hidden_state_featurizer.

Offline (any box): the fail-closed contract — no MLX backend => RuntimeError at
construction, bad spec => RuntimeError; and the featurizer-agnostic vector-probe
helpers over a toy featurizer.

Apple Silicon / MLX box (the Mac cluster test bench): the real path — construct
the featurizer, embed two texts, check shape/normalization/determinism, and train
a vector probe on real hidden states. Skipped automatically where mlx is absent.
"""
from __future__ import annotations

import pytest

from agent.activation_probes import (
    LinearProbe,
    build_hidden_state_featurizer,
    evaluate_vector_probe,
    train_vector_probe,
)


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        return True
    except Exception:
        return False


# ---------------- offline contract (runs everywhere) ----------------

@pytest.mark.skipif(_mlx_available(), reason="offline fail-closed contract; mlx present here")
def test_fails_closed_without_mlx_backend():
    with pytest.raises(RuntimeError, match="MLX backend"):
        build_hidden_state_featurizer("mlx")


def test_unknown_spec_fails_closed():
    with pytest.raises(RuntimeError, match="unknown featurizer spec"):
        build_hidden_state_featurizer("cuda-something")


def _toy_featurizer(text: str) -> list[float]:
    """Deterministic 3-dim stand-in: counts of trust/doubt/other tokens."""
    words = (text or "").lower().split()
    trust = sum(w in ("verified", "cited", "confirmed") for w in words)
    doubt = sum(w in ("fabricate", "invent", "guess") for w in words)
    return [float(trust), float(doubt), float(len(words))]


def test_vector_probe_trains_and_separates_toy_classes():
    rows = [
        {"id": "p1", "text": "verified cited confirmed", "label": True},
        {"id": "p2", "text": "cited verified", "label": True},
        {"id": "n1", "text": "fabricate invent guess", "label": False},
        {"id": "n2", "text": "guess fabricate", "label": False},
    ]
    probe = train_vector_probe(rows, _toy_featurizer, name="toy")
    assert isinstance(probe, LinearProbe) and len(probe.weights) == 3
    report = evaluate_vector_probe(probe, rows, _toy_featurizer)
    assert report["metrics"]["accuracy"] == 1.0
    assert report["candidateOnly"] is True


def test_vector_probe_single_class_fails_closed_to_zero_probe():
    rows = [{"id": "p1", "text": "verified", "label": True}]
    probe = train_vector_probe(rows, _toy_featurizer)
    assert all(w == 0.0 for w in probe.weights)
    # a zero probe scores 0.5 everywhere and flags at threshold, never separating
    rep = evaluate_vector_probe(probe, rows, _toy_featurizer)
    assert rep["n"] == 1


# ---------------- real MLX path (Mac cluster test bench) ----------------

@pytest.mark.skipif(not _mlx_available(), reason="requires MLX backend (Apple Silicon bench)")
def test_real_hidden_state_featurizer_shape_and_determinism():
    fz = build_hidden_state_featurizer("mlx")
    v1 = fz("The Analects were compiled by Confucius' disciples.")
    v2 = fz("The Analects were compiled by Confucius' disciples.")
    v3 = fz("Completely different sentence about thermodynamics.")
    assert len(v1) > 100, "hidden size expected (model hidden dim)"
    assert v1 == v2, "pure forward pass must be deterministic"
    assert v1 != v3
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-3, "features must be L2-normalized"


@pytest.mark.skipif(not _mlx_available(), reason="requires MLX backend (Apple Silicon bench)")
def test_real_vector_probe_over_hidden_states():
    fz = build_hidden_state_featurizer("mlx")
    rows = [
        {"id": "p1", "text": "I cannot verify this attribution; I abstain.", "label": True},
        {"id": "p2", "text": "No verified source exists, so I will not assert an author.", "label": True},
        {"id": "n1", "text": "It was definitely written by Laozi in 500 BC, trust me.", "label": False},
        {"id": "n2", "text": "The author is certainly Zhuangzi; no citation needed.", "label": False},
    ]
    probe = train_vector_probe(rows, fz, name="hidden_state_probe")
    rep = evaluate_vector_probe(probe, rows, fz)
    # train-set separation is a wiring check, NOT a capability claim
    assert rep["metrics"]["accuracy"] >= 0.75
