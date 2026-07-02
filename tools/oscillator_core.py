# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Shared Kuramoto / dynamical-systems primitives for the O-series epistemics tools.

Inspiration: unconv-ai/Un-0 generates images by integrating coupled-oscillator
(Kuramoto) dynamics and reading out the settled state. The O-series reuses the SAME
mechanism as a *verification* signal for an LLM: run candidate answers (or evidence
sources) as coupled oscillators and read the **order parameter r in [0,1]** — how
coherently the population settles — as a confidence / consensus score.

This module is deliberately dependency-light (numpy only; numpy>=1.26 is already a
hard repo dep via agent.vector_store). It contains NO model call and NO training; it
is a pure, deterministic numerical core so the O-tools stay fail-closed and testable
offline.

Honesty note: `hash_embed` is a token-hashing stand-in for a real semantic embedder —
the SAME kind of documented seam as agent.activation_probes.build_hidden_state_featurizer.
Its geometry captures lexical overlap, not deep meaning; a real sentence embedder is
the drop-in that makes the geometry semantic. Every consumer surfaces this.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
EMBED_DIM_DEFAULT = 64

# --- semantic-embedder seam -------------------------------------------------
# `hash_embed` is a lexical stand-in. When OSC_EMBED_BACKEND=minilm, it delegates
# to a real sentence embedder (sentence-transformers/all-MiniLM-L6-v2, loaded
# offline from the HF cache) so the coherence/residual geometry becomes semantic.
# The default (unset) path stays the pure blake2b hash so the offline unit tests
# and CI are unchanged. This is the single chokepoint feeding O1/O3/O4.
_MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MINILM = None
_MINILM_DIM = 384
_EMBED_CACHE: dict[str, np.ndarray] = {}


def active_embed_backend() -> str:
    """Which embedder `hash_embed` is currently delegating to (honesty surface)."""
    if os.environ.get("OSC_EMBED_BACKEND", "").strip().lower() == "minilm":
        return f"minilm:{_MINILM_NAME}"
    return "hash:blake2b-lexical"


def _load_minilm():
    global _MINILM
    if _MINILM is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_MINILM_NAME)
        mdl = AutoModel.from_pretrained(_MINILM_NAME)
        mdl.eval()
        _MINILM = (tok, mdl, torch)
    return _MINILM


def _semantic_embed(text: str) -> np.ndarray:
    """Mean-pooled, L2-normalized all-MiniLM-L6-v2 vector (empty text -> zero vector).

    Preserves hash_embed's two load-bearing invariants: non-empty text yields a
    unit vector; empty/whitespace yields the zero 'no content' sentinel.
    """
    s = str(text)
    if not s.strip():
        return np.zeros(_MINILM_DIM, dtype=np.float64)
    cached = _EMBED_CACHE.get(s)
    if cached is not None:
        return cached
    tok, mdl, torch = _load_minilm()
    with torch.no_grad():
        enc = tok([s], padding=True, truncation=True, max_length=256, return_tensors="pt")
        out = mdl(**enc)
        last = out.last_hidden_state  # (1, seq, hidden)
        mask = enc["attention_mask"].unsqueeze(-1).type_as(last)
        summed = (last * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        vec = summed / counts
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)[0]
        v = vec.cpu().numpy().astype(np.float64)
    _EMBED_CACHE[s] = v
    return v


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())


def hash_embed(text: str, dim: int = EMBED_DIM_DEFAULT) -> np.ndarray:
    """Deterministic bag-of-token-hashes unit vector (a stand-in for a semantic embedder).

    Signed hashing (feature-hashing trick): each token maps to a bucket with a +/-1 sign,
    so unrelated tokens do not systematically inflate similarity. Returns a unit vector
    (zero vector for empty text, which callers treat as 'no content').

    When OSC_EMBED_BACKEND=minilm, delegates to a real semantic embedder (all-MiniLM-L6-v2);
    `dim` is then ignored (the model's native 384-d output is returned). The empty-text ->
    zero-vector and non-empty -> unit-norm invariants hold on both paths.
    """
    if os.environ.get("OSC_EMBED_BACKEND", "").strip().lower() == "minilm":
        return _semantic_embed(text)
    v = np.zeros(int(dim), dtype=np.float64)
    toks = _tokens(text)
    if not toks:
        return v
    for t in toks:
        h = int(hashlib.blake2b(t.encode("utf-8"), digest_size=8).hexdigest(), 16)
        idx = h % int(dim)
        sign = 1.0 if (h >> 1) & 1 else -1.0
        v[idx] += sign
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def text_similarity(a: str, b: str, *, dim: int = EMBED_DIM_DEFAULT) -> float:
    """Cosine similarity of hash embeddings, clipped to [0,1] (negatives -> 0 coupling)."""
    va, vb = hash_embed(a, dim), hash_embed(b, dim)
    if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
        return 0.0
    return float(max(0.0, np.dot(va, vb)))


def similarity_matrix(texts: Sequence[str], *, dim: int = EMBED_DIM_DEFAULT) -> np.ndarray:
    """Symmetric [N,N] coupling matrix K_ij = sim(text_i, text_j), zero diagonal."""
    embs = np.stack([hash_embed(t, dim) for t in texts]) if texts else np.zeros((0, dim))
    if embs.shape[0] == 0:
        return np.zeros((0, 0))
    k = embs @ embs.T
    k = np.clip(k, 0.0, None)
    np.fill_diagonal(k, 0.0)
    return k


def order_parameter(theta: np.ndarray) -> float:
    """Kuramoto order parameter r = (1/N)|sum_j exp(i*theta_j)| in [0,1].

    r=1 is full phase synchrony (unanimous), r->0 is incoherence (a split population).
    """
    if theta.size == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * theta))))


def run_kuramoto(
    coupling: np.ndarray,
    *,
    steps: int = 40,
    dt: float = 0.1,
    gain: float = 2.0,
    seed: int = 0,
    omega: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Integrate omega=0 Kuramoto dynamics under a fixed coupling matrix (Euler, as in Un-0).

    dtheta_i/dt = omega_i + (gain/N) * sum_j K_ij * sin(theta_j - theta_i)

    With omega=0 and non-negative coupling, a well-connected (mutually similar) population
    synchronizes -> high final r; a population that splits into dissimilar clusters settles
    at low global r. Returns (final_theta, r_history) where r_history[t] is r after step t.
    This is a strict *generalization* of a majority vote: identical answers couple at 1.0
    (they fully sync), near-duplicates couple partially (they still reinforce), whereas a
    vote-count sees only exact-match buckets.
    """
    n = coupling.shape[0]
    if n == 0:
        return np.zeros(0), []
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    w = np.zeros(n) if omega is None else np.asarray(omega, dtype=np.float64)
    hist: list[float] = []
    for _ in range(int(steps)):
        # pairwise phase differences -> coupling torque
        diff = theta[None, :] - theta[:, None]        # [i,j] = theta_j - theta_i
        torque = (coupling * np.sin(diff)).sum(axis=1)
        theta = theta + dt * (w + (gain / n) * torque)
        hist.append(order_parameter(theta))
    return theta, hist


def consensus_r(texts: Sequence[str], *, dim: int = EMBED_DIM_DEFAULT,
                steps: int = 40, dt: float = 0.1, gain: float = 2.0, seed: int = 0) -> float:
    """End-to-end: embed texts -> similarity coupling -> Kuramoto -> final order parameter r.

    The training-free confidence readout O1 uses. With <2 texts, coherence is undefined:
    a single sample returns 1.0 (trivially self-coherent), empty returns 0.0.
    """
    n = len(texts)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    k = similarity_matrix(texts, dim=dim)
    _, hist = run_kuramoto(k, steps=steps, dt=dt, gain=gain, seed=seed)
    return hist[-1] if hist else 0.0