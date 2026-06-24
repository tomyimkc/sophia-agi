# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Axis-vector composition + orthogonalization (pure stdlib).

Takes OCEAN target signs only — never an MBTI string (veneer-invariance).
Default scheme is C4 soft-projection (best signal retention per arXiv:2602.15847).
Orthogonalization reduces, it does NOT eliminate, behavioral cross-trait
interference — validate each axis behaviorally after composition.
"""
from __future__ import annotations

from agent.steering.vectors import Vector, add, cosine, dot, normalize, scale, sub


def gram_matrix(vs: "dict[str, Vector]") -> dict:
    keys = sorted(vs)
    out: dict = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            out[f"{keys[i]}|{keys[j]}"] = round(cosine(vs[keys[i]], vs[keys[j]]), 6)
    return out


def soft_project(vs: "dict[str, Vector]", beta: float = 0.5) -> "dict[str, Vector]":
    keys = sorted(vs)
    hats = {k: normalize(vs[k]) for k in keys}
    out: dict = {}
    for i in keys:
        d = vs[i][:]
        for j in keys:
            if j == i:
                continue
            d = sub(d, scale(hats[j], beta * dot(d, hats[j])))
        out[i] = d
    return out


def gram_schmidt(vs: "dict[str, Vector]") -> "dict[str, Vector]":
    keys = sorted(vs)
    basis: list = []
    out: dict = {}
    for k in keys:
        u = vs[k][:]
        for b in basis:
            u = sub(u, scale(b, dot(u, b)))
        u = normalize(u)
        basis.append(u)
        out[k] = u
    return out


def compose_vectors(vs: "dict[str, Vector]", alphas: "dict[str, float]", *,
                    scheme: str = "soft_proj") -> "tuple[Vector, dict]":
    gram_before = gram_matrix(vs)
    if scheme == "soft_proj":
        ortho = soft_project(vs)
    elif scheme == "gram_schmidt":
        ortho = gram_schmidt(vs)
    elif scheme == "raw":
        ortho = {k: v[:] for k, v in vs.items()}
    else:
        raise ValueError(f"unknown scheme {scheme!r}; use soft_proj|gram_schmidt|raw")
    keys = sorted(ortho)
    dim = len(next(iter(ortho.values()))) if ortho else 0
    composed: Vector = [0.0] * dim
    per_axis_norm: dict = {}
    for k in keys:
        vhat = normalize(ortho[k])
        per_axis_norm[k] = round(sum(x * x for x in ortho[k]) ** 0.5, 6)
        composed = add(composed, scale(vhat, alphas.get(k, 0.0)))
    manifest = {
        "scheme": scheme, "axes": keys, "gram": gram_before,
        "per_axis_norm": per_axis_norm, "alphas": {k: alphas.get(k, 0.0) for k in keys},
        "normalized": True,
    }
    return composed, manifest
