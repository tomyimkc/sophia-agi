# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic order-k Markov corpus with a *known* irreducible entropy.

The point of a scaling-law study is to fit ``L = E + A·x^-p`` and check it. To check it
honestly you need to know the true ``E``. So instead of scraping text (whose entropy is
unknown), we synthesize the corpus from an order-``k`` Markov source with a seeded,
peaky conditional distribution. The source's average conditional entropy is computable
in closed form (``source_entropy``) and *is* the irreducible loss any model is bounded by.

This makes every downstream claim falsifiable: a fitted floor that lands far from the
analytic entropy means the fit (or the model) is wrong, and the study says so.

Different ``mix`` labels (see ``mixed_corpus``) let the data-mixing study build corpora
from several distinct sources with controllable ratios — the same machinery DeepSeek's
data direction calls 配比 (mixture-ratio) research, at toy scale.
"""
from __future__ import annotations

import math
import random


def make_source(vocab: int, order: int, *, seed: int = 0, peak: float = 4.0) -> dict:
    """Build a seeded order-``order`` Markov transition table.

    ``peak`` controls how concentrated each conditional is (higher = lower entropy =
    more learnable structure). Returns {context_tuple: [p_0..p_{V-1}]}.
    """
    rng = random.Random(seed)
    table: dict[tuple, list[float]] = {}

    def fill(prefix: tuple) -> None:
        if len(prefix) == order:
            logits = [rng.gauss(0.0, peak) for _ in range(vocab)]
            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            z = sum(exps)
            table[prefix] = [e / z for e in exps]
            return
        for s in range(vocab):
            fill(prefix + (s,))

    fill(())
    return {"vocab": vocab, "order": order, "table": table, "seed": seed, "peak": peak}


def drifted_source(source: dict, drift: float, *, seed: int = 0) -> dict:
    """A copy of ``source`` whose conditionals are perturbed by ``drift`` — a stand-in
    for an imperfect synthetic-data generator. ``drift=0`` reproduces the source exactly;
    larger drift pushes each conditional toward noise, modeling distribution shift /
    quality loss that drives model collapse when synthetic data dominates."""
    rng = random.Random(seed)
    V = source["vocab"]
    new_table: dict[tuple, list[float]] = {}
    for ctx, probs in source["table"].items():
        noise = [rng.random() for _ in range(V)]
        zn = sum(noise) or 1.0
        noise = [x / zn for x in noise]
        mixed = [(1 - drift) * p + drift * n for p, n in zip(probs, noise)]
        z = sum(mixed) or 1.0
        new_table[ctx] = [x / z for x in mixed]
    return {"vocab": V, "order": source["order"], "table": new_table,
            "seed": seed, "peak": source.get("peak"), "drift": drift}


def source_entropy(source: dict) -> float:
    """Average conditional entropy of the source, in nats — the irreducible loss ``E``.

    Contexts are equiprobable by construction (uniform start + the source is its own
    stationary-ish generator over a long run), so we average conditional entropy
    uniformly over contexts. Reported as a *reference floor*, not an exact stationary
    entropy — honest about the approximation.
    """
    table = source["table"]
    if not table:
        return float("nan")
    total = 0.0
    for probs in table.values():
        total += -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return total / len(table)


def sample_stream(source: dict, n: int, *, seed: int = 0) -> "list[int]":
    """Sample ``n`` symbols from the Markov source."""
    rng = random.Random(seed)
    V, order, table = source["vocab"], source["order"], source["table"]
    ctx = tuple(rng.randrange(V) for _ in range(order))
    out: list[int] = list(ctx)
    while len(out) < n:
        probs = table[ctx]
        r = rng.random()
        acc = 0.0
        nxt = V - 1
        for s, p in enumerate(probs):
            acc += p
            if r <= acc:
                nxt = s
                break
        out.append(nxt)
        ctx = tuple(out[-order:])
    return out[:n]


def to_examples(stream: "list[int]", context: int) -> "list[tuple[list[int], int]]":
    """Slice a symbol stream into (context-window, next-symbol) training pairs."""
    out: list[tuple[list[int], int]] = []
    for i in range(context, len(stream)):
        out.append((stream[i - context:i], stream[i]))
    return out


def mixed_corpus(sources: "list[dict]", weights: "list[float]", n: int,
                 *, context: int, seed: int = 0) -> "list[tuple[list[int], int]]":
    """Build a training set by sampling ``n`` total examples from several sources in
    proportion to ``weights`` (the data 配比). Weights are normalized; counts are
    rounded to sum to ``n``. Sources must share a vocab."""
    assert len(sources) == len(weights) and sources, "need matching sources/weights"
    tot = float(sum(weights)) or 1.0
    counts = [int(round(n * w / tot)) for w in weights]
    # fix rounding drift so counts sum to n
    counts[0] += n - sum(counts)
    examples: list[tuple[list[int], int]] = []
    for idx, (src, cnt) in enumerate(zip(sources, counts)):
        if cnt <= 0:
            continue
        stream = sample_stream(src, cnt + context, seed=seed + 1000 * idx)
        examples.extend(to_examples(stream, context))
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:n]


__all__ = [
    "make_source", "drifted_source", "source_entropy", "sample_stream",
    "to_examples", "mixed_corpus",
]
