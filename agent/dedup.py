# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Near-duplicate collapse for retrieval candidate pools.

Sophia's corpus carries genuine near-duplicates: chunking overlap, and teacher-example
variants of the same item (``...-r0`` / ``...-r1``, paraphrased re-asks). When those land in
one candidate pool they waste top-k slots and let the sparse view's redundant hits crowd out
diverse, relevant chunks. This module collapses them with a deterministic, dependency-light
**word-shingle Jaccard** clusterer: greedy single-pass assignment to the first cluster whose
representative is similar enough, keeping the earliest (highest-ranked) member as the survivor.

Offline, CPU-only, order-stable — same input → same survivors. O(n·c) over a small candidate
pool (n items, c clusters), which is ample at retrieval-pool sizes (tens, not millions).

Honest bound: this collapses **near-identical surface text**, not distinct-but-related chunks
(two different books with the same question template are NOT merged — that is a ranking/field
problem, handled by weighting/rerank, not dedup). So it improves result *diversity* and trims
redundant variants; it is not a recall fix on its own.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

_WORD_RE = re.compile(r"[a-z0-9一-鿿]+")

#: Default Jaccard threshold to treat two texts as near-duplicates. 0.8 is conservative:
#: r0/r1 variants and chunk-overlap pairs clear it; merely on-topic chunks do not.
DEFAULT_THRESHOLD = 0.8


def shingles(text: str, *, k: int = 4) -> frozenset[str]:
    """Word ``k``-shingles of ``text`` (overlapping k-grams), lowercased.

    Falls back to the bag of unigrams when the text is shorter than ``k`` words, so short
    titles still compare meaningfully instead of degenerating to the empty set.
    """
    words = _WORD_RE.findall((text or "").lower())
    if len(words) < k:
        return frozenset(words)
    return frozenset(" ".join(words[i:i + k]) for i in range(len(words) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def dedupe(
    items: Iterable[T],
    *,
    text_of: Callable[[T], str],
    threshold: float = DEFAULT_THRESHOLD,
    k: int = 4,
) -> tuple[list[T], int]:
    """Collapse near-duplicates, keeping the first (best-ranked) member of each cluster.

    Returns ``(survivors, dropped_count)``. Input order is preserved among survivors, so a
    ranked pool stays ranked. Deterministic: greedy single-pass, ties resolved by position.
    """
    survivors: list[T] = []
    sigs: list[frozenset[str]] = []
    dropped = 0
    for item in items:
        sig = shingles(text_of(item), k=k)
        if any(jaccard(sig, prev) >= threshold for prev in sigs):
            dropped += 1
            continue
        survivors.append(item)
        sigs.append(sig)
    return survivors, dropped


def dedupe_chunks(chunks, *, threshold: float = DEFAULT_THRESHOLD):
    """Convenience wrapper for :class:`agent.retrieval.SourceChunk`-like objects.

    Keys on the chunk **body** (``excerpt``/``text``), not the title: variant labels like
    ``...-r0`` / ``...-r1`` differ in the title but share identical content, and that content
    is what makes them duplicates. Falls back to the title only when the body is empty.
    """
    def _text(c) -> str:  # noqa: ANN001
        body = getattr(c, "excerpt", "") or getattr(c, "text", "")
        return body or getattr(c, "title", "")

    survivors, _ = dedupe(chunks, text_of=_text, threshold=threshold)
    return survivors


__all__ = ["DEFAULT_THRESHOLD", "dedupe", "dedupe_chunks", "jaccard", "shingles"]
