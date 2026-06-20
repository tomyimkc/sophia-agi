"""Token-aware recursive chunking with overlap and stable chunk ids.

Replaces the "truncate to N chars" ingestion in retrieval. Splits on paragraphs,
then sentences, then words, packing units into ~max_tokens windows with an
overlap tail so retrieval gets coherent, fully-covered, citable chunks instead of
a single truncated head.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — no tokenizer dependency."""
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    id: str
    source: str
    index: int
    text: str
    tokens: int


def _split_units(text: str, max_chars: int) -> list[str]:
    """Recursive split: paragraphs -> sentences -> words, so no unit exceeds max_chars."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sentence in re.split(r"(?<=[.!?。！？])\s+", para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                words = sentence.split()
                buf = ""
                for word in words:
                    if len(buf) + len(word) + 1 > max_chars:
                        if buf:
                            units.append(buf)
                        buf = word
                    else:
                        buf = f"{buf} {word}".strip()
                if buf:
                    units.append(buf)
    return units


def chunk_text(text: str, *, source_id: str = "doc", max_tokens: int = 400, overlap_tokens: int = 50) -> list[Chunk]:
    """Pack units into ~max_tokens chunks with a ~overlap_tokens overlap tail."""
    text = (text or "").strip()
    if not text:
        return []
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    units = _split_units(text, max_chars)
    chunks: list[Chunk] = []
    buf = ""
    for unit in units:
        if buf and len(buf) + len(unit) + 1 > max_chars:
            chunks.append(buf)
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = f"{tail} {unit}".strip() if tail else unit
        else:
            buf = f"{buf} {unit}".strip() if buf else unit
    if buf:
        chunks.append(buf)
    return [
        Chunk(id=f"{source_id}#{i}", source=source_id, index=i, text=c, tokens=estimate_tokens(c))
        for i, c in enumerate(chunks)
    ]
