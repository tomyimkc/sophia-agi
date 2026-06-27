# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Dependency-free byte-level tokenizer with reserved provenance special tokens.

Why byte-level: it is reversible, deterministic, needs **no training**, handles
the corpus's EN+中文 mix natively (UTF-8 bytes), and depends on nothing — so it
runs in CI, on the pure-Python charter path, and identically on the DGX Spark and
the M3. A learned BPE merge table can be layered on later (``merges`` hook below)
without changing the special-token IDs.

The point of this file in the from-scratch plan: **reserve the born-gated vocab
now.** Idea #1 in ``docs/06-Roadmap/From-Scratch-LLM-Brainstorm.md`` trains a
model to emit an inline attribution trail. If we add those tokens only later, the
embedding table has to be resized and re-initialised. By reserving them at id
256+ from day one, a plain-text pretrain and a born-gated pretrain share one
stable vocabulary.

    tok = ByteProvenanceTokenizer()
    ids = tok.encode("Confucius did not write the Dao De Jing.")
    tok.decode(ids)                       # exact round-trip
    tok.encode_with_specials("X <src>analects</src>")  # special tokens become single ids
"""
from __future__ import annotations

import re

# Reserved provenance / control tokens, in fixed order. IDs are 256 + index, so
# byte values 0..255 keep their natural ids and these never collide. APPEND-ONLY:
# never reorder or remove an entry, or existing checkpoints break.
PROVENANCE_SPECIALS: tuple[str, ...] = (
    "<|endoftext|>",      # document boundary
    "<src>", "</src>",    # inline source span:  <src>analects</src>
    "<conf_hi>", "<conf_lo>",   # calibrated-confidence markers (idea #3)
    "<abstain>",          # the model declines, fail-closed
    "<doNotMergeWith>",   # lineage-separation marker (儒家 / 道家 stay distinct)
    "<doNotAttributeTo>",
)

_BYTE_VOCAB = 256


class ByteProvenanceTokenizer:
    """Byte-level codec; special tokens occupy ids ``256 .. 256+len(specials)-1``."""

    def __init__(self, specials: "tuple[str, ...]" = PROVENANCE_SPECIALS) -> None:
        self.specials = tuple(specials)
        self.special_to_id = {s: _BYTE_VOCAB + i for i, s in enumerate(self.specials)}
        self.id_to_special = {i: s for s, i in self.special_to_id.items()}
        # Longest-first alternation so "</src>" matches before "<src>" etc.
        self._special_re = re.compile(
            "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True))
        )

    # -- accounting -----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return _BYTE_VOCAB + len(self.specials)

    def special_id(self, token: str) -> int:
        return self.special_to_id[token]

    @property
    def eot_id(self) -> int:
        return self.special_to_id["<|endoftext|>"]

    # -- encode / decode ------------------------------------------------------
    def encode(self, text: str) -> "list[int]":
        """Plain text → byte ids (no special-token parsing)."""
        return list(text.encode("utf-8"))

    def encode_with_specials(self, text: str) -> "list[int]":
        """Encode text, turning any literal special-token substring into its id.

        Used for born-gated corpora where the provenance markers are written
        inline. Text between markers is byte-encoded as usual.
        """
        ids: list[int] = []
        pos = 0
        for m in self._special_re.finditer(text):
            if m.start() > pos:
                ids.extend(text[pos:m.start()].encode("utf-8"))
            ids.append(self.special_to_id[m.group()])
            pos = m.end()
        if pos < len(text):
            ids.extend(text[pos:].encode("utf-8"))
        return ids

    def decode(self, ids: "list[int]") -> str:
        """Ids → text. Specials render as their literal string; bytes are
        coalesced into one UTF-8 decode so multi-byte 中文 round-trips."""
        out: list[str] = []
        buf: list[int] = []

        def flush() -> None:
            if buf:
                out.append(bytes(buf).decode("utf-8", errors="replace"))
                buf.clear()

        for i in ids:
            if i in self.id_to_special:
                flush()
                out.append(self.id_to_special[i])
            elif 0 <= i < _BYTE_VOCAB:
                buf.append(i)
            else:
                flush()
                out.append("�")  # unknown id → replacement char
        flush()
        return "".join(out)


__all__ = ["ByteProvenanceTokenizer", "PROVENANCE_SPECIALS"]
