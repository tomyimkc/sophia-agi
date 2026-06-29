# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Corpus loading + tokenisation for the from-scratch GPT (dependency-free).

Turns ``training/corpus.jsonl`` (528 bilingual source-discipline chat rows) into
a flat token stream with ``<|endoftext|>`` between documents. Kept torch-free so
it runs in CI and on the pure-Python charter path; ``train.py`` does the tensor
batching. A tiny synthetic fallback keeps the smoke test hermetic when the corpus
file is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

from pretraining.gpt.tokenizer import ByteProvenanceTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "training" / "corpus.jsonl"

_ROLE_TAG = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def corpus_documents(path: "Path | None" = None) -> "list[str]":
    """Each chat row → one document string (role-tagged, newline-joined turns)."""
    path = Path(path) if path else DEFAULT_CORPUS
    docs: list[str] = []
    if not path.exists():
        return docs
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        msgs = row.get("messages", [])
        turns = [f"{_ROLE_TAG.get(m.get('role', ''), '')} {m.get('content', '')}".strip()
                 for m in msgs]
        if turns:
            docs.append("\n".join(turns))
    return docs


def _synthetic_documents(n: int = 64) -> "list[str]":
    """Deterministic fallback so the smoke path never depends on the corpus file."""
    seeds = [
        "Confucius did not write the Dao De Jing. 老子 is the traditional attribution.",
        "Festinger developed cognitive dissonance in the 1950s, not Freud.",
        "Socrates wrote no books; Plato wrote the Republic.",
        "When sources conflict or are absent, abstain rather than fabricate.",
    ]
    return [seeds[i % len(seeds)] for i in range(n)]


def token_stream(
    tokenizer: "ByteProvenanceTokenizer | None" = None,
    *, path: "Path | None" = None, with_specials: bool = False,
) -> "list[int]":
    """Flatten the corpus to one id list, ``<|endoftext|>``-separated.

    ``with_specials=True`` parses inline provenance markers (born-gated corpora);
    the plain corpus has none, so the default byte path is used.
    """
    tok = tokenizer or ByteProvenanceTokenizer()
    docs = corpus_documents(path) or _synthetic_documents()
    enc = tok.encode_with_specials if with_specials else tok.encode
    ids: list[int] = []
    for doc in docs:
        ids.extend(enc(doc))
        ids.append(tok.eot_id)
    return ids


def train_val_split(ids: "list[int]", val_frac: float = 0.1) -> "tuple[list[int], list[int]]":
    n_val = max(1, int(len(ids) * val_frac))
    return ids[:-n_val], ids[-n_val:]


__all__ = ["corpus_documents", "token_stream", "train_val_split", "DEFAULT_CORPUS"]
