# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Born-gated corpus: structured provenance records → inline-marked training text.

This is idea #1 from ``docs/06-Roadmap/From-Scratch-LLM-Brainstorm.md`` made
runnable. Instead of bolting provenance on at inference time, we write it *into*
the training text as the special tokens the tokenizer already reserves
(``<src>``, ``<conf_hi/lo>``, ``<doNotAttributeTo>``). A model pretrained on this
learns *"a title co-occurs with its source token and its confidence marker, and
explicitly with the authors it must NOT be attributed to"* — the structure
Sophia normally enforces with a gate.

Dependency-free and deterministic, so it runs in CI and on the charter path. The
generated text is fed to ``token_stream(..., with_specials=True)``.

    from pretraining.gpt.born_gated import born_gated_documents
    docs = born_gated_documents()            # from data/attributions.json
    # "The Dao De Jing is attributed to <src>laozi</src> <conf_lo> .
    #  Do not attribute it to <doNotAttributeTo>confucius</doNotAttributeTo> ..."
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDS = ROOT / "data" / "attributions.json"

# Confidence vocabulary → reserved marker. Anything not clearly settled is LOW,
# fail-closed (uncertainty defaults to caution, never to false confidence).
_HI = {"attributed", "consensus", "documented", "established", "certain"}
# Anything not in _HI (legendary / disputed / uncertain / compiled / traditional /
# contested / ...) is treated as LOW confidence by _conf_marker, fail-closed.


def _conf_marker(author_confidence: str) -> str:
    return "<conf_hi>" if author_confidence.lower() in _HI else "<conf_lo>"


def _record_line(rec: dict) -> "str | None":
    """One born-gated sentence for a record, or None if it lacks the fields
    needed to be honest (fail-closed: no title or no source → skip, don't guess)."""
    title = rec.get("canonicalTitleEn") or rec.get("canonicalTitle") or rec.get("titleEn")
    author = rec.get("attributedAuthor") or rec.get("source") or rec.get("author")
    if not title or not author:
        return None
    conf = _conf_marker(str(rec.get("authorConfidence", "")))
    parts = [f"The {title} is attributed to <src>{author}</src> {conf} ."]
    not_authors = rec.get("doNotAttributeTo") or []
    for wrong in not_authors:
        parts.append(f"Do not attribute it to <doNotAttributeTo>{wrong}</doNotAttributeTo> .")
    tradition = rec.get("tradition")
    if tradition:
        parts.append(f"Tradition: <src>{tradition}</src> ; keep lineages distinct .")
    return " ".join(parts)


def load_records(path: "Path | None" = None) -> "list[dict]":
    path = Path(path) if path else DEFAULT_RECORDS
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [v for v in data.values() if isinstance(v, dict)]
    if isinstance(data, list):
        return [v for v in data if isinstance(v, dict)]
    return []


def _synthetic_records() -> "list[dict]":
    """Hermetic fallback so the path never depends on the data file."""
    return [
        {"canonicalTitleEn": "Dao De Jing", "attributedAuthor": "laozi",
         "authorConfidence": "legendary", "tradition": "daoist",
         "doNotAttributeTo": ["confucius", "socrates"]},
        {"canonicalTitleEn": "Republic", "attributedAuthor": "plato",
         "authorConfidence": "attributed", "tradition": "platonist",
         "doNotAttributeTo": ["socrates"]},
    ]


def born_gated_documents(path: "Path | None" = None) -> "list[str]":
    """Provenance records → list of inline-marked document strings."""
    recs = load_records(path) or _synthetic_records()
    lines = [ln for ln in (_record_line(r) for r in recs) if ln]
    return lines


def born_gated_token_stream(tokenizer=None, *, path: "Path | None" = None) -> "list[int]":
    """Encode the born-gated corpus with special-token parsing on."""
    from pretraining.gpt.tokenizer import ByteProvenanceTokenizer

    tok = tokenizer or ByteProvenanceTokenizer()
    ids: list[int] = []
    for doc in born_gated_documents(path):
        ids.extend(tok.encode_with_specials(doc))
        ids.append(tok.eot_id)
    return ids


__all__ = ["born_gated_documents", "born_gated_token_stream", "load_records"]
