# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tokenizer analysis (idea #6): is the byte-level codec fair to EN and 中文?

A byte-level tokenizer spends 1 byte per ASCII char but ~3 bytes per CJK char
(UTF-8). That asymmetry is a real, measurable property worth reporting honestly
before scaling the corpus — it tells you how much of the context window each
language consumes and whether lineage terms (老子 / 孔子) stay distinct.

Dependency-free and deterministic; feeds the 配比 / data-mixing discussion in
``pretraining/data_mixing`` and the brainstorm doc.

    python -m pretraining.gpt.tokenizer_analysis
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pretraining.gpt.data import corpus_documents
from pretraining.gpt.tokenizer import ByteProvenanceTokenizer

HERE = Path(__file__).resolve().parent

# Lineage terms that MUST stay distinct — a tokenizer that maps them to the same
# id prefix would blur 儒家 / 道家 at the input layer.
_LINEAGE_TERMS = ["老子", "孔子", "Laozi", "Confucius", "Plato", "Socrates", "Freud", "Festinger"]


def bytes_per_char(text: str) -> float:
    if not text:
        return 0.0
    return len(text.encode("utf-8")) / len(text)


def language_efficiency(tokenizer: "ByteProvenanceTokenizer | None" = None) -> dict:
    """Tokens (bytes) per visible character, split by script, over the corpus."""
    tok = tokenizer or ByteProvenanceTokenizer()
    ascii_chars = cjk_chars = 0
    ascii_bytes = cjk_bytes = 0
    for doc in corpus_documents() or []:
        for ch in doc:
            b = len(ch.encode("utf-8"))
            if ord(ch) < 128:
                ascii_chars += 1
                ascii_bytes += b
            elif "一" <= ch <= "鿿":  # CJK unified ideographs
                cjk_chars += 1
                cjk_bytes += b
    return {
        "ascii_chars": ascii_chars,
        "cjk_chars": cjk_chars,
        "ascii_tokens_per_char": round(ascii_bytes / ascii_chars, 3) if ascii_chars else 0.0,
        "cjk_tokens_per_char": round(cjk_bytes / cjk_chars, 3) if cjk_chars else 0.0,
        "cjk_token_tax": round((cjk_bytes / cjk_chars) / (ascii_bytes / ascii_chars), 2)
        if ascii_chars and cjk_chars else None,
    }


def lineage_separation(tokenizer: "ByteProvenanceTokenizer | None" = None) -> dict:
    """Confirm distinct lineage terms get distinct token sequences (no collisions)."""
    tok = tokenizer or ByteProvenanceTokenizer()
    enc = {term: tok.encode(term) for term in _LINEAGE_TERMS}
    seqs = list(enc.values())
    collisions = sum(
        1 for i in range(len(seqs)) for j in range(i + 1, len(seqs)) if seqs[i] == seqs[j]
    )
    return {
        "terms": len(_LINEAGE_TERMS),
        "distinct_encodings": len({tuple(s) for s in seqs}),
        "collisions": collisions,
        "all_distinct": collisions == 0,
    }


def report(tokenizer: "ByteProvenanceTokenizer | None" = None) -> dict:
    tok = tokenizer or ByteProvenanceTokenizer()
    return {
        "canClaimAGI": False,
        "boundary": "tokenizer property report — descriptive stats, not a quality claim",
        "vocab_size": tok.vocab_size,
        "reserved_specials": len(tok.specials),
        "language_efficiency": language_efficiency(tok),
        "lineage_separation": lineage_separation(tok),
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Tokenizer EN/中文 fairness report.")
    ap.add_argument("--report", action="store_true", help="write tokenizer-analysis-latest.json")
    args = ap.parse_args(argv)

    rep = report()
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.report:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (HERE / "tokenizer-analysis-latest.json").write_text(
            json.dumps({**rep, "generatedAt": stamp}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"[tokenizer_analysis] wrote {HERE / 'tokenizer-analysis-latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
