# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Lightweight RAG over Sophia corpus data, docs, and training examples."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import DATA_DIR, DOCS_DIR, ROOT, TRAINING_DIR, WIKI_DIR


@dataclass
class SourceChunk:
    path: str
    title: str
    excerpt: str
    score: float
    # Provenance carried from OKF wiki frontmatter (empty for non-wiki sources).
    page_id: "str | None" = None
    tradition: "str | None" = None
    author_confidence: "str | None" = None
    do_not_attribute_to: list = field(default_factory=list)


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z\u4e00-\u9fff]{3,}", text.lower()) if len(t) > 2}


def _score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    body = set(_tokenize(text))
    if not body:
        return 0.0
    return len(query_tokens & body) / len(query_tokens)


def _load_json_records(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[tuple[str, str]] = []
    for key, record in data.items():
        if isinstance(record, dict):
            chunks.append((key, json.dumps(record, ensure_ascii=False)))
        else:
            chunks.append((key, str(record)))
    return chunks


def _provenance_from_meta(meta: dict) -> "dict | None":
    """Extract retrieval-facing provenance from OKF frontmatter, or None."""
    if not meta:
        return None
    keys = ("tradition", "authorConfidence", "doNotAttributeTo")
    if not any(meta.get(k) for k in keys) and not meta.get("id"):
        return None
    return {
        "page_id": meta.get("id"),
        "tradition": meta.get("tradition"),
        "author_confidence": meta.get("authorConfidence"),
        "do_not_attribute_to": list(meta.get("doNotAttributeTo") or []),
    }


def _iter_markdown(path: Path, max_chars: int = 4000) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    from agent.chunking import chunk_text
    from okf import frontmatter

    # Strip OKF frontmatter so it is not indexed as body noise (disputes/wiki carry it).
    text = frontmatter.strip(path.read_text(encoding="utf-8"))
    title = path.stem.replace("-", " ")
    chunks = chunk_text(text, source_id=path.stem)
    if len(chunks) <= 1:
        return [(title, text[:max_chars] if not chunks else chunks[0].text)]
    return [(f"{title} [chunk {c.index}]", c.text) for c in chunks]


def _markdown_provenance(path: Path) -> "dict | None":
    if not path.exists():
        return None
    from okf import frontmatter

    meta, _ = frontmatter.parse(path.read_text(encoding="utf-8"))
    return _provenance_from_meta(meta)


def collect_corpus() -> list[tuple]:
    """Return (path_label, title, text, provenance|None) for all searchable sources.

    OKF wiki pages (and frontmatter'd disputes) carry a provenance dict so retrieval
    can rank by source confidence and surface doNotAttributeTo constraints inline.
    """
    items: list[tuple] = []

    for json_path in sorted(DATA_DIR.glob("*.json")):
        for key, text in _load_json_records(json_path):
            items.append((f"data/{json_path.name}", key, text, None))

    for example in sorted(TRAINING_DIR.glob("*.json")):
        payload = json.loads(example.read_text(encoding="utf-8"))
        assistant = next((m["content"] for m in payload.get("messages", []) if m.get("role") == "assistant"), "")
        user = next((m["content"] for m in payload.get("messages", []) if m.get("role") == "user"), "")
        items.append((f"training/{example.name}", example.stem, f"Q: {user}\nA: {assistant[:2000]}", None))

    doc_roots = [
        WIKI_DIR,  # OKF provenance wiki — first-class, provenance-stamped source
        DOCS_DIR / "08-Domains",
        DOCS_DIR / "07-Growth",
        DOCS_DIR / "06-Roadmap",
        DOCS_DIR / "04-Disputes",
        DOCS_DIR / "09-Agent",
        ROOT / "agi-proof",
        ROOT / "GOOD_FIRST_ISSUES.md",
        ROOT / "README.md",
    ]
    for root in doc_roots:
        if root.is_file():
            prov = _markdown_provenance(root)
            for title, text in _iter_markdown(root):
                items.append((str(root.relative_to(ROOT)), title, text, prov))
        elif root.is_dir():
            for md in sorted(root.rglob("*.md")):
                prov = _markdown_provenance(md)
                for title, text in _iter_markdown(md):
                    items.append((str(md.relative_to(ROOT)), title, text, prov))

    return items


# A retrieved page on its own tradition outranks a raw JSON dump of the same fact.
_CONFIDENCE_BOOST = {
    "consensus": 0.20, "attributed": 0.12, "compiled": 0.10, "layered": 0.08,
    "disputed": 0.04, "legendary": 0.02, "anachronism_risk": 0.0, "none_extant": 0.0,
}


def _retrieve_keyword(query: str, *, top_k: int = 8) -> list[SourceChunk]:
    query_tokens = _tokenize(query)
    ranked: list[SourceChunk] = []
    for item in collect_corpus():
        path_label, title, text = item[0], item[1], item[2]
        prov = item[3] if len(item) > 3 else None
        score = _score(query_tokens, f"{title} {text}")
        if score <= 0:
            continue
        if prov:  # provenance boost: curated, confident wiki pages win over raw dumps
            score += 0.05 + _CONFIDENCE_BOOST.get(prov.get("author_confidence"), 0.0)
        excerpt = text[:1200] + ("..." if len(text) > 1200 else "")
        ranked.append(SourceChunk(
            path=path_label, title=title, excerpt=excerpt, score=score,
            page_id=(prov or {}).get("page_id"),
            tradition=(prov or {}).get("tradition"),
            author_confidence=(prov or {}).get("author_confidence"),
            do_not_attribute_to=list((prov or {}).get("do_not_attribute_to") or []),
        ))
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked[:top_k]


def embed_query_for_index(query: str, idir, *, has_embeddings: bool = True):
    """Embed ``query`` with the SAME backend that built the index at ``idir``.

    Returns an embedding vector or ``None`` (keyword mode, no committed vectors, or an
    embedder error). Shared by :func:`retrieve` and the hybrid retriever so both embed the
    query in exactly one space — the committed local hashing backend is offline/CPU, so
    vector recall works under airgap with no API key.
    """
    from agent.config import load_dotenv
    from agent.vector_store import embedding_backend_id

    load_dotenv()
    backend = (os.environ.get("SOPHIA_RAG_BACKEND") or "auto").strip().lower()
    if backend == "keyword" or not has_embeddings:
        return None
    index_backend = embedding_backend_id(idir)
    if index_backend == "local-hash-v1":
        try:
            from agent.rag_local_embed import embed_query

            return embed_query(query)
        except Exception:
            return None
    if backend in {"gemini", "vertex", "auto"}:
        if backend == "vertex":
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        try:
            from agent.rag_embed import embed_query

            return embed_query(query)
        except Exception:
            return None
    return None


def retrieve(query: str, *, top_k: int = 8) -> list[SourceChunk]:
    """Retrieve sources — prefers curated `rag/index` when present."""
    try:
        from agent.vector_store import index_dir, load_index, search

        idir = index_dir()
        indexed = load_index(idir)
        if indexed:
            has_embeddings = indexed[0].embedding is not None
            query_embedding = embed_query_for_index(query, idir, has_embeddings=has_embeddings)
            return search(query, indexed, top_k=top_k, query_embedding=query_embedding)
    except Exception:
        pass
    return _retrieve_keyword(query, top_k=top_k)


def format_context(chunks: list[SourceChunk]) -> str:
    if not chunks:
        return "(No matching sources — answer from general reasoning and flag uncertainty.)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"### Source {i}: {chunk.path} / {chunk.title} (relevance {chunk.score:.2f})"
        confidence = getattr(chunk, "author_confidence", None)
        tradition = getattr(chunk, "tradition", None)
        dna = getattr(chunk, "do_not_attribute_to", None) or []
        prov_bits = []
        if confidence:
            prov_bits.append(f"confidence={confidence}")
        if tradition:
            prov_bits.append(f"tradition={tradition}")
        if prov_bits:
            header += " [" + ", ".join(prov_bits) + "]"
        body = chunk.excerpt
        if dna:
            # surface the source-discipline constraint at generation time
            body += f"\n⚠ Source discipline — do NOT attribute this to: {', '.join(dna)}."
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)
