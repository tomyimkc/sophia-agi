"""Lightweight RAG over Sophia corpus data, docs, and training examples."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from agent.config import DATA_DIR, DOCS_DIR, ROOT, TRAINING_DIR


@dataclass
class SourceChunk:
    path: str
    title: str
    excerpt: str
    score: float


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


def _iter_markdown(path: Path, max_chars: int = 4000) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")[:max_chars]
    title = path.stem.replace("-", " ")
    return [(title, text)]


def collect_corpus() -> list[tuple[str, str, str]]:
    """Return (path_label, title, text) for all searchable sources."""
    items: list[tuple[str, str, str]] = []

    for json_path in sorted(DATA_DIR.glob("*.json")):
        for key, text in _load_json_records(json_path):
            items.append((f"data/{json_path.name}", key, text))

    for example in sorted(TRAINING_DIR.glob("*.json")):
        payload = json.loads(example.read_text(encoding="utf-8"))
        assistant = next((m["content"] for m in payload.get("messages", []) if m.get("role") == "assistant"), "")
        user = next((m["content"] for m in payload.get("messages", []) if m.get("role") == "user"), "")
        items.append((f"training/{example.name}", example.stem, f"Q: {user}\nA: {assistant[:2000]}"))

    doc_roots = [
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
            for title, text in _iter_markdown(root):
                items.append((str(root.relative_to(ROOT)), title, text))
        elif root.is_dir():
            for md in sorted(root.rglob("*.md")):
                for title, text in _iter_markdown(md):
                    items.append((str(md.relative_to(ROOT)), title, text))

    return items


def _retrieve_keyword(query: str, *, top_k: int = 8) -> list[SourceChunk]:
    query_tokens = _tokenize(query)
    ranked: list[SourceChunk] = []
    for path_label, title, text in collect_corpus():
        score = _score(query_tokens, f"{title} {text}")
        if score <= 0:
            continue
        excerpt = text[:1200] + ("..." if len(text) > 1200 else "")
        ranked.append(SourceChunk(path=path_label, title=title, excerpt=excerpt, score=score))
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked[:top_k]


def retrieve(query: str, *, top_k: int = 8) -> list[SourceChunk]:
    """Retrieve sources — prefers curated `rag/index` when present."""
    try:
        from agent.vector_store import index_dir, load_index, search

        indexed = load_index(index_dir())
        if indexed:
            from agent.config import load_dotenv

            load_dotenv()
            backend = (os.environ.get("SOPHIA_RAG_BACKEND") or "auto").strip().lower()
            if backend == "vertex":
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            query_embedding = None
            use_vectors = backend in {"gemini", "vertex", "auto"} and indexed[0].embedding is not None
            if use_vectors:
                try:
                    from agent.rag_embed import embed_query

                    query_embedding = embed_query(query)
                except Exception:
                    use_vectors = False
            return search(
                query,
                indexed,
                top_k=top_k,
                query_embedding=query_embedding if use_vectors else None,
            )
    except Exception:
        pass
    return _retrieve_keyword(query, top_k=top_k)


def format_context(chunks: list[SourceChunk]) -> str:
    if not chunks:
        return "(No matching sources — answer from general reasoning and flag uncertainty.)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"### Source {i}: {chunk.path} / {chunk.title} (relevance {chunk.score:.2f})\n{chunk.excerpt}")
    return "\n\n".join(parts)
