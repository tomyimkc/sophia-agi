"""Online RAG: curated retrieve → Gemini → epistemic gate."""

from __future__ import annotations

import os
from typing import Any

from agent.config import load_dotenv
from agent.gate import check_response
from agent.gemini_llm import complete_with_context
from agent.llm import complete as claude_complete
from agent.retrieval import format_context, retrieve


def retrieve_online(query: str, *, top_k: int = 8) -> list:
    return retrieve(query, top_k=top_k)


def _backend() -> str:
    load_dotenv()
    return (os.environ.get("SOPHIA_RAG_BACKEND") or "auto").strip().lower()


def generate_answer(question: str, context: str) -> str:
    backend = _backend()
    if backend == "vertex":
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    if backend in {"claude", "anthropic"}:
        return claude_complete(
            "You are Sophia AGI with source discipline. Use retrieved sources only.",
            f"## Sources\n{context}\n\n## Question\n{question}",
        )
    if backend in {"keyword", "local"}:
        return claude_complete(
            "You are Sophia AGI with source discipline. Use retrieved sources only.",
            f"## Sources\n{context}\n\n## Question\n{question}",
        )
    try:
        return complete_with_context(question, context)
    except Exception:
        return claude_complete(
            "You are Sophia AGI with source discipline. Use retrieved sources only.",
            f"## Sources\n{context}\n\n## Question\n{question}",
        )


def answer_question(question: str, *, mode: str = "advisor", top_k: int = 8) -> dict[str, Any]:
    chunks = retrieve_online(question, top_k=top_k)
    context = format_context(chunks)
    answer = generate_answer(question, context)
    gate = check_response(
        answer,
        mode=mode,
        question=question,
        sources=[c.path for c in chunks],
        strict_attribution=True,
    )
    return {
        "question": question,
        "answer": answer,
        "sources": [{"path": c.path, "title": c.title, "score": c.score} for c in chunks],
        "gate": gate,
    }