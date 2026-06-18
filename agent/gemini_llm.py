"""Gemini generation for Sophia online RAG."""

from __future__ import annotations

from agent.google_genai_client import gemini_model, make_genai_client

SYSTEM = (
    "You are Sophia AGI — a provenance-aware instructor across philosophy, psychology, "
    "history, and religion. Use ONLY the retrieved sources below. Deny lineage-merge traps. "
    "Use council panel format for religion questions when appropriate. Label pop myths. "
    "End with a concise 中文 summary. Cite source paths you used."
)


def complete_with_context(question: str, context: str, *, max_output_tokens: int = 2048) -> str:
    client = make_genai_client()
    prompt = (
        f"{SYSTEM}\n\n"
        f"## Retrieved sources (curated Sophia corpus only)\n{context}\n\n"
        f"## User question\n{question}\n\n"
        "Respond with: Analysis → Recommendation → cited source paths → 中文摘要"
    )
    response = client.models.generate_content(
        model=gemini_model(),
        contents=prompt,
        config={"max_output_tokens": max_output_tokens, "temperature": 0.2},
    )
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    return str(response)