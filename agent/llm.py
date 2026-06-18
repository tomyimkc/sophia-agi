"""LLM client for Sophia agent (Anthropic / LLMHub via .env)."""

from __future__ import annotations

import os

from agent.config import anthropic_api_key, anthropic_base_url, load_dotenv, normalize_api_keys


def complete(system: str, user: str, *, max_tokens: int = 2400) -> str:
    load_dotenv()
    normalize_api_keys()
    key = anthropic_api_key()
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY or CLAUDE_API_KEY in .env")

    import anthropic

    kwargs: dict = {"api_key": key}
    base = anthropic_base_url()
    if base:
        kwargs["base_url"] = base
    client = anthropic.Anthropic(**kwargs)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text")