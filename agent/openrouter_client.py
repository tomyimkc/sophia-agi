"""Optional OpenRouter client for hidden comparison / judge backends.

No API key is stored in the repo. Use either:

  OPENROUTER_API_KEY=... python tools/run_hidden_eval_openrouter.py ...
  python tools/run_hidden_eval_openrouter.py --api-key-file private/secrets/openrouter_api_key ...

The client uses OpenRouter's OpenAI-compatible chat-completions endpoint.
Network is optional; CI uses mocks and never calls this module live.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_api_key(*, api_key: str | None = None, api_key_file: str | Path | None = None) -> str:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key and api_key_file:
        key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY or --api-key-file is required for live OpenRouter calls")
    return key.strip()


def chat_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    timeout_sec: int = 120,
    app_title: str = "Sophia AGI",
) -> dict[str, Any]:
    key = load_api_key(api_key=api_key, api_key_file=api_key_file)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tomyimkc.github.io/sophia-agi/",
            "X-Title": app_title,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310 - explicit user API backend
        data = json.loads(resp.read().decode("utf-8"))
    return data


def extract_text(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"OpenRouter response missing choices[0].message.content: {response}") from exc


def make_complete(*, model: str, api_key: str | None = None, api_key_file=None,
                  temperature: float = 0.0, max_tokens: int = 256, retries: int = 4):
    """Return a ``complete(system, user)`` callable for CPQA, with backoff retries.

    Lets OpenRouter serve as an independent judge/answer gateway behind the same
    interface as agent.deepseek_llm / agent.llmhub_llm.
    """
    import ssl
    import time
    import urllib.error

    def complete(system: str, user: str, *, max_tokens: int = max_tokens) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(retries):
            try:
                resp = chat_completion(model=model, messages=messages, api_key=api_key,
                                       api_key_file=api_key_file, temperature=temperature,
                                       max_tokens=max_tokens)
                return extract_text(resp)
            except (urllib.error.URLError, ssl.SSLError, TimeoutError, ConnectionError) as exc:
                code = getattr(exc, "code", None)
                if attempt == retries - 1 or (code is not None and code < 500 and code != 429):
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    return complete


__all__ = ["OPENROUTER_URL", "load_api_key", "chat_completion", "extract_text", "make_complete"]
