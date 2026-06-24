"""Optional DeepSeek client (OpenAI-compatible) for live CPQA control-flow runs.

No API key is stored in the repo. Provide it at runtime:

    DEEPSEEK_API_KEY=... python tools/run_continual_qa_llm.py
    python tools/run_continual_qa_llm.py --api-key-file private/secrets/deepseek_api_key

Uses DeepSeek's OpenAI-compatible chat-completions endpoint over stdlib urllib (which
honors the environment proxy + CA bundle). Network is optional; CI uses mocks and never
calls this module live.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


def load_api_key(*, api_key: str | None = None, api_key_file: str | Path | None = None) -> str:
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key and api_key_file:
        key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY or --api-key-file is required for live DeepSeek calls")
    return key.strip()


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if cafile and Path(cafile).exists():
        ctx.load_verify_locations(cafile)
    return ctx


def chat_completion(*, messages, model: str = DEFAULT_MODEL, api_key: str | None = None,
                    api_key_file=None, temperature: float = 0.0, max_tokens: int = 32,
                    timeout_sec: int = 120, retries: int = 4) -> str:
    key = load_api_key(api_key=api_key, api_key_file=api_key_file)
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(DEEPSEEK_URL, data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    # Retry transient network/5xx errors with exponential backoff.
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec, context=_ssl_context()) as resp:
                return json.load(resp)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, ssl.SSLError, TimeoutError, ConnectionError) as exc:
            code = getattr(exc, "code", None)
            if attempt == retries - 1 or (code is not None and code < 500 and code != 429):
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def make_complete(*, model: str = DEFAULT_MODEL, api_key: str | None = None, api_key_file=None,
                  temperature: float = 0.0, max_tokens: int = 32):
    """Return a ``complete(system, user)`` callable for LLMController."""

    def complete(system: str, user: str, *, max_tokens: int = max_tokens) -> str:
        return chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model, api_key=api_key, api_key_file=api_key_file,
            temperature=temperature, max_tokens=max_tokens,
        )

    return complete


__all__ = ["chat_completion", "make_complete", "load_api_key", "DEFAULT_MODEL"]
