"""Optional LLMHub client (OpenAI-compatible multi-provider gateway).

LLMHub exposes many model families (Anthropic, Google, OpenAI, Qwen, ...) behind one
OpenAI-compatible chat-completions endpoint, which lets CPQA assemble a *cross-provider*
judge panel from a single gateway. No API key is stored in the repo:

    LLMHUB_API_KEY=... python tools/run_continual_qa_judged.py --answer llmhub:gpt-5-mini ...
    python tools/run_continual_qa_judged.py --api-key-file private/secrets/llmhub_api_key ...

Network is optional; CI uses mocks and never calls this module live.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from pathlib import Path

# Base URL is configurable; LLMHub is OpenAI-compatible. HTTPS is required — the plain
# http:// host 301-redirects, and urllib downgrades POST->GET across the redirect.
BASE_URL = os.environ.get("LLMHUB_BASE_URL", "https://api.llmhub.com.cn") + "/v1/chat/completions"


def load_api_key(*, api_key: str | None = None, api_key_file: str | Path | None = None) -> str:
    key = api_key or os.environ.get("LLMHUB_API_KEY")
    if not key and api_key_file:
        key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("LLMHUB_API_KEY or --api-key-file is required for live LLMHub calls")
    return key.strip()


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if cafile and Path(cafile).exists():
        ctx.load_verify_locations(cafile)
    return ctx


def chat_completion(*, messages, model: str, api_key: str | None = None, api_key_file=None,
                    temperature: float = 0.0, max_tokens: int = 256, timeout_sec: int = 180) -> str:
    key = load_api_key(api_key=api_key, api_key_file=api_key_file)
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(BASE_URL, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_sec, context=_ssl_context()) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def make_complete(*, model: str, api_key: str | None = None, api_key_file=None,
                  temperature: float = 0.0, max_tokens: int = 256):
    def complete(system: str, user: str, *, max_tokens: int = max_tokens) -> str:
        return chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model, api_key=api_key, api_key_file=api_key_file,
            temperature=temperature, max_tokens=max_tokens,
        )

    return complete


__all__ = ["chat_completion", "make_complete", "load_api_key", "BASE_URL"]
