"""Unified model adapter for Sophia AGI.

One abstraction over every backend the repo touches, so the agent harness,
distillation, and eval layers depend on a single interface instead of the
fragmented access in agent/llm.py, agent/gemini_llm.py, and the call_model
helpers inside tools/run_hidden_eval_sophia.py.

Providers
---------
- ``anthropic``     : Claude via the anthropic SDK (lazy import).
- ``openai``        : ANY OpenAI-compatible /chat/completions endpoint — covers
                      GLM-5.2 (Zhipu), vLLM, SGLang, Ollama, llama.cpp server,
                      DeepSeek, and OpenAI itself. Uses urllib (no new deps).
- ``grok``          : the local grok CLI (subprocess), mirroring the hidden runner.
- ``openclaw``      : the local OpenClaw CLI gateway (subprocess) — unified text
                      inference (``openclaw infer model run --json``) routed across
                      OpenClaw's own provider/auth profiles. Pure inference (writes no
                      knowledge); offline-stubbable, degrades to ``ok=False`` when absent.
- ``mock``          : deterministic, offline — lets the whole stack be tested
                      without network or credentials.

Features: named presets, reasoning effort, streaming (openai/mock), native
tool-calling pass-through, retry with backoff, fallback chain, and cost/latency
tracking. No required third-party dependency at import time.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.config import (
    anthropic_api_key,
    anthropic_base_url,
    is_real_secret,
    load_dotenv,
    normalize_api_keys,
)

ROOT = Path(__file__).resolve().parents[1]

# Approximate USD per 1M tokens (input, output). Override via SOPHIA_MODEL_PRICES
# (JSON object keyed by model substring). Unknown models cost 0 and are flagged.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (0.8, 4.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "deepseek": (0.27, 1.1),
    "glm-5": (0.8, 3.0),
    "glm-4": (0.6, 2.2),
    "glm": (0.6, 2.2),
    "qwen": (0.0, 0.0),
    "llama": (0.0, 0.0),
    "mistral": (0.0, 0.0),
}

# provider preset -> partial config. "kind" is the transport family.
PRESETS: dict[str, dict[str, Any]] = {
    "anthropic": {"kind": "anthropic", "model": "claude-sonnet-4-6"},
    "openai": {"kind": "openai", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini"},
    "glm": {"kind": "openai", "base_url": "https://open.bigmodel.cn/api/paas/v4", "api_key_env": "ZHIPUAI_API_KEY", "model": "glm-4.6"},
    "deepseek": {"kind": "openai", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
    "ollama": {"kind": "openai", "base_url": "http://localhost:11434/v1", "api_key_env": "OLLAMA_API_KEY", "model": "llama3.1", "api_key_default": "ollama"},
    "vllm": {"kind": "openai", "base_url": "http://localhost:8000/v1", "api_key_env": "VLLM_API_KEY", "model": "local", "api_key_default": "EMPTY"},
    "sglang": {"kind": "openai", "base_url": "http://localhost:30000/v1", "api_key_env": "SGLANG_API_KEY", "model": "local", "api_key_default": "EMPTY"},
    "llamacpp": {"kind": "openai", "base_url": "http://localhost:8080/v1", "api_key_env": "LLAMACPP_API_KEY", "model": "local", "api_key_default": "sk-no-key"},
    "grok": {"kind": "grok", "model": "grok-cli"},
    "openclaw": {"kind": "openclaw", "model": "xai/grok-4.3"},
    "mock": {"kind": "mock", "model": "mock-1"},
}

TRANSIENT_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class ModelConfig:
    """Resolved configuration for one model endpoint."""

    kind: str  # anthropic | openai | grok | openclaw | mock
    model: str
    label: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    api_key_default: str | None = None  # for local servers that need a dummy key
    max_tokens: int = 2400
    temperature: float = 0.2
    reasoning_effort: str | None = None  # low | medium | high (provider-dependent)
    timeout_sec: int = 120

    def resolved_key(self) -> str | None:
        if self.kind == "anthropic":
            return anthropic_api_key()
        if self.api_key_env:
            value = (os.environ.get(self.api_key_env) or "").strip()
            if is_real_secret(value):
                return value
        return self.api_key_default


@dataclass
class Attempt:
    provider: str
    model: str
    ok: bool
    latency_sec: float
    error: str | None = None


@dataclass
class ModelResult:
    text: str
    provider: str
    model: str
    ok: bool = True
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_sec: float = 0.0
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    attempts: list[Attempt] = field(default_factory=list)
    raw: dict[str, Any] | None = None

    def to_log(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "ok": self.ok,
            "error": self.error,
            "promptTokens": self.prompt_tokens,
            "completionTokens": self.completion_tokens,
            "costUsd": round(self.cost_usd, 6),
            "latencySec": round(self.latency_sec, 3),
            "finishReason": self.finish_reason,
            "toolCalls": self.tool_calls,
            "fallbackUsed": self.fallback_used,
            "attempts": [a.__dict__ for a in self.attempts],
        }


def _prices() -> dict[str, tuple[float, float]]:
    override = os.environ.get("SOPHIA_MODEL_PRICES")
    if not override:
        return DEFAULT_PRICES
    try:
        data = json.loads(override)
        merged = dict(DEFAULT_PRICES)
        for key, value in data.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                merged[key] = (float(value[0]), float(value[1]))
        return merged
    except (json.JSONDecodeError, TypeError, ValueError):
        return DEFAULT_PRICES


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, bool]:
    """Return (usd, known). known=False when no price entry matched the model."""
    lowered = model.lower()
    for key, (price_in, price_out) in sorted(_prices().items(), key=lambda kv: -len(kv[0])):
        if key in lowered:
            return (prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out, True)
    return (0.0, False)


def resolve_config(spec: str | None = None) -> ModelConfig:
    """Build a ModelConfig from a spec ("provider", "provider:model", or preset)
    or from the environment when spec is None."""
    load_dotenv()
    normalize_api_keys()
    if spec is None:
        spec = os.environ.get("SOPHIA_MODEL_PROVIDER")
    if not spec:
        spec = _auto_provider()

    provider, _, model_override = spec.partition(":")
    provider = provider.strip().lower()
    preset = PRESETS.get(provider)
    if preset is None:
        raise ValueError(f"unknown model provider {provider!r}; valid: {', '.join(sorted(PRESETS))}")

    cfg = ModelConfig(
        kind=preset["kind"],
        model=model_override.strip() or os.environ.get("SOPHIA_MODEL") or preset["model"],
        label=provider,
        base_url=os.environ.get("SOPHIA_MODEL_BASE_URL") or preset.get("base_url"),
        api_key_env=preset.get("api_key_env"),
        api_key_default=preset.get("api_key_default"),
        max_tokens=int(os.environ.get("SOPHIA_MAX_TOKENS", "2400")),
        temperature=float(os.environ.get("SOPHIA_TEMPERATURE", "0.2")),
        reasoning_effort=os.environ.get("SOPHIA_REASONING_EFFORT") or None,
        timeout_sec=int(os.environ.get("SOPHIA_TIMEOUT_SEC", "120")),
    )
    if cfg.kind == "anthropic" and not cfg.base_url:
        cfg.base_url = anthropic_base_url()
    return cfg


def _auto_provider() -> str:
    if anthropic_api_key():
        return "anthropic"
    if os.environ.get("SOPHIA_MODEL_BASE_URL"):
        return "openai"
    for provider in ("glm", "deepseek", "openai"):
        cfg = PRESETS[provider]
        env = cfg.get("api_key_env")
        if env and is_real_secret((os.environ.get(env) or "").strip()):
            return provider
    if (Path.home() / ".grok" / "auth.json").exists():
        return "grok"
    return "mock"


# --------------------------------------------------------------------------- #
# Provider transports
# --------------------------------------------------------------------------- #


def _call_mock(system: str, user: str, cfg: ModelConfig, *, on_token: Callable[[str], None] | None, **_: Any) -> ModelResult:
    """Deterministic offline provider for tests and dry runs.

    Honors SOPHIA_MOCK_RESPONSE / a {"mockResponse": ...} hint embedded in the
    user prompt; otherwise echoes a structured, gate-friendly answer.
    """
    forced = os.environ.get("SOPHIA_MOCK_RESPONSE")
    if forced is not None:
        text = forced
    else:
        head = user.strip().splitlines()[0][:200] if user.strip() else "(empty)"
        text = (
            f"[mock:{cfg.model}] Analysis of: {head}\n"
            "Decision: proceed (mock). source discipline noted.\n"
            "中文摘要: 模拟回答。"
        )
    if on_token:
        for token in text.split(" "):
            on_token(token + " ")
    pt, ct = max(1, len(user) // 4), max(1, len(text) // 4)
    return ModelResult(text=text, provider="mock", model=cfg.model, prompt_tokens=pt, completion_tokens=ct, finish_reason="stop")


def _call_anthropic(system: str, user: str, cfg: ModelConfig, *, tools: list[dict] | None, **_: Any) -> ModelResult:
    key = cfg.resolved_key()
    if not key:
        return ModelResult(text="", provider="anthropic", model=cfg.model, ok=False, error="no ANTHROPIC_API_KEY/CLAUDE_API_KEY")
    import anthropic  # lazy

    kwargs: dict[str, Any] = {"api_key": key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    client = anthropic.Anthropic(**kwargs)
    create_kwargs: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if tools:
        create_kwargs["tools"] = tools
    response = client.messages.create(**create_kwargs)
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    tool_calls = [
        {"id": b.id, "name": b.name, "arguments": b.input}
        for b in response.content
        if getattr(b, "type", None) == "tool_use"
    ]
    usage = getattr(response, "usage", None)
    return ModelResult(
        text=text,
        provider="anthropic",
        model=cfg.model,
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        finish_reason=getattr(response, "stop_reason", None),
        tool_calls=tool_calls,
    )


def _call_openai_compatible(
    system: str,
    user: str,
    cfg: ModelConfig,
    *,
    tools: list[dict] | None,
    on_token: Callable[[str], None] | None,
    **_: Any,
) -> ModelResult:
    key = cfg.resolved_key()
    if not key:
        return ModelResult(text="", provider=cfg.label or "openai", model=cfg.model, ok=False, error=f"no API key (set {cfg.api_key_env})")
    base = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": bool(on_token),
    }
    if cfg.reasoning_effort:
        payload["reasoning_effort"] = cfg.reasoning_effort
    if tools:
        payload["tools"] = tools
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout_sec) as response:
            if on_token:
                return _consume_stream(response, cfg, on_token)
            raw = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            body = ""
        return ModelResult(
            text="", provider=cfg.label or "openai", model=cfg.model, ok=False,
            error=f"HTTP {exc.code}: {body}".strip(), finish_reason="error",
        )
    except (urllib.error.URLError, TimeoutError) as exc:
        return ModelResult(text="", provider=cfg.label or "openai", model=cfg.model, ok=False, error=repr(exc))
    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message", {})
    usage = raw.get("usage", {})
    tool_calls = [
        {"id": tc.get("id"), "name": tc.get("function", {}).get("name"), "arguments": tc.get("function", {}).get("arguments")}
        for tc in message.get("tool_calls", []) or []
    ]
    return ModelResult(
        text=message.get("content") or "",
        provider=cfg.label or "openai",
        model=raw.get("model", cfg.model),
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        finish_reason=choice.get("finish_reason"),
        tool_calls=tool_calls,
        raw=raw,
    )


def _consume_stream(response: Any, cfg: ModelConfig, on_token: Callable[[str], None]) -> ModelResult:
    parts: list[str] = []
    for line in response:
        line = line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        token = delta.get("content")
        if token:
            parts.append(token)
            on_token(token)
    text = "".join(parts)
    return ModelResult(
        text=text,
        provider=cfg.label or "openai",
        model=cfg.model,
        completion_tokens=max(1, len(text) // 4),
        finish_reason="stop",
    )


def _call_grok(system: str, user: str, cfg: ModelConfig, **_: Any) -> ModelResult:
    prompt = f"{system}\n\n{user}"
    run_cwd = (ROOT / "private" / "agent-grok-cwd").resolve()
    run_cwd.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=run_cwd, prefix="sophia-prompt-", suffix=".md", delete=False) as handle:
        handle.write(prompt)
        prompt_file = Path(handle.name)
    command = [
        "grok", "--prompt-file", str(prompt_file), "--cwd", str(run_cwd),
        "--output-format", "plain", "--max-turns", "8",
        "--no-memory", "--no-plan", "--no-subagents", "--disable-web-search", "--verbatim",
        "--system-prompt-override", "You are Sophia's answerer. Answer directly from the prompt and provided context.",
    ]
    try:
        proc = subprocess.run(command, cwd=run_cwd, text=True, capture_output=True, timeout=cfg.timeout_sec, check=False)
        ok = proc.returncode == 0
        import re

        text = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout or "").strip()
        return ModelResult(text=text, provider="grok", model=cfg.model, ok=ok, error=None if ok else (proc.stderr or "")[-500:], finish_reason="stop" if ok else "error")
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return ModelResult(text="", provider="grok", model=cfg.model, ok=False, error=repr(exc))
    finally:
        if prompt_file.exists():
            prompt_file.unlink()


def _call_openclaw(system: str, user: str, cfg: ModelConfig, **_: Any) -> ModelResult:
    """OpenClaw gateway via its local CLI: ``openclaw infer model run --json``.

    ``cfg.model`` is the OpenClaw route ``<provider>/<model>`` (e.g. ``xai/grok-4.3``);
    OpenClaw owns provider auth/fallback. Pure inference — writes no knowledge, so this
    transport never touches the provenance gate. Degrades to ``ok=False`` when the binary
    is absent so the fallback chain (``...,mock``) keeps the stack offline-testable.
    """
    prompt = f"{system}\n\n{user}"
    binary = os.environ.get("SOPHIA_OPENCLAW_BIN", "openclaw")
    command = [binary, "infer", "model", "run", "--model", cfg.model, "--prompt", prompt, "--json"]
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=cfg.timeout_sec, check=False)
        if proc.returncode != 0:
            return ModelResult(text="", provider="openclaw", model=cfg.model, ok=False, error=(proc.stderr or proc.stdout or "")[-500:], finish_reason="error")
        data = json.loads(proc.stdout or "{}")
        outputs = data.get("outputs") if isinstance(data, dict) else None
        text = ""
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            text = (outputs[0].get("text") or "").strip()
        ok = bool(isinstance(data, dict) and data.get("ok", True)) and bool(text)
        return ModelResult(
            text=text,
            provider="openclaw",
            model=cfg.model,
            ok=ok,
            error=None if ok else "openclaw returned no usable text",
            finish_reason="stop" if ok else "error",
            raw=data if isinstance(data, dict) else None,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, TypeError) as exc:
        return ModelResult(text="", provider="openclaw", model=cfg.model, ok=False, error=repr(exc), finish_reason="error")


_TRANSPORTS: dict[str, Callable[..., ModelResult]] = {
    "mock": _call_mock,
    "anthropic": _call_anthropic,
    "openai": _call_openai_compatible,
    "grok": _call_grok,
    "openclaw": _call_openclaw,
}


def _is_transient(error: str | None) -> bool:
    if not error:
        return False
    if any(code in error for code in ("429", "500", "502", "503", "504", "timed out", "timeout", "Connection")):
        return True
    return False


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def _egress_blocked_for(cfg: "ModelConfig") -> bool:
    """Under the airgap profile, refuse any model transport that leaves the host.
    Local providers (ollama / llama.cpp / vLLM on localhost) are still allowed."""
    from agent.dataflow.firewall import egress_blocked

    if not egress_blocked():
        return False
    url = getattr(cfg, "base_url", "") or ""
    return not any(host in url for host in _LOCAL_HOSTS)


# --------------------------------------------------------------------------- #
# Client with retry + fallback
# --------------------------------------------------------------------------- #


class ModelClient:
    """Generate text with retry, fallback chain, and cost/latency tracking."""

    def __init__(self, primary: ModelConfig, fallbacks: list[ModelConfig] | None = None, *, retries: int = 2, backoff_sec: float = 1.0):
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.retries = max(1, retries)
        self.backoff_sec = backoff_sec

    def generate(
        self,
        system: str,
        user: str,
        *,
        tools: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> ModelResult:
        attempts: list[Attempt] = []
        configs = [self.primary, *self.fallbacks]
        for index, cfg in enumerate(configs):
            transport = _TRANSPORTS[cfg.kind]
            if cfg.kind != "mock" and _egress_blocked_for(cfg):
                err = f"airgap profile blocks egress to model provider '{cfg.label or cfg.kind}'"
                attempts.append(Attempt(cfg.label or cfg.kind, cfg.model, False, 0.0, err))
                continue
            for attempt in range(self.retries):
                started = time.monotonic()
                try:
                    result = transport(system, user, cfg, tools=tools, on_token=on_token)
                except Exception as exc:  # transport-level failure
                    result = ModelResult(text="", provider=cfg.label or cfg.kind, model=cfg.model, ok=False, error=repr(exc))
                result.latency_sec = round(time.monotonic() - started, 3)
                attempts.append(Attempt(cfg.label or cfg.kind, cfg.model, result.ok, result.latency_sec, result.error))
                if result.ok:
                    cost, known = estimate_cost(result.model, result.prompt_tokens, result.completion_tokens)
                    result.cost_usd = cost
                    if not known:
                        result.raw = {**(result.raw or {}), "costNote": "no price entry for model; cost=0"}
                    result.fallback_used = index > 0
                    result.attempts = attempts
                    return result
                if attempt < self.retries - 1 and _is_transient(result.error):
                    sleep(self.backoff_sec * (2 ** attempt))
                else:
                    break  # move to next fallback config
        failed = ModelResult(
            text="",
            provider=self.primary.label or self.primary.kind,
            model=self.primary.model,
            ok=False,
            error=attempts[-1].error if attempts else "no attempts",
            attempts=attempts,
        )
        return failed


def default_client(spec: str | None = None) -> ModelClient:
    """Build a ModelClient from the environment.

    SOPHIA_MODEL_PROVIDER selects the primary; SOPHIA_MODEL_FALLBACKS is a
    comma-separated list of provider[:model] specs.
    """
    primary = resolve_config(spec)
    fallbacks: list[ModelConfig] = []
    raw = os.environ.get("SOPHIA_MODEL_FALLBACKS", "").strip()
    if raw:
        for token in raw.split(","):
            token = token.strip()
            if token:
                try:
                    fallbacks.append(resolve_config(token))
                except ValueError:
                    continue
    retries = int(os.environ.get("SOPHIA_MODEL_RETRIES", "2"))
    return ModelClient(primary, fallbacks, retries=retries)


def complete(system: str, user: str, *, max_tokens: int = 2400, spec: str | None = None) -> str:
    """Backward-compatible string API (drop-in for agent.llm.complete).

    Raises RuntimeError on failure to preserve the previous contract.
    """
    cfg = resolve_config(spec)
    cfg.max_tokens = max_tokens
    client = ModelClient(cfg, _env_fallbacks(), retries=int(os.environ.get("SOPHIA_MODEL_RETRIES", "2")))
    result = client.generate(system, user)
    if not result.ok:
        raise RuntimeError(f"model call failed ({result.provider}/{result.model}): {result.error}")
    return result.text


def _env_fallbacks() -> list[ModelConfig]:
    raw = os.environ.get("SOPHIA_MODEL_FALLBACKS", "").strip()
    out: list[ModelConfig] = []
    for token in raw.split(",") if raw else []:
        token = token.strip()
        if token:
            try:
                out.append(resolve_config(token))
            except ValueError:
                continue
    return out
