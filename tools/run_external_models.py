#!/usr/bin/env python3
"""Run external LLMs on Sophia benchmark templates.

Option A (.env file — set once):
  ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL  -> Claude (Anthropic-compatible proxy)
  ANTHROPIC_API_KEY or CLAUDE_API_KEY     -> Claude Sonnet (api.anthropic.com default)
  MONICA_API_KEY                       -> GPT + Claude + Grok via Monica gateway
  OPENAI_API_KEY / XAI_API_KEY         -> individual direct providers

Usage:
  python tools/run_external_models.py --all
  python tools/run_external_models.py --domain religion
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmark" / "model_runs"
DOMAINS = ("philosophy", "psychology", "history", "religion")

MONICA_BASE_URL = "https://openapi.monica.im/v1"

SYSTEM = (
    "You are a Sophia AGI instructor using philosophy-style source discipline across all domains. "
    "Rules: (1) Name recordIds and authors precisely — e.g. deny Sigmund Freud for cognitive dissonance; "
    "affirm Leon Festinger. (2) Tag psychology subfields explicitly: cognitive, clinical, pop_myth. "
    "(3) Religion answers use council panel format with all seats named; cite traditions "
    "(Christianity, Buddhism, Islam, Daoist) by name. (4) Label pop claims as myth or misconception. "
    "(5) Reject universal claims with 'not all'. (6) English + canonical Chinese terms; end with 中文 summary. "
    "Keep answers focused — avoid unrelated council panels on non-religion psychology questions."
)

# provider id -> (gateway model id, direct env var)
PROVIDERS: dict[str, tuple[str, str]] = {
    "gpt-4o": ("gpt-4o", "OPENAI_API_KEY"),
    "claude-sonnet": ("claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    "grok": ("x-ai/grok-3-beta", "XAI_API_KEY"),
}


def is_real_secret(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    lowered = value.lower()
    if "your" in lowered or lowered.endswith("_here") or value in {"...", "xxx"}:
        return False
    return True


def load_dotenv() -> None:
    """Load KEY=VALUE lines from repo-root .env if present."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if is_real_secret(value):
            os.environ.setdefault(key, value)


def normalize_api_keys() -> None:
    """CLAUDE_API_KEY is an alias for ANTHROPIC_API_KEY."""
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]


def monica_api_key() -> str | None:
    value = os.environ.get("MONICA_API_KEY", "").strip()
    return value if is_real_secret(value) else None


def anthropic_api_key() -> str | None:
    value = (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY") or "").strip()
    return value if is_real_secret(value) else None


def anthropic_base_url() -> str | None:
    value = (os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("CLAUDE_BASE_URL") or "").strip()
    return value.rstrip("/") if value else None


def direct_api_key(provider: str) -> str | None:
    _, env_name = PROVIDERS[provider]
    if env_name == "ANTHROPIC_API_KEY":
        return anthropic_api_key()
    value = os.environ.get(env_name, "").strip()
    return value if is_real_secret(value) else None


def model_override(provider: str, default: str) -> str:
    env_map = {
        "gpt-4o": "OPENAI_MODEL",
        "claude-sonnet": "ANTHROPIC_MODEL",
        "grok": "XAI_MODEL",
    }
    return os.environ.get(env_map.get(provider, ""), default)


def ask_openai_compatible(question: str, model: str, *, api_key: str, base_url: str | None = None) -> str:
    from openai import OpenAI

    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def ask_anthropic_native(question: str, model: str) -> str:
    import anthropic

    kwargs: dict = {"api_key": anthropic_api_key()}
    base = anthropic_base_url()
    if base:
        kwargs["base_url"] = base
    client = anthropic.Anthropic(**kwargs)
    response = client.messages.create(
        model=model,
        max_tokens=1200,
        system=SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def ask_xai_native(question: str, model: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            "temperature": 0.2,
        }
    ).encode()
    request = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ['XAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"xAI HTTP {exc.code}: {detail}") from exc
    return data["choices"][0]["message"]["content"]


def ask_provider(provider: str, question: str) -> str:
    gateway_model, _ = PROVIDERS[provider]
    model = model_override(provider, gateway_model)

    direct = direct_api_key(provider)
    if direct:
        if provider == "gpt-4o":
            base = os.environ.get("OPENAI_BASE_URL")
            return ask_openai_compatible(question, model, api_key=direct, base_url=base)
        if provider == "claude-sonnet":
            return ask_anthropic_native(question, model)
        return ask_xai_native(question, model)

    monica_key = monica_api_key()
    if monica_key:
        return ask_openai_compatible(
            question,
            model,
            api_key=monica_key,
            base_url=MONICA_BASE_URL,
        )

    _, env_name = PROVIDERS[provider]
    raise RuntimeError(f"Set {env_name} (or CLAUDE_API_KEY) or MONICA_API_KEY in .env")


def provider_available(provider: str) -> bool:
    return bool(direct_api_key(provider) or monica_api_key())


def run_label(provider: str) -> str:
    if direct_api_key(provider):
        if provider == "claude-sonnet" and anthropic_base_url():
            host = anthropic_base_url().replace("https://", "").replace("http://", "")
            return f"{provider} ({host})"
        return provider
    if monica_api_key():
        return f"{provider} (monica)"
    return provider


def detect_providers() -> list[str]:
    return [name for name in PROVIDERS if provider_available(name)]


def describe_backend() -> str:
    parts: list[str] = []
    if anthropic_api_key():
        base = anthropic_base_url()
        if base:
            parts.append(f"Claude via {base}")
        else:
            parts.append("Claude (api.anthropic.com)")
    if os.environ.get("OPENAI_API_KEY", "").strip():
        parts.append("GPT (direct OpenAI API)")
    if os.environ.get("XAI_API_KEY", "").strip():
        parts.append("Grok (direct xAI API)")
    if monica_api_key():
        parts.append("Monica gateway (fallback for providers without direct keys)")
    return " | ".join(parts) if parts else "no API keys found"


def run_provider(provider: str, domain: str) -> Path | None:
    if provider not in PROVIDERS:
        print(f"  skip unknown provider: {provider}")
        return None
    if not provider_available(provider):
        _, env_name = PROVIDERS[provider]
        alias = " or CLAUDE_API_KEY" if env_name == "ANTHROPIC_API_KEY" else ""
        print(f"  skip {provider}: set {env_name}{alias} or MONICA_API_KEY in .env")
        return None

    bench_path = ROOT / "tests" / f"benchmark-{domain}.json"
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    label = run_label(provider)
    responses: dict[str, str] = {}

    for case in bench["cases"]:
        question = case["question"]
        print(f"  {label} / {domain} / {case['id']}...")
        responses[case["id"]] = ask_provider(provider, question)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{provider}-{domain}.json"
    out.write_text(
        json.dumps({"domain": domain, "model": label, "responses": responses}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = OUT_DIR / f"{provider}-{domain}.report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "score_benchmark.py"),
            str(out),
            "--domain",
            domain,
            "--out",
            str(report),
        ],
        check=False,
    )
    return out


def main() -> int:
    load_dotenv()
    normalize_api_keys()

    parser = argparse.ArgumentParser(description="Run Sophia AGI benchmarks on external LLMs")
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Providers to run (default: auto-detect from .env keys)",
    )
    args = parser.parse_args()

    domains = DOMAINS if args.all else (args.domain,)
    if not domains or (None in domains and not args.all):
        parser.error("Specify --domain or --all")

    providers = args.providers if args.providers is not None else detect_providers()
    if not providers:
        print("No API keys in .env. Add one of:")
        print("  ANTHROPIC_API_KEY=sk-ant-...   (Claude direct — recommended)")
        print("  CLAUDE_API_KEY=sk-ant-...      (alias)")
        print("  MONICA_API_KEY=...             (GPT + Claude + Grok gateway)")
        return 1

    print(f"Backend: {describe_backend()}")
    print(f"Providers: {', '.join(providers)}")

    for domain in domains:
        print(f"Domain: {domain}")
        for provider in providers:
            if provider not in PROVIDERS:
                print(f"  skip unknown provider: {provider}")
                continue
            run_provider(provider, domain)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())