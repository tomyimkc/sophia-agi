#!/usr/bin/env python3
"""Run external LLMs on Sophia benchmark templates (optional API keys).

Environment variables:
  OPENAI_API_KEY   -> gpt-4o
  ANTHROPIC_API_KEY -> claude-sonnet
  XAI_API_KEY      -> grok (xAI API)

Usage:
  python tools/run_external_models.py --domain philosophy
  python tools/run_external_models.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "benchmark" / "templates"
OUT_DIR = ROOT / "benchmark" / "model_runs"
DOMAINS = ("philosophy", "psychology", "history", "religion")

SYSTEM = (
    "You are a Sophia AGI instructor. Use source discipline. "
    "For religion use council panel format with all seats named. "
    "English + canonical Chinese terms; end with 中文 summary."
)


def load_template(domain: str) -> dict:
    return json.loads((TEMPLATE_DIR / f"responses-{domain}.template.json").read_text(encoding="utf-8"))


def ask_openai(question: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return r.choices[0].message.content or ""


def ask_anthropic(question: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1200,
        system=SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(b.text for b in r.content if b.type == "text")


def ask_xai(question: str) -> str:
    import urllib.request

    import json as _json

    api_key = os.environ["XAI_API_KEY"]
    model = os.environ.get("XAI_MODEL", "grok-2-latest")
    body = _json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            "temperature": 0.2,
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = _json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def run_provider(name: str, domain: str, bench_cases: dict[str, str]) -> Path | None:
    providers = {
        "gpt-4o": ("OPENAI_API_KEY", ask_openai),
        "claude-sonnet": ("ANTHROPIC_API_KEY", ask_anthropic),
        "grok": ("XAI_API_KEY", ask_xai),
    }
    if name not in providers:
        return None
    env_key, fn = providers[name]
    if not os.environ.get(env_key):
        print(f"  skip {name}: {env_key} not set")
        return None

    bench_path = ROOT / "tests" / f"benchmark-{domain}.json"
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    responses = {}
    for case in bench["cases"]:
        q = case["question"]
        print(f"  {name} / {domain} / {case['id']}...")
        responses[case["id"]] = fn(q)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}-{domain}.json"
    out.write_text(
        json.dumps({"domain": domain, "model": name, "responses": responses}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = OUT_DIR / f"{name}-{domain}.report.json"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "score_benchmark.py"), str(out), "--domain", domain, "--out", str(report)],
        check=False,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--providers", nargs="*", default=["gpt-4o", "claude-sonnet", "grok"])
    args = parser.parse_args()

    domains = DOMAINS if args.all else (args.domain,)
    if not domains or (None in domains and not args.all):
        parser.error("Specify --domain or --all")

    for domain in domains:
        print(f"Domain: {domain}")
        for provider in args.providers:
            run_provider(provider, domain, {})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())