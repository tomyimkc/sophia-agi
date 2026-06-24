#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""sophia-guard — run ANY local model behind Sophia's source-discipline gate.

A thin CLI over agent.guarded.guarded_complete: it retrieves context, generates an
answer with the configured model, and enforces the provenance gate — repairing once,
abstaining with citations, hedging, or passing through, per --on-fail. Use it to put
a small/local model (via the unified model adapter: ollama, llama.cpp, grok,
openclaw, …) under the same "don't merge lineages" discipline the platform applies.

    echo "Who wrote the Dao De Jing?" | python tools/sophia_guard.py
    python tools/sophia_guard.py --query "..." --provider ollama:llama3.2 --on-fail abstain --json

Exit code: 0 when an answer is surfaced (clean / repaired / abstained / hedged),
1 on a model error or a passthrough that still violates — so a caller can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.guarded import guarded_complete  # noqa: E402
from agent.retrieval import format_context, retrieve  # noqa: E402


def _make_generate(spec: "str | None") -> Callable:
    """Build a (system, user) -> ModelResult generator from a provider spec."""
    from agent.model import ModelClient, resolve_config

    client = ModelClient(resolve_config(spec))
    return lambda system, user: client.generate(system, user)


def run(
    query: str,
    *,
    on_fail: "str | None" = None,
    spec: "str | None" = None,
    top_k: int = 8,
    records: "dict | None" = None,
    policy: "str | None" = None,
    generate: "Callable | None" = None,
    retrieve_fn: Callable = retrieve,
    format_context_fn: Callable = format_context,
) -> dict:
    """Answer ``query`` behind the gate; return a JSON-serialisable result dict."""
    result = guarded_complete(
        query,
        on_fail=on_fail,
        generate=generate if generate is not None else _make_generate(spec),
        records=records,
        policy=policy,
        top_k=top_k,
        retrieve_fn=retrieve_fn,
        format_context_fn=format_context_fn,
    )
    return {
        "text": result.text,
        "ok": result.ok,
        "passed": result.passed,
        "action": result.action,
        "attempts": result.attempts,
        "violations": result.violations,
        "reasons": result.reasons,
        "contextUsed": result.context_used,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local model behind Sophia's source-discipline gate.")
    parser.add_argument("--query", help="question to answer (default: read stdin)")
    parser.add_argument("--on-fail", choices=["repair", "abstain", "hedge", "passthrough"],
                        help="what to do on a provenance violation (default: $SOPHIA_ON_FAIL or repair)")
    parser.add_argument("--provider", dest="spec", help="model spec, e.g. ollama:llama3.2 (default: environment)")
    parser.add_argument("--policy", choices=["provenance", "citation", "arithmetic", "code"],
                        help="machine-checked gate to enforce (default: provenance / $SOPHIA_POLICY)")
    parser.add_argument("--top-k", type=int, default=8, help="sources to retrieve")
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = parser.parse_args()

    query = args.query if args.query is not None else sys.stdin.read()
    query = (query or "").strip()
    if not query:
        parser.error("no query given (use --query or pipe text on stdin)")

    result = run(query, on_fail=args.on_fail, spec=args.spec, policy=args.policy, top_k=args.top_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["text"])
        if result["action"] != "clean":
            print(f"\n[sophia-guard: {result['action']}]", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
