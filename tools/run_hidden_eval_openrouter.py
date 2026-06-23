#!/usr/bin/env python3
"""Generate hidden-eval response JSON with OpenRouter.

This tool is a transport adapter only. It does NOT judge the hidden comparison;
score with tools/run_hidden_eval_full.py afterward.

Example:
  OPENROUTER_API_KEY=... python tools/run_hidden_eval_openrouter.py \
    --pack private/hidden-evals/level3/PACK.json \
    --mode sophia_full \
    --model anthropic/claude-3.5-sonnet \
    --out private/hidden-evals/level3/responses.sophia_full.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from agent.openrouter_client import chat_completion, extract_text  # noqa: E402
from hidden_eval_protocol import load_json, validate_pack  # noqa: E402

SYSTEMS = {
    "raw": "Answer directly. Do not use Sophia-specific gates or internal wiki. Be concise.",
    "raw_tools": "Answer with general reasoning and any source-like knowledge you have. Be concise.",
    "rag_only": "Answer as if using retrieval context only. Cite uncertainty when evidence is missing.",
    "gate_only": "Apply a strict fail-closed source discipline gate. If unsupported, say HELD/ABSTAIN.",
    "sophia_full": (
        "You are Sophia-full: decompose claims, verify sources, refuse unsupported specifics, "
        "preserve provenance, and include a short 中文摘要. Never assert an unsupported attribution."
    ),
}


def build_prompt(case: dict, mode: str) -> list[dict[str, str]]:
    materials = case.get("materials") or []
    mat = "\n".join(json.dumps(m, ensure_ascii=False) if not isinstance(m, str) else m for m in materials)
    user = (
        f"Hidden eval case id: {case['id']}\n"
        f"Domain: {case.get('domain')}\n"
        f"Prompt:\n{case['prompt']}\n\n"
        f"Materials:\n{mat or '(none)'}\n\n"
        "Return only the answer text."
    )
    return [{"role": "system", "content": SYSTEMS.get(mode, SYSTEMS["raw"])}, {"role": "user", "content": user}]


def run(pack: dict, *, mode: str, model: str, api_key_file: Path | None, timeout_sec: int, limit: int | None) -> dict:
    responses: dict[str, str] = {}
    logs: dict[str, dict] = {}
    cases = pack["cases"][: limit or None]
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} via OpenRouter {model}", file=sys.stderr, flush=True)
        resp = chat_completion(model=model, messages=build_prompt(case, mode), api_key_file=api_key_file, timeout_sec=timeout_sec)
        responses[case["id"]] = extract_text(resp)
        logs[case["id"]] = {"backend": "openrouter", "model": model, "mode": mode, "responseKeys": sorted(resp.keys())}
    return {"packId": pack["packId"], "model": model, "mode": mode, "responses": responses, "logs": logs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--mode", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--model", required=True, help="OpenRouter model id, e.g. anthropic/claude-3.5-sonnet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None, help="debug: only run first N cases")
    args = ap.parse_args(argv)
    pack = load_json(args.pack)
    errors = validate_pack(pack)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2), file=sys.stderr)
        return 2
    payload = run(pack, mode=args.mode, model=args.model, api_key_file=args.api_key_file,
                  timeout_sec=args.timeout_sec, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
