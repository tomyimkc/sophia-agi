#!/usr/bin/env python3
"""Verifier-gated distillation: teacher model -> verified SFT data + rejected set.

Runs a teacher (any adapter provider — GLM-5.2, DeepSeek, Claude) over a set of
prompts, gates each answer through the epistemic gate + required-keyword checks,
keeps only verified-good answers for SFT, and captures rejected answers + the full
trajectory. This is how you grow a smaller local student that behaves like a
specialized frontier agent — without training on teacher hallucinations.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import check_response  # noqa: E402
from agent.model import ModelClient, default_client  # noqa: E402
from agent.prompts import MODE_PROMPTS  # noqa: E402

DEFAULT_PROMPTS: list[dict[str, Any]] = [
    {"id": "ddj", "prompt": "Did Confucius write the Dao De Jing? Identify the correct tradition and author.", "mustInclude": ["Laozi"], "mustAvoid": ["Confucius wrote"]},
    {"id": "rome", "prompt": "When did the Western Roman Empire fall, and why is one date an oversimplification?", "mustInclude": ["476"]},
    {"id": "brain", "prompt": "Is the '10% of the brain' claim supported? Label any myth.", "mustInclude": ["myth"]},
]


def distill_one(item: dict[str, Any], client: ModelClient) -> dict[str, Any]:
    system = item.get("system") or MODE_PROMPTS["advisor"]
    prompt = item["prompt"]
    result = client.generate(system, prompt)
    base = {"id": item.get("id"), "prompt": prompt, "costUsd": result.cost_usd, "model": result.model}
    if not result.ok or not result.text.strip():
        return {**base, "accepted": False, "reasons": [result.error or "empty"], "answer": result.text}
    gate = check_response(result.text, mode="advisor", question=prompt)
    lowered = result.text.lower()
    missing = [k for k in item.get("mustInclude", []) if k.lower() not in lowered]
    forbidden = [k for k in item.get("mustAvoid", []) if k.lower() in lowered]
    accepted = gate.get("passed", False) and not missing and not forbidden
    reasons = list(gate.get("warnings", [])) + list(gate.get("violations", []))
    reasons += [f"missing:{k}" for k in missing] + [f"forbidden:{k}" for k in forbidden]
    return {**base, "accepted": accepted, "answer": result.text, "reasons": reasons}


def distill(prompts: list[dict[str, Any]], client: ModelClient) -> dict[str, Any]:
    sft: list[dict] = []
    rejected: list[dict] = []
    trajectory: list[dict] = []
    total_cost = 0.0
    for item in prompts:
        outcome = distill_one(item, client)
        total_cost += outcome.get("costUsd", 0.0) or 0.0
        trajectory.append({k: v for k, v in outcome.items() if k != "answer"} | {"accepted": outcome["accepted"]})
        if outcome["accepted"]:
            sft.append({
                "messages": [
                    {"role": "system", "content": MODE_PROMPTS["advisor"]},
                    {"role": "user", "content": outcome["prompt"]},
                    {"role": "assistant", "content": outcome["answer"]},
                ],
                "metadata": {"source": "distillation", "teacher": outcome["model"], "id": outcome["id"]},
            })
        else:
            rejected.append({"id": outcome["id"], "prompt": outcome["prompt"], "rejected": outcome["answer"], "reasons": outcome["reasons"]})
    return {
        "accepted": len(sft),
        "rejected": len(rejected),
        "total": len(prompts),
        "acceptRate": round(len(sft) / len(prompts), 3) if prompts else 0.0,
        "totalCostUsd": round(total_cost, 6),
        "sft": sft,
        "rejectedRows": rejected,
        "trajectory": trajectory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifier-gated distillation export")
    parser.add_argument("prompts", nargs="?", type=Path, default=None, help="prompts JSON [{id,prompt,mustInclude,mustAvoid}]")
    parser.add_argument("--provider", default=None, help="teacher provider (default from env; e.g. deepseek, glm:glm-5.2)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training")
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8")) if args.prompts else DEFAULT_PROMPTS
    client = default_client(args.provider)
    data = distill(prompts, client)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "distill_sft.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data["sft"]) + ("\n" if data["sft"] else ""), encoding="utf-8")
    (args.out_dir / "distill_rejected.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data["rejectedRows"]) + ("\n" if data["rejectedRows"] else ""), encoding="utf-8")
    summary = {k: data[k] for k in ("accepted", "rejected", "total", "acceptRate", "totalCostUsd")}
    summary["runAt"] = datetime.now().isoformat(timespec="seconds")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
