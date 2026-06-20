#!/usr/bin/env python3
"""Turn agent run traces into SFT + DPO training data (the flywheel).

Reads append-only run logs from agent/memory/agent_runs/*.jsonl (written by the
harness), keeps only VERIFIED-good step outputs for SFT, builds (chosen, rejected)
preference pairs where a step has both a passing and a failing attempt, routes
failures to a rejected set, and de-leaks against the visible benchmark before
emitting. Implements the lora-dataset-creation skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json  # noqa: E402
from agent.config import ROOT as CFG_ROOT  # noqa: E402
from agent.prompts import MODE_PROMPTS  # noqa: E402

RUNS_DIR = ROOT / "agent" / "memory" / "agent_runs"


def _read_run(path: Path) -> dict[str, Any]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next((e for e in events if e.get("type") == "task_start"), {})
    outputs = [e for e in events if e.get("type") == "step_output"]
    return {"goal": start.get("goal", ""), "mode": start.get("mode", "advisor"), "outputs": outputs}


def _benchmark_questions() -> set[str]:
    questions: set[str] = set()
    for path in DOMAIN_BENCH.values():
        if path.exists():
            for case in load_json(path).get("cases", []):
                q = str(case.get("question", "")).strip().lower()
                if q:
                    questions.add(q)
    return questions


def collect(runs_dir: Path = RUNS_DIR, *, deleak: bool = True) -> dict[str, Any]:
    sft: list[dict] = []
    dpo: list[dict] = []
    rejected: list[dict] = []
    seen_sft: set[str] = set()
    holdout = _benchmark_questions() if deleak else set()
    leaked = 0

    for path in sorted(runs_dir.glob("*.jsonl")):
        run = _read_run(path)
        goal = run["goal"]
        if not goal:
            continue
        if deleak and goal.strip().lower() in holdout:
            leaked += 1
            continue
        system = MODE_PROMPTS.get(run["mode"], MODE_PROMPTS["advisor"])

        by_step: dict[str, dict[str, list[str]]] = {}
        for ev in run["outputs"]:
            bucket = by_step.setdefault(ev.get("step", "s?"), {"passed": [], "failed": []})
            text = ev.get("output", "")
            if not text.strip():
                continue
            (bucket["passed"] if ev.get("passed") else bucket["failed"]).append(text)

        for step_id, bucket in by_step.items():
            for good in bucket["passed"]:
                key = hashlib.sha256((goal + "␟" + good).encode()).hexdigest()
                if key in seen_sft:
                    continue
                seen_sft.add(key)
                sft.append({
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": goal},
                        {"role": "assistant", "content": good},
                    ],
                    "metadata": {"source": "agent-trace", "taskFile": path.name, "stepId": step_id},
                })
            for bad in bucket["failed"]:
                rejected.append({"prompt": goal, "rejected": bad, "metadata": {"taskFile": path.name, "stepId": step_id}})
            if bucket["passed"] and bucket["failed"]:
                dpo.append({
                    "prompt": goal,
                    "chosen": bucket["passed"][0],
                    "rejected": bucket["failed"][0],
                    "metadata": {"source": "agent-trace", "taskFile": path.name, "stepId": step_id},
                })

    return {"sft": sft, "dpo": dpo, "rejected": rejected, "leakedSkipped": leaked, "leakageChecked": deleak}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect SFT/DPO data from agent run traces")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training")
    parser.add_argument("--no-deleak", action="store_true", help="skip benchmark leakage check")
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(json.dumps({"error": f"no runs dir: {args.runs_dir}"}, indent=2))
        return 1
    data = collect(args.runs_dir, deleak=not args.no_deleak)
    _write_jsonl(args.out_dir / "agent_sft.jsonl", data["sft"])
    _write_jsonl(args.out_dir / "agent_dpo.jsonl", data["dpo"])
    _write_jsonl(args.out_dir / "agent_rejected.jsonl", data["rejected"])
    summary = {
        "sftRows": len(data["sft"]),
        "dpoPairs": len(data["dpo"]),
        "rejectedRows": len(data["rejected"]),
        "leakedSkipped": data["leakedSkipped"],
        "leakageChecked": data["leakageChecked"],
        "outDir": str(args.out_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
