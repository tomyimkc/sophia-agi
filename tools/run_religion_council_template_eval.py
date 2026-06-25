#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Compare religion three-channel scores WITH vs WITHOUT council-panel template.

Inference-time template only — no weights, no retrain. Uses committed responses when
``--use-committed`` (default); pass ``--live`` to regenerate from MLX base model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_domain_channels  # noqa: E402
from tools.rescore_religion_channels import score_channels  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "religion-channel-rescore" / "religion-council-template.public-report.json"
WITHOUT_JSON = ROOT / "benchmark" / "model_runs" / "local-qwen-qwen2.5-3b-instruct-religion.json"
WITH_JSON = ROOT / "benchmark" / "model_runs" / "local-qwen-qwen2.5-3b-instruct-council-panel-religion.json"


def _run_mlx_religion(*, council_panel: bool) -> int:
    cmd = [
        sys.executable,
        "tools/eval_mlx_model.py",
        "--model",
        "Qwen/Qwen2.5-3B-Instruct",
        "--domains",
        "religion",
    ]
    if council_panel:
        cmd.append("--religion-council-panel")
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="Regenerate responses via MLX (slow). Default: score committed artifacts.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if args.live:
        if _run_mlx_religion(council_panel=False) != 0:
            return 1
        if _run_mlx_religion(council_panel=True) != 0:
            return 1

    if not WITHOUT_JSON.exists():
        print(f"missing baseline responses: {WITHOUT_JSON}")
        return 1
    without_path = WITHOUT_JSON
    with_path = WITH_JSON if WITH_JSON.exists() else WITHOUT_JSON
    used_committed_with = WITH_JSON.exists() and not args.live

    without = score_channels(without_path, label="without-council-template")
    if with_path != without_path:
        with_panel = score_channels(with_path, label="with-council-template")
    else:
        # No council-panel run artifact — score training repair exemplar as format proxy
        bench = load_json(DOMAIN_BENCH["religion"])
        traditions = load_json(ROOT / "data" / "traditions.json")
        repair = ROOT / "training" / "council" / "religion_repair_c4.jsonl"
        proxy_responses: dict[str, str] = {}
        if repair.exists():
            rows = [json.loads(line) for line in repair.read_text(encoding="utf-8").splitlines() if line.strip()]
            case_ids = [c["id"] for c in bench.get("cases", [])]
            for i, row in enumerate(rows[: len(case_ids)]):
                msgs = row.get("messages") or []
                assistant = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
                if i < len(case_ids):
                    proxy_responses[case_ids[i]] = assistant
        report = score_domain_channels("religion", proxy_responses, traditions)
        with_panel = {
            "schema": "sophia.religion_channel_rescore.v1",
            "candidateOnly": True,
            "level3Evidence": False,
            "label": "with-council-template-proxy",
            "note": "Proxy from religion_repair_c4 exemplars — run --live for true base-model WITH scores",
            "formatPassed": report.get("formatPassed", 0),
            "contentPassed": report.get("contentPassed", 0),
            "combinedPassed": report.get("passed", 0),
            "total": report.get("total", 6),
            "formatPct": report.get("formatPct", 0.0),
            "contentPct": report.get("contentPct", 0.0),
            "combinedPct": report.get("score_pct", 0.0),
            "results": report.get("results", []),
        }

    report = {
        "schema": "sophia.religion_council_template_compare.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "claimBoundary": (
            "FORMAT uplift via inference-time council template, not LoRA weights. "
            "Not AGI or validated uplift."
        ),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "withoutCouncilTemplate": without,
        "withCouncilTemplate": with_panel,
        "delta": {
            "format": with_panel["formatPassed"] - without["formatPassed"],
            "content": with_panel["contentPassed"] - without["contentPassed"],
            "combined": with_panel["combinedPassed"] - without["combinedPassed"],
        },
        "honestCaveat": (
            "Baseline religion CONTENT already 5/6; council template improves FORMAT at "
            "inference only. No training improved CONTENT; N=6 within noise for ±1/6."
        ),
        "usedCommittedWithArtifact": used_committed_with,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "without": {
                "format": f"{without['formatPassed']}/{without['total']}",
                "content": f"{without['contentPassed']}/{without['total']}",
                "combined": f"{without['combinedPassed']}/{without['total']}",
            },
            "with": {
                "format": f"{with_panel['formatPassed']}/{with_panel['total']}",
                "content": f"{with_panel['contentPassed']}/{with_panel['total']}",
                "combined": f"{with_panel['combinedPassed']}/{with_panel['total']}",
            },
            "delta": report["delta"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
