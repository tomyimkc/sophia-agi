#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Re-score religion benchmark responses with FORMAT vs CONTENT channels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import (  # noqa: E402
    DOMAIN_BENCH,
    load_json,
    score_case_channels,
)
from tools.score_benchmark import load_responses  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "religion-channel-rescore" / "religion-channels.public-report.json"


def score_channels(responses_path: Path, *, label: str) -> dict:
    bench = load_json(DOMAIN_BENCH["religion"])
    traditions = load_json(ROOT / "data" / "traditions.json")
    payload = load_json(responses_path)
    responses = load_responses(payload)

    results = []
    fmt_pass = content_pass = combined_pass = 0
    for case in bench.get("cases", []):
        cid = case["id"]
        ch = score_case_channels(case, responses.get(cid, ""), traditions)
        if ch["formatPassed"]:
            fmt_pass += 1
        if ch["contentPassed"]:
            content_pass += 1
        if ch["passed"]:
            combined_pass += 1
        results.append({"id": cid, **ch})

    total = len(bench.get("cases", [])) or 1
    return {
        "schema": "sophia.religion_channel_rescore.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "label": label,
        "responsesPath": str(responses_path),
        "formatPassed": fmt_pass,
        "contentPassed": content_pass,
        "combinedPassed": combined_pass,
        "total": total,
        "formatPct": round(100.0 * fmt_pass / total, 1),
        "contentPct": round(100.0 * content_pass / total, 1),
        "combinedPct": round(100.0 * combined_pass / total, 1),
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Religion FORMAT vs CONTENT channel rescore")
    ap.add_argument(
        "--baseline",
        default=str(ROOT / "benchmark" / "model_runs" / "local-qwen-qwen2.5-3b-instruct-religion.json"),
    )
    ap.add_argument(
        "--adapter",
        default=str(ROOT / "benchmark" / "model_runs" / "local-sophia-v4-religion-repair-religion.json"),
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    baseline = score_channels(Path(args.baseline), label="baseline-qwen2.5-3b")
    adapter = score_channels(Path(args.adapter), label="sophia-v4-religion-repair")

    report = {
        "schema": "sophia.religion_channel_rescore_bundle.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "claimBoundary": (
            "FORMAT (council-panel) and CONTENT (substance) scored separately. "
            "Not an AGI or validated uplift claim."
        ),
        "baseline": baseline,
        "adapter": adapter,
        "delta": {
            "format": adapter["formatPassed"] - baseline["formatPassed"],
            "content": adapter["contentPassed"] - baseline["contentPassed"],
            "combined": adapter["combinedPassed"] - baseline["combinedPassed"],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
