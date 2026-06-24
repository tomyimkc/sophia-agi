#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run all phase benchmarks recommended for Sophia's next evidence layer.

Phases:
1. SEIB-100 (epistemic integrity + skill/MCP/gate/full ablations)
2. Counterfactual Retraction + Belief Graph 50
3. AgentBench-Sophia 30
4. GPQA-Provenance smoke
5. Code-provenance adaptation 30
6. SEIB-Arena-20 smoke

All reports are candidate-only unless a later real-model/multi-judge run clears
the no-overclaim gate. This script is CI-safe and deterministic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.public_sanitize import sanitize_public_artifact  # noqa: E402
from tools.run_agentbench_sophia import run as run_agentbench  # noqa: E402
from tools.run_belief_revision_benchmark import run as run_belief_revision  # noqa: E402
from tools.run_code_provenance import run as run_code_provenance  # noqa: E402
from tools.run_epistemic_arena import run as run_arena  # noqa: E402
from tools.run_gpqa_provenance import run as run_gpqa  # noqa: E402
from tools.run_seib import run as run_seib  # noqa: E402

OUT = ROOT / "agi-proof" / "benchmark-results" / "all-phase-benchmarks.public-report.json"


def _summary(report: dict) -> dict:
    return {
        "schema": report.get("schema"),
        "benchmark": report.get("benchmark"),
        "candidateOnly": report.get("candidateOnly"),
        "level3Evidence": report.get("level3Evidence"),
        "validated": report.get("validated"),
        "ok": report.get("ok"),
        "metrics": report.get("metrics") or report.get("deltas") or {},
        "n": report.get("n") or report.get("nCases"),
        "claimBoundary": report.get("claimBoundary"),
    }


def run(out: str | Path = OUT) -> dict:
    components = {
        "seib100": run_seib(),
        "beliefRevision50": run_belief_revision(),
        "agentbenchSophia30": run_agentbench(),
        "gpqaProvenanceSmoke": run_gpqa(),
        "codeProvenance30": run_code_provenance(),
        "seibArena20": run_arena(),
    }
    summaries = {k: _summary(v) for k, v in components.items()}
    invariants = {
        "candidate_boundary": all(v.get("candidateOnly") is True and v.get("level3Evidence") is False for v in summaries.values()),
        "all_components_ok": all(v.get("ok") is True for v in summaries.values()),
        "no_component_validated_as_headline": all(v.get("validated") is False for v in summaries.values()),
    }
    report = {
        "schema": "sophia.all_phase_benchmarks.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "validated": False,
        "claimBoundary": "All-phase benchmark suite is candidate evidence infrastructure. It is not an AGI claim; headline results require real models, >=3 runs, >=2 independent judge families when semantic judging is used, kappa>=0.40, CIs, and explicit false-positive costs.",
        "components": summaries,
        "invariants": invariants,
        "ok": all(invariants.values()),
    }
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = run()
    print(json.dumps({"ok": report["ok"], "out": str(OUT), "invariants": report["invariants"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
