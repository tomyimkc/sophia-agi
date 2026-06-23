#!/usr/bin/env python3
"""Run Sophia's offline AGI-missing-pillars mechanism checks.

These artifacts are candidate infrastructure only: they demonstrate mechanisms
for program induction, active verification agenda generation, MCTS planning,
predictive world modelling, safe plasticity, and layered memory. They do not prove
AGI and are marked accordingly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.active_inference import build_active_agenda  # noqa: E402
from agent.continual_plasticity import demo_plasticity_report  # noqa: E402
from agent.layered_memory import demo_memory_report  # noqa: E402
from agent.planner_mcts import run_mcts  # noqa: E402
from agent.predictive_world_model import demo_world_model_report  # noqa: E402
from agent.program_induction import evaluate_program_induction  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "agi-kernel" / "missing-pillars.public-report.json"
FACT_REPORT = ROOT / "agi-proof" / "fact-check-live" / "fact-check-live-eval.public-report.json"


def _load_fact_report() -> dict:
    if FACT_REPORT.exists():
        return json.loads(FACT_REPORT.read_text(encoding="utf-8"))
    return {"cases": [
        {"id": "demo-held", "claim": "US inflation increased in 2021", "label": "true", "verdict": "held", "confidence": 0.35, "type": "econ_empirical", "risk": "high", "reason": "insufficient independent sources"},
        {"id": "demo-low", "claim": "Jane Austen wrote Pride and Prejudice", "label": "true", "verdict": "accepted", "confidence": 0.76, "type": "open_empirical", "risk": "normal", "reason": "lexical support"},
    ]}


def build_report(seed: int = 0) -> dict:
    program = evaluate_program_induction(seed=seed)
    active = build_active_agenda(_load_fact_report(), limit=12)
    mcts = run_mcts("US inflation increased in 2021", iterations=120, seed=seed)
    world = demo_world_model_report()
    plasticity = demo_plasticity_report()
    memory = demo_memory_report()
    invariants = {
        "program_induction_ok": bool(program.get("ok")),
        "active_agenda_has_actions": bool(active.get("invariants", {}).get("all_gaps_have_actions")),
        "mcts_returns_plan": bool(mcts.get("plan")),
        "world_model_ok": all(world.get("invariants", {}).values()),
        "plasticity_gate_ok": all(plasticity.get("invariants", {}).values()),
        "layered_memory_ok": all(memory.get("invariants", {}).values()),
    }
    return {
        "schema": "sophia.agi_missing_pillars_bundle.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "claimBoundary": "Mechanism bundle for AGI-candidate infrastructure; does not prove AGI.",
        "seed": seed,
        "components": {
            "programInduction": program,
            "activeInference": active,
            "mctsPlanning": mcts,
            "predictiveWorldModel": world,
            "continualPlasticity": plasticity,
            "layeredMemory": memory,
        },
        "invariants": invariants,
        "ok": all(invariants.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    report = build_report(seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "out": str(out), "invariants": report["invariants"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
