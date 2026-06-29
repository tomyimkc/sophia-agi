#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Adversarial eval: single-agent baseline vs team-agents orchestrator.

Uses the EXTERNAL scorer in ``provenance_bench/team_agents_benchmark.py`` (disjoint
from the intrinsic gate). Reports role-fidelity, hand-off integrity, calibrated
abstention, false-consensus, and panel independence (effective-N) at ≥3 seeds with
bootstrap 95% CI.

    python tools/eval_team_agents.py --mode mock --dry-run
    python tools/eval_team_agents.py --mode real --model mlx:Qwen/Qwen2.5-3B-Instruct \\
        --adapter training/mlx_adapters/sophia-v3 --backend mlx --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.council_deliberate import deliberate  # noqa: E402
from agent.steering import stats  # noqa: E402
from agent.team_agents import deliberate_team, measure_panel_independence  # noqa: E402
from provenance_bench.team_agents_benchmark import (  # noqa: E402
    HELDOUT,
    PROBE,
    load_cases,
    score_deliberation,
    verify_manifest,
)

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "team-agents.public-report.json"
SINGLE_SYSTEM = (
    "You are a careful source-disciplined Sophia advisor. "
    "Be concise; cite sources when grounded."
)
CONSENSUS_THRESHOLD_N_EFF = 2.0


def _resolve_model_spec(model: str, backend: str) -> str:
    if model.startswith(("mlx:", "hf:", "ollama:", "mock", "openrouter:", "anthropic:")):
        return model
    if backend == "mlx":
        return f"mlx:{model}"
    if backend == "hf":
        return f"hf:{model}"
    return model


def _make_client(
    *,
    mode: str,
    seed: int,
    model: str,
    adapter: str | None,
    backend: str,
):
    if mode == "mock":
        return _mock_client(seed)
    from agent.model import default_client

    spec = _resolve_model_spec(model, backend)
    if adapter:
        os.environ["SOPHIA_MLX_ADAPTER"] = adapter
    return default_client(spec)


def _load_jsonl_local(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mock_client(seed: int = 0):
    """Deterministic stub: seat-aware answers + conflict-flag synthesis."""
    n = [0]

    def generate(system: str, user: str):
        n[0] += 1
        low = system.lower()
        if "synthesis chair" in low:
            text = (
                "Flag conflict: Risk seat urges caution on runway while Growth seat "
                "favors deployment. Escalate to human — not consensus. Not advice."
            )
            return SimpleNamespace(ok=True, text=text)
        if any(k in low for k in ("tail", "risk", "compliance", "caution", "judge clerk")):
            return SimpleNamespace(ok=True, text="Caution: runway burn and AML/KYC compliance risk flagged.")
        if "corporate-finance" in low or "runway" in user.lower():
            return SimpleNamespace(ok=True, text="Runway is 20 months at current burn; monitor AML for Stripe.")
        if "value-investing" in low:
            return SimpleNamespace(ok=True, text="Margin of safety: LTV/CAC ratio 3x looks adequate with caveats.")
        if "labor" in low or "minimum wage" in user.lower():
            return SimpleNamespace(ok=True, text="Minimum wage rise yields HK$44/hour for HK$40 base (+10%).")
        if "trade" in low or "tariff" in user.lower():
            return SimpleNamespace(ok=True, text="Tariff incidence shared between importer and consumer.")
        return SimpleNamespace(ok=True, text=f"Balanced trade-off analysis (seed={seed}). Not advice.")

    return SimpleNamespace(generate=generate, spec=f"mock:seed{seed}")


def _single_agent(case: dict, client) -> object:
    ans = client.generate(SINGLE_SYSTEM, case["prompt"]).text
    from agent.council_deliberate import Deliberation

    return Deliberation(
        query=case["prompt"], councilId=case.get("councilId"), seats=[], guardians=[],
        synthesis=ans, gatedOutSeatIds=[], note="sophia_single baseline",
    )


def _team_agents(case: dict, client, *, seat_clients=None) -> object:
    gold = case.get("externalGold")
    return deliberate_team(
        case["prompt"], client=client, seat_clients=seat_clients,
        council_id=case.get("councilId"), gold=gold, max_seats=3,
    )


def _rate(rows: list[dict], key: str) -> float:
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 4) if rows else 0.0


def _eval_cases(cases: list[dict], *, client, condition: str, seat_clients=None) -> list[dict]:
    rows = []
    for case in cases:
        if condition == "sophia_single":
            d = _single_agent(case, client)
        else:
            d = _team_agents(case, client, seat_clients=seat_clients)
        sc = score_deliberation(d, case)
        rows.append({
            "id": case["id"],
            "caseKind": case.get("caseKind"),
            "passed": sc.passed,
            "roleFidelity": sc.roleFidelity,
            "handoffIntegrity": sc.handoffIntegrity,
            "calibratedAbstention": sc.calibratedAbstention,
            "falseConsensus": sc.falseConsensus,
        })
    return rows


def run_eval(*, mode: str = "mock", seeds: list[int], model: str = "mock", adapter: str | None = None, backend: str = "mlx", seat_models: list[str] | None = None, dry_run: bool = False) -> dict:
    manifest = verify_manifest()
    heldout = _load_jsonl_local(HELDOUT)
    probes = _load_jsonl_local(PROBE)

    if dry_run:
        return {
            "schema": "sophia.team_agents_eval.v1",
            "mode": mode,
            "dryRun": True,
            "baseModel": model.split(":", 1)[-1] if ":" in model else model,
            "adapterPath": adapter,
            "nCases": len(heldout),
            "nProbeCases": len(probes),
            "benchmarkContentHash": manifest["contentHash"],
            "canClaimAGI": False,
            "candidateOnly": True,
            "evaluatorDisjointFromTrainingGate": True,
        }

    per_seed: dict[str, list[dict]] = {"sophia_single": [], "sophia_team_orchestrator": []}
    homo_indep: list[dict] = []
    hetero_indep: list[dict] = []

    for seed in seeds:
        client = _make_client(mode=mode, seed=seed, model=model, adapter=adapter, backend=backend)
        seat_clients = None
        if seat_models:
            seat_clients = [
                _make_client(
                    mode=mode,
                    seed=seed + i + 1,
                    model=m,
                    adapter=adapter,
                    backend=backend,
                )
                for i, m in enumerate(seat_models)
            ]
        sa_rows = _eval_cases(heldout, client=client, condition="sophia_single")
        ta_rows = _eval_cases(
            heldout,
            client=client,
            condition="sophia_team_orchestrator",
            seat_clients=seat_clients,
        )
        per_seed["sophia_single"].append(sa_rows)
        per_seed["sophia_team_orchestrator"].append(ta_rows)
        homo_indep.append(measure_panel_independence(probes, client=client, max_seats=3))
        hetero_indep.append(measure_panel_independence(
            probes, client=client, seat_clients=[_mock_client(seed + 1), _mock_client(seed + 2)],
            max_seats=3,
        ))

    def _aggregate(all_rows: list[list[dict]]) -> dict:
        keys = ("passed", "roleFidelity", "handoffIntegrity", "calibratedAbstention", "falseConsensus")
        out = {}
        for k in keys:
            rates = [_rate(rows, k) for rows in all_rows]
            out[k + "Rate"] = round(statistics.fmean(rates), 4) if rates else 0.0
            out[k + "PerSeed"] = rates
        return out

    single_m = _aggregate(per_seed["sophia_single"])
    team_m = _aggregate(per_seed["sophia_team_orchestrator"])

    composite_single = [
        statistics.fmean([r["passed"] for r in rows]) for rows in per_seed["sophia_single"]
    ]
    composite_team = [
        statistics.fmean([r["passed"] for r in rows]) for rows in per_seed["sophia_team_orchestrator"]
    ]
    diff_ci = stats.bootstrap_diff_ci(composite_team, composite_single, seed=0)

    traps = [c for c in heldout if c.get("caseKind") == "coordination_trap"]
    trap_false_single = _rate(_eval_cases(traps, client=_make_client(mode=mode, seed=0, model=model, adapter=adapter, backend=backend), condition="sophia_single"),
                              "falseConsensus")
    trap_false_team = _rate(_eval_cases(traps, client=_make_client(mode=mode, seed=0, model=model, adapter=adapter, backend=backend), condition="sophia_team_orchestrator"),
                            "falseConsensus")

    homo_rho = statistics.fmean(d["meanPairwiseRho"] for d in homo_indep)
    homo_neff = statistics.fmean(d["effectiveN"] for d in homo_indep)
    hetero_rho = statistics.fmean(d["meanPairwiseRho"] for d in hetero_indep)
    hetero_neff = statistics.fmean(d["effectiveN"] for d in hetero_indep)

    report = {
        "schema": "sophia.team_agents_eval.v1",
        "mode": mode,
        "model": model,
        "baseModel": model.split(":", 1)[-1] if ":" in model else model,
        "adapterPath": adapter,
        "backend": backend,
        "seeds": seeds,
        "nCases": len(heldout),
        "nProbeCases": len(probes),
        "benchmarkContentHash": manifest["contentHash"],
        "candidateOnly": True,
        "canClaimAGI": False,
        "evaluatorDisjointFromTrainingGate": True,
        "conditions": {
            "sophia_single": single_m,
            "sophia_team_orchestrator": team_m,
        },
        "compositeDiff": {
            "point": round(statistics.fmean(composite_team) - statistics.fmean(composite_single), 4),
            "ci95": diff_ci,
            "ciExcludesZero": diff_ci[0] > 0 or diff_ci[1] < 0,
        },
        "trapFalseConsensus": {
            "sophia_single": trap_false_single,
            "sophia_team_orchestrator": trap_false_team,
        },
        "independence": {
            "homo": {
                "meanPairwiseRho": round(homo_rho, 4),
                "effectiveN": round(homo_neff, 4),
                "claimsConsensus": homo_neff >= CONSENSUS_THRESHOLD_N_EFF,
                "label": (
                    "correlated panel — not consensus"
                    if homo_neff < CONSENSUS_THRESHOLD_N_EFF
                    else "effective panel"
                ),
            },
            "hetero": {
                "meanPairwiseRho": round(hetero_rho, 4),
                "effectiveN": round(hetero_neff, 4),
                "claimsConsensus": hetero_neff >= CONSENSUS_THRESHOLD_N_EFF,
                "label": (
                    "correlated panel — not consensus"
                    if hetero_neff < CONSENSUS_THRESHOLD_N_EFF
                    else "effective panel"
                ),
            },
            "consensusThresholdNEff": CONSENSUS_THRESHOLD_N_EFF,
        },
        "promotionNote": (
            "Internal gate only — run promote_adapter.py with solverChecked after a "
            "validated adapter ladder; this mock eval is not promotion evidence."
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("mock", "real"), default="mock")
    ap.add_argument("--model", default="mock", help="model spec or bare name with --backend")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (sets SOPHIA_MLX_ADAPTER)")
    ap.add_argument("--backend", choices=("mlx", "hf"), default="mlx")
    ap.add_argument("--seat-models", default="", help="comma-separated specs for heterogeneous seats")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds (≥3)")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not args.dry_run and len(seeds) < 3:
        print("ERROR: need ≥3 seeds for CI reporting", file=sys.stderr)
        return 1
    if args.mode == "real" and args.model == "mock":
        print("ERROR: --mode real requires --model", file=sys.stderr)
        return 1

    seat_models = [s.strip() for s in args.seat_models.split(",") if s.strip()] or None
    report = run_eval(
        mode=args.mode,
        seeds=seeds,
        model=args.model,
        adapter=args.adapter,
        backend=args.backend,
        seat_models=seat_models,
        dry_run=args.dry_run,
    )
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "nCases", "benchmarkContentHash", "compositeDiff", "independence", "canClaimAGI",
    ) if k in report}, indent=2))
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        rel = out
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
