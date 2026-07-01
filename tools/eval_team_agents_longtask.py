#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Long-task eval: single-agent baseline vs team-agents orchestrator.

Uses the EXTERNAL scorer in ``provenance_bench/team_agents_longtask_benchmark.py``
(disjoint from the intrinsic gate). Reports sub-step coverage, role fidelity,
hand-off integrity, false-consensus on traps, and panel effective-N at ≥3 seeds
with bootstrap 95% CI.

    python tools/eval_team_agents_longtask.py --mode mock --dry-run
    python tools/eval_team_agents_longtask.py --mode real \\
        --model mlx:Qwen/Qwen2.5-3B-Instruct \\
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

from agent.steering import stats  # noqa: E402
from agent.team_agents import deliberate_team, measure_panel_independence  # noqa: E402
from provenance_bench.team_agents_benchmark import PROBE  # noqa: E402
from provenance_bench.team_agents_longtask_benchmark import (  # noqa: E402
    HELDOUT,
    score_longtask,
    verify_manifest,
)

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "team-agents-longtask.public-report.json"
SINGLE_SYSTEM = (
    "You are a careful source-disciplined Sophia advisor. "
    "Be concise; cite sources when grounded."
)
CONSENSUS_THRESHOLD_N_EFF = 2.0


def _wrap_prompt_tracker(client):
    """Record every ``generate(system, user)`` call for prompt-parity checks."""
    records: list[tuple[str, str]] = []

    def generate(system: str, user: str):
        records.append((system, user))
        return client.generate(system, user)

    wrapped = SimpleNamespace(generate=generate, spec=getattr(client, "spec", ""))
    wrapped._prompt_records = records  # type: ignore[attr-defined]
    return wrapped


def _seat_user_prompts(records: list[tuple[str, str]]) -> list[str]:
    """User strings for substantive seats (excludes synthesis-chair overhead)."""
    return [user for system, user in records if "synthesis chair" not in system.lower()]


def _verify_same_case_prompt(case_prompt: str, single_records: list[tuple[str, str]],
                             team_records: list[tuple[str, str]]) -> bool:
    """Both conditions must pass the identical case ``prompt`` to the model user slot."""
    single_users = [user for _, user in single_records]
    if len(single_users) != 1 or single_users[0] != case_prompt:
        return False
    team_seat_users = _seat_user_prompts(team_records)
    if not team_seat_users:
        return False
    return all(user == case_prompt for user in team_seat_users)


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
    """Deterministic stub tuned for long multi-step cases."""
    n = [0]

    def generate(system: str, user: str):
        n[0] += 1
        low = system.lower()
        user_low = user.lower()
        if "synthesis chair" in low:
            text = (
                "Flag conflict: Risk seat urges caution on runway while Growth seat "
                "favors deployment. Per Corporate Finance seat: runway is 25 months. "
                "Per Compliance seat: PDPO transfer rules apply. Escalate to human — "
                "not consensus. Not advice."
            )
            return SimpleNamespace(ok=True, text=text)
        if any(k in low for k in ("tail", "risk", "compliance", "caution", "judge clerk")):
            return SimpleNamespace(
                ok=True,
                text="Caution: runway burn, AML/KYC compliance, PDPO transfer risk flagged.",
            )
        if "corporate-finance" in low or "runway" in user_low or "burn" in user_low:
            return SimpleNamespace(
                ok=True,
                text="Runway is 25 months at current burn; burn multiple 2.0; monitor AML for Stripe.",
            )
        if "value-investing" in low or "ltv" in user_low:
            return SimpleNamespace(
                ok=True,
                text="Margin of safety: LTV/CAC ratio 3x looks adequate with caveats on churn.",
            )
        if "labor" in low or "minimum wage" in user_low or "gig" in user_low:
            return SimpleNamespace(
                ok=True,
                text="Minimum wage rise yields HK$44/hour; gig reclassification raises employment cost.",
            )
        if "trade" in low or "tariff" in user_low:
            return SimpleNamespace(
                ok=True,
                text="Tariff incidence shared between importer and consumer; pass-through depends on elasticity.",
            )
        if "contract" in low or "nda" in user_low or "arbitration" in user_low:
            return SimpleNamespace(
                ok=True,
                text="NDA governing law HK; arbitration clause favors HK seat; subsidiary vs entity trade-off.",
            )
        if "macro" in low or "recession" in user_low or "inflation" in user_low:
            return SimpleNamespace(
                ok=True,
                text="Recession playbook: cut discretionary spend; inflation pricing trade-off balanced.",
            )
        if "micro" in low or "network effect" in user_low or "subsidy" in user_low:
            return SimpleNamespace(
                ok=True,
                text="Network effect present with repeat buyers; subsidy shifts demand curve upward.",
            )
        return SimpleNamespace(
            ok=True,
            text=(
                f"Step A: runway and burn analysis. Step B: PDPO transfer obligations. "
                f"Step C: entity structure trade-off (seed={seed}). Not advice."
            ),
        )

    return SimpleNamespace(generate=generate, spec=f"mock:seed{seed}")


def _single_agent(case: dict, client) -> object:
    prompt = case["prompt"]
    ans = client.generate(SINGLE_SYSTEM, prompt).text
    from agent.council_deliberate import Deliberation

    return Deliberation(
        query=prompt,
        councilId=case.get("councilId"),
        seats=[],
        guardians=[],
        synthesis=ans,
        gatedOutSeatIds=[],
        note="sophia_single baseline",
    )


def _team_agents(case: dict, client, *, seat_clients=None) -> object:
    prompt = case["prompt"]
    gold = case.get("externalGold")
    return deliberate_team(
        prompt,
        client=client,
        seat_clients=seat_clients,
        council_id=case.get("councilId"),
        gold=gold,
        max_seats=4,
    )


def _run_condition(case: dict, *, client, condition: str, seat_clients=None) -> tuple[object, list[tuple[str, str]]]:
    tracked = _wrap_prompt_tracker(client)
    if condition == "sophia_single":
        d = _single_agent(case, tracked)
    else:
        d = _team_agents(case, tracked, seat_clients=seat_clients)
    return d, tracked._prompt_records  # type: ignore[attr-defined]


def verify_prompt_parity(cases: list[dict], *, client) -> dict:
    """Assert single and team paths both receive the exact case prompt (seats only for team)."""
    mismatches: list[str] = []
    for case in cases:
        prompt = case["prompt"]
        _, single_records = _run_condition(case, client=client, condition="sophia_single")
        _, team_records = _run_condition(case, client=client, condition="sophia_team_orchestrator")
        if not _verify_same_case_prompt(prompt, single_records, team_records):
            mismatches.append(case["id"])
    return {
        "samePromptBothConditions": not mismatches,
        "nCasesChecked": len(cases),
        "mismatches": mismatches,
    }


def _rate(rows: list[dict], key: str) -> float:
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 4) if rows else 0.0


def _mean_rate(rows: list[dict], key: str) -> float:
    vals = [r.get(key, 0.0) for r in rows if key in r]
    return round(statistics.fmean(vals), 4) if vals else 0.0


def _eval_cases(cases: list[dict], *, client, condition: str, seat_clients=None) -> list[dict]:
    rows = []
    for case in cases:
        d, _records = _run_condition(case, client=client, condition=condition, seat_clients=seat_clients)
        sc = score_longtask(d, case)
        rows.append({
            "id": case["id"],
            "caseKind": case.get("caseKind"),
            "passed": sc.passed,
            "taskCompletion": sc.taskCompletion,
            "subStepCoverage": sc.subStepCoverage,
            "subStepCoverageOk": sc.subStepCoverageOk,
            "roleFidelity": sc.roleFidelity,
            "handoffIntegrity": sc.handoffIntegrity,
            "calibratedAbstention": sc.calibratedAbstention,
            "falseConsensus": sc.falseConsensus,
        })
    return rows


def run_eval(
    *,
    mode: str = "mock",
    seeds: list[int],
    model: str = "mock",
    adapter: str | None = None,
    backend: str = "mlx",
    seat_models: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    manifest = verify_manifest()
    heldout = _load_jsonl_local(HELDOUT)
    probes = _load_jsonl_local(PROBE) if PROBE.exists() else []

    if dry_run:
        return {
            "schema": "sophia.team_agents_longtask_eval.v1",
            "mode": mode,
            "dryRun": True,
            "baseModel": model.split(":", 1)[-1] if ":" in model else model,
            "adapterPath": adapter,
            "nCases": len(heldout),
            "benchmarkContentHash": manifest["contentHash"],
            "samePromptBothConditions": True,
            "canClaimAGI": False,
            "candidateOnly": True,
            "evaluatorDisjointFromTrainingGate": True,
        }

    parity_client = _make_client(mode=mode, seed=0, model=model, adapter=adapter, backend=backend)
    prompt_parity = verify_prompt_parity(heldout, client=parity_client)
    if not prompt_parity["samePromptBothConditions"]:
        raise RuntimeError(
            "prompt parity check failed — single and team must receive identical case prompts: "
            + ", ".join(prompt_parity["mismatches"])
        )

    per_seed: dict[str, list[list[dict]]] = {
        "sophia_single": [],
        "sophia_team_orchestrator": [],
    }
    homo_indep: list[dict] = []

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
        if probes:
            homo_indep.append(measure_panel_independence(probes, client=client, max_seats=3))

    def _aggregate(all_rows: list[list[dict]]) -> dict:
        keys = (
            "passed",
            "taskCompletion",
            "subStepCoverageOk",
            "roleFidelity",
            "handoffIntegrity",
            "calibratedAbstention",
            "falseConsensus",
        )
        out: dict = {}
        for k in keys:
            rates = [_rate(rows, k) for rows in all_rows]
            out[k + "Rate"] = round(statistics.fmean(rates), 4) if rates else 0.0
            out[k + "PerSeed"] = rates
        out["subStepCoverageMean"] = round(
            statistics.fmean([_mean_rate(rows, "subStepCoverage") for rows in all_rows]), 4,
        )
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

    traps = [c for c in heldout if c.get("caseKind") == "long_coordination_trap"]
    trap_client = _make_client(mode=mode, seed=0, model=model, adapter=adapter, backend=backend)
    trap_false_single = _rate(
        _eval_cases(traps, client=trap_client, condition="sophia_single"),
        "falseConsensus",
    )
    trap_false_team = _rate(
        _eval_cases(traps, client=trap_client, condition="sophia_team_orchestrator"),
        "falseConsensus",
    )

    homo_neff = statistics.fmean(d["effectiveN"] for d in homo_indep) if homo_indep else 0.0
    homo_rho = statistics.fmean(d["meanPairwiseRho"] for d in homo_indep) if homo_indep else 0.0

    report = {
        "schema": "sophia.team_agents_longtask_eval.v1",
        "mode": mode,
        "model": model,
        "baseModel": model.split(":", 1)[-1] if ":" in model else model,
        "adapterPath": adapter,
        "backend": backend,
        "seeds": seeds,
        "nCases": len(heldout),
        "benchmarkContentHash": manifest["contentHash"],
        "samePromptBothConditions": prompt_parity["samePromptBothConditions"],
        "promptParity": prompt_parity,
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
        "nCases", "benchmarkContentHash", "samePromptBothConditions", "compositeDiff",
        "conditions", "canClaimAGI",
    ) if k in report}, indent=2))
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        rel = out
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
