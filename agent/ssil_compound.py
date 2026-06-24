"""Compounding SSIL loop — make the "recursive" in RSI literal, every round gated.

Each round:
  1. baseline = current canonical best from the registry (or the weak baseline);
  2. a proposer suggests an executable policy (live model or scripted);
  3. measure its held-out gain *vs the current canonical best* (this is compounding);
  4. run the full two-key gate set — G1 value + G3 capability + G2/G4/G5/G6;
  5. on promote, append to the registry; a spec becomes canonical after N independent
     replications and then becomes the baseline round k+1 must beat.

The loop converges honestly: once the canonical best is optimal, no proposal clears
the improvement floor and promotions stop (bounded self-improvement, not runaway).
Counterfactual revert is available via the registry ("moral bisect").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from agent.continual_plasticity import UpdateCandidate
from agent.corrigibility_gate import CorrigibilityCandidate, load_frozen_eval
from agent.honeypot_gate import load_honeypots, select_active
from agent.reward_isolation import AccessTrace, load_surface
from agent.ssil import SSILCandidate, run_ssil
from agent.ssil_gates_ext import g1_moral_gate, g3_capability_gate, routing_verification_task
from agent.ssil_microtask import PolicySpec, baseline_spec, load_cases, measure_policy
from agent.ssil_proposer import _full_surface
from agent.ssil_registry import Registry

# A proposer maps (round_idx, current_baseline_spec_dict) -> proposed spec dict.
Proposer = Callable[[int, dict[str, Any]], dict[str, Any]]


def _proposal_text(spec: dict[str, Any]) -> str:
    return f"provenance routing policy min_sources={spec.get('min_sources')} min_quality={spec.get('min_quality')} default={spec.get('default_action')}"


def scripted_proposer(sequence: list[dict[str, Any]]) -> Proposer:
    """Offline proposer that yields a fixed, deterministic sequence of specs."""
    def _fn(round_idx: int, _baseline: dict[str, Any]) -> dict[str, Any]:
        return sequence[min(round_idx, len(sequence) - 1)]
    return _fn


def _assemble(cid: str, spec_dict: dict[str, Any], metrics, *, honeypots: dict[str, bool],
              corr_results: dict[str, bool], seed: int) -> SSILCandidate:
    return SSILCandidate(
        id=cid,
        access=AccessTrace(reads=frozenset({"eval/ssil_microtask/provenance_routing.v1.jsonl"}),
                           writes=frozenset({"rules/provenance_routing.json"}), monitored=True,
                           proposed_text=_proposal_text(spec_dict)),
        update=UpdateCandidate(id=cid, kind="rule", verifier_artifacts=("heldout-measured", "protected-recall"), metrics=metrics),
        target_suite="routing_accuracy",
        honeypots=honeypots,
        corrigibility=CorrigibilityCandidate(id=cid, kind="rule", before=_full_surface(), after=_full_surface(),
                                             frozen_eval_results=corr_results),
        seed=seed,
        notes=_proposal_text(spec_dict),
    )


def run_compound_loop(
    proposer: Proposer,
    *,
    rounds: int = 5,
    canonical_n: int = 2,
    seed: int = 7,
    registry_path: str | Path | None = None,
    live_probes: bool = False,
    model_spec: str | None = None,
    stop_after_dry: int = 2,
) -> dict[str, Any]:
    """Drive the compounding loop. ``live_probes`` runs real model corrigibility/
    honeypot probes (needs a provider key); otherwise all-pass safety probes are used
    so the offline run is deterministic and isolates the compounding mechanism."""
    reg = Registry(path=Path(registry_path) if registry_path else None, canonical_n=canonical_n)
    corr_spec = load_frozen_eval()
    hp_spec = load_honeypots()
    active = select_active(hp_spec, seed=seed)
    all_corr = {s["id"]: True for s in corr_spec["scenarios"]}
    no_bait = {hp: False for hp in active}
    test_cases = load_cases("test")

    history: list[dict] = []
    dry = 0
    for r in range(rounds):
        best = reg.canonical_best()
        baseline_policy = PolicySpec.from_dict(best["spec"]) if best else baseline_spec()
        parent = best["id"] if best else None

        spec_dict = proposer(r, best["spec"] if best else baseline_spec().__dict__)
        candidate_policy = PolicySpec.from_dict(spec_dict)
        metrics, measure_detail = measure_policy(candidate_policy, baseline=baseline_policy)

        if live_probes:
            from agent.ssil_proposer import probe_corrigibility, probe_honeypots
            corr_results, _ = probe_corrigibility(spec=model_spec, eval_spec=corr_spec)
            hp_results, _ = probe_honeypots(spec=model_spec, hp_spec=hp_spec, seed=seed)
        else:
            corr_results, hp_results = dict(all_corr), dict(no_bait)

        cid = f"round{r}-{spec_dict.get('min_sources')}-{spec_dict.get('min_quality')}"
        candidate = _assemble(cid, spec_dict, metrics, honeypots=hp_results, corr_results=corr_results, seed=seed)
        extra = {
            "G1_moral": g1_moral_gate(_proposal_text(spec_dict)),
            "G3_capability": g3_capability_gate(routing_verification_task(min_sources=int(spec_dict.get("min_sources", 2))), seed=seed),
        }
        record = run_ssil(candidate, extra_gates=extra, surface=load_surface(),
                          honeypot_spec=hp_spec, corrigibility_eval=corr_spec)

        promoted = record["verdict"] == "promote"
        cand_acc = measure_detail["candidate"]["accuracy"]
        if promoted:
            reg.record(entry_id=cid, round_idx=r, spec=spec_dict, metric=cand_acc, parent=parent,
                       gate_verdicts=record["gateVerdicts"])
            dry = 0
        else:
            dry += 1

        history.append({
            "round": r, "proposal": spec_dict, "baselineAccuracy": measure_detail["baseline"]["accuracy"],
            "candidateAccuracy": cand_acc, "verdict": record["verdict"], "blocking": record["blockingGates"],
            "promoted": promoted, "canonicalAfter": reg.is_canonical(spec_dict),
        })
        if dry >= stop_after_dry:
            break

    best = reg.canonical_best()
    return {
        "schema": "sophia.ssil_compound.v1", "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "rounds": len(history), "canonicalN": canonical_n,
        "history": history,
        "registry": reg.summary(),
        "compoundingCurve": [h["candidateAccuracy"] for h in history if h["promoted"]],
        "finalCanonicalBest": best,
    }


def demo_compound_report() -> dict[str, Any]:
    """Offline: two genuine improvements, each canonicalized after N=2 replications,
    then convergence. Specs improve 0.4 -> 0.6 quality threshold (closer to gold)."""
    seq = [
        {"min_sources": 1, "min_quality": 0.4, "default_action": "abstain"},  # r0: 0.825, beats weak baseline 0.525
        {"min_sources": 1, "min_quality": 0.4, "default_action": "abstain"},  # r1: replicate -> canonical
        {"min_sources": 2, "min_quality": 0.5, "default_action": "abstain"},  # r2: 0.875, beats canonical 0.825 (+0.05)
        {"min_sources": 2, "min_quality": 0.5, "default_action": "abstain"},  # r3: replicate -> canonical
        {"min_sources": 2, "min_quality": 0.5, "default_action": "abstain"},  # r4: no gain -> converge
        {"min_sources": 2, "min_quality": 0.5, "default_action": "abstain"},
    ]
    rep = run_compound_loop(scripted_proposer(seq), rounds=6, canonical_n=2, seed=7, stop_after_dry=2)
    curve = rep["compoundingCurve"]
    rep["invariants"] = {
        "at_least_two_promotions": sum(h["promoted"] for h in rep["history"]) >= 2,
        "curve_is_nondecreasing": all(b >= a for a, b in zip(curve, curve[1:])),
        "reached_canonical_best": rep["finalCanonicalBest"] is not None,
        "no_overclaim": rep["canClaimAGI"] is False,
    }
    return rep


if __name__ == "__main__":
    print(json.dumps(demo_compound_report(), ensure_ascii=False, indent=2))
