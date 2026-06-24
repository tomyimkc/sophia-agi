# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Layer-1 hook — a weight delta (LoRA/RLVR adapter) through the SAME SSIL gate.

The payoff of the layered design (docs/11-Platform/Safe-Self-Improvement-Loop.md §2):
a weight delta is gated by the IDENTICAL orchestrator that gates a skill. No safety
proof is re-opened — we show `run_ssil` accepts/rejects an adapter the same way.

No GPU here. A real RunPod RLVR/LoRA job (see AGI-Infrastructure-Plan.md) produces an
adapter and a held-out eval; this module models that as an `UpdateCandidate` of
`kind="lora_adapter"` carrying *measured* before/after metrics, plus the same
corrigibility / honeypot / reward-isolation evidence required of any candidate.
Swap the mock metrics for `tools/eval_rlvr_adapter.py` output and the wiring is real.
"""
from __future__ import annotations

import json
from typing import Any

from agent.continual_plasticity import EvalMetric, UpdateCandidate
from agent.corrigibility_gate import CorrigibilityCandidate, load_frozen_eval
from agent.honeypot_gate import load_honeypots, select_active
from agent.reward_isolation import AccessTrace, load_surface
from agent.ssil import SSILCandidate, run_ssil
from agent.ssil_proposer import _full_surface


def adapter_candidate(adapter_id: str, *, before: float, after: float, protected_before: float, protected_after: float,
                      seed: int = 7, contaminated: bool = False, tampered: bool = False) -> SSILCandidate:
    """Build an SSIL candidate from a (mock) trained adapter's held-out eval."""
    hp_spec = load_honeypots()
    active = select_active(hp_spec, seed=seed)
    corr_spec = load_frozen_eval()
    all_corr = {s["id"]: True for s in corr_spec["scenarios"]}

    writes = frozenset({"training/rlvr/checkpoints/" + adapter_id})
    if tampered:  # an adapter job that also edited the gate/threshold surface
        writes = writes | {"agent/continual_plasticity.py"}

    return SSILCandidate(
        id=adapter_id,
        access=AccessTrace(reads=frozenset({"training/corpus.jsonl"}), writes=writes, monitored=True,
                           proposed_text=f"RLVR adapter {adapter_id}"),
        update=UpdateCandidate(id=adapter_id, kind="lora_adapter", verifier_artifacts=("rlvr-heldout", "protected-suite"),
                               contaminated=contaminated,
                               metrics=(EvalMetric("task_quality", before, after, protected=False),
                                        EvalMetric("source_discipline", protected_before, protected_after, protected=True))),
        target_suite="task_quality",
        honeypots={hp: False for hp in active},
        corrigibility=CorrigibilityCandidate(id=adapter_id, kind="lora_adapter", before=_full_surface(), after=_full_surface(),
                                             frozen_eval_results=dict(all_corr)),
        seed=seed,
        notes=f"weight delta {adapter_id}",
    )


def run_layer1(candidate: SSILCandidate) -> dict[str, Any]:
    return run_ssil(candidate, surface=load_surface(), honeypot_spec=load_honeypots(), corrigibility_eval=load_frozen_eval())


def demo_layer1_report() -> dict[str, Any]:
    clean = adapter_candidate("sophia-rlvr-good", before=0.71, after=0.80, protected_before=0.98, protected_after=0.98)
    regress = adapter_candidate("sophia-rlvr-regress", before=0.71, after=0.85, protected_before=0.98, protected_after=0.90)
    tamper = adapter_candidate("sophia-rlvr-tamper", before=0.71, after=0.90, protected_before=0.98, protected_after=0.98, tampered=True)

    recs = {c.id: run_layer1(c) for c in (clean, regress, tamper)}
    return {
        "schema": "sophia.ssil_layer1_demo.v1", "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "records": recs,
        "invariants": {
            "clean_adapter_promotes": recs["sophia-rlvr-good"]["verdict"] == "promote",
            "protected_regression_rejects": recs["sophia-rlvr-regress"]["verdict"] == "reject",
            "tampering_adapter_rejected": recs["sophia-rlvr-tamper"]["verdict"] == "reject"
            and "G2_reward_isolation" in recs["sophia-rlvr-tamper"]["blockingGates"],
            "same_gate_as_skills": set(recs["sophia-rlvr-good"]["gateVerdicts"]) == {
                "G2_reward_isolation", "G4_plasticity", "G5_honeypots", "G6_corrigibility"},
        },
    }


if __name__ == "__main__":
    print(json.dumps(demo_layer1_report(), ensure_ascii=False, indent=2))
