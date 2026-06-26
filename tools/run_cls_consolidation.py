#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prepare a CLS consolidation run — select stable wiki facts, emit a training manifest.

Complementary Learning Systems: the wiki/OKF graph is the fast hippocampus; weights are the
slow neocortex. This script does the OFFLINE, safe half of consolidation — pick the stable,
grounded (gate-cleared) facts worth distilling and write a manifest — and stops at the GPU
boundary. It does NOT fabricate eval metrics or promote anything: real distillation +
the anti-forgetting promotion gate (agent.continual_plasticity) run on a GPU box afterward,
consuming this manifest.

    python tools/run_cls_consolidation.py                 # offline: select + write manifest
    # then, on a GPU box: train a LoRA on the manifest, eval it, and route the candidate
    # through agent.continual_plasticity.evaluate_update (protected suites must not regress).

Why this split: catastrophic forgetting is re-introduced the moment knowledge enters
weights, so the only safe path is to consolidate a *small, stable* set and gate it. The
selection is the safe part; the gate is the guarantee; the training is the GPU part.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cls_consolidation import PROTECTED_SUITES, select_consolidation_set, stability_streaks  # noqa: E402
from agent.continual_retention import Snapshot, belief_state  # noqa: E402
from okf import build_graph  # noqa: E402
from okf.belief_state_projection import (  # noqa: E402
    GENESIS_EPOCH, UNRECORDED_REINFORCEMENT, UNRECORDED_SURPRISE, project_corpus,
)
from okf.decay_okf import plan_decay  # noqa: E402
from okf.forgetting_audit import ForgettingAudit  # noqa: E402
from okf.page import load_pages  # noqa: E402
from tools.audit_cpqa_recall import classify_source  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "cls-consolidation-manifest.json"
DEFAULT_AUDIT_OUT = ROOT / "agi-proof" / "benchmark-results" / "cls-consolidation-audit.jsonl"


def build_selection(wiki_dir, *, min_stable_snapshots: int = 1) -> "dict":
    """Select stable, grounded, answer-bearing facts to consolidate (offline, no metrics).

    This also runs the belief-dynamics layer (``plan_decay``) over the projected corpus
    and records every consolidation selection in a tamper-evident ``ForgettingAudit``
    ledger — the genuine value of wiring the forgetting layer into a real run. The
    dynamics plan itself is mostly trivial over the real corpus today because the
    projection's time/surprise fields are UNRECORDED PLACEHOLDERS (see
    ``okf.belief_state_projection`` HONESTY CONTRACT); the audit trail is what is real.
    The selected set is unchanged by the dynamics — the anti-forgetting gate
    (``evaluate_update``) remains the floor downstream.
    """
    pages = load_pages(wiki_dir)
    graph = build_graph(pages)
    state = belief_state(graph)                       # grounded facts -> confidence rank
    # One snapshot of the current corpus: every grounded fact has a 1-step stability streak.
    snap = Snapshot(task_id="wiki", grounded=dict(state), introduced=tuple(state))
    streaks = stability_streaks([snap])
    # Gate-cleared = grounded AND answer-bearing (a thin stub is not worth distilling yet).
    by_id = {p.id: p for p in pages}
    gate_cleared = [fid for fid in state
                    if fid in by_id and classify_source(by_id[fid])["answerBearing"]]
    selected = select_consolidation_set(streaks, gate_cleared, min_stable_snapshots=min_stable_snapshots)

    # ---- Belief dynamics over the real corpus (candidate, level3Evidence: false) ----
    # The projection carries ONLY author_confidence honestly; timestamps/surprise/
    # reinforcement are unrecorded placeholders (documented in belief_state_projection).
    # plan_decay therefore returns a real, auditable plan whose content is mostly
    # "leave as-is" — it cannot, honestly, suppress a belief on time/surprise grounds
    # because those signals were never measured. The audit ledger is the real artifact.
    beliefs = project_corpus(pages)
    decay_plan = plan_decay(beliefs, now=GENESIS_EPOCH).to_dict()

    # Record this consolidation run in the tamper-evident ledger. Each selected fact
    # is a `consolidate` event; the ledger is hash-chained and verifiable.
    audit = ForgettingAudit()
    for fid in selected:
        audit.append(_lifecycle_event("consolidate", fid, "cls_selection"))
    audit_chain_valid = audit.verify()

    return {"pages": len(pages), "grounded": len(state), "gateCleared": len(gate_cleared),
            "selected": selected,
            # Dynamics + audit — see HONESTY CONTRACT; placeholders, NOT measured signals.
            "decayPlan": decay_plan,
            "auditChain": audit.to_list(),
            "auditChainValid": audit_chain_valid,
            "projectionHonestyNote": (
                "BeliefState projected with UNRECORDED PLACEHOLDERS: written_at/"
                "last_reinforced_at=GENESIS_EPOCH, surprise=0.0 (unmeasured, not "
                "'unsurprising'), reinforcement_count=0 (unmeasured, not 'never used'). "
                "See okf/belief_state_projection HONESTY CONTRACT and "
                "agi-proof/okf-consistency/RESEARCH_FOLLOWUP.md."
            ),
            "unrecordedPlaceholders": {
                "writtenAtEpoch": GENESIS_EPOCH,
                "surprise": UNRECORDED_SURPRISE,
                "reinforcementCount": UNRECORDED_REINFORCEMENT,
            }}


def _lifecycle_event(event: str, node_id: str, reason: str):
    """Construct a LifecycleEvent without importing the dataclass at module top (keeps
    the dynamics opt-in and the failure mode local)."""
    from okf.forgetting_audit import LifecycleEvent
    return LifecycleEvent(event=event, node_id=node_id, reason=reason)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default=str(ROOT / "wiki"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--audit-out", default=str(DEFAULT_AUDIT_OUT))
    ap.add_argument("--min-stable", type=int, default=1)
    args = ap.parse_args()

    sel = build_selection(args.wiki, min_stable_snapshots=args.min_stable)
    manifest = {
        "schema": "sophia.cls_consolidation_manifest.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "gpuRequired": True,
        "selectedFacts": sel["selected"],
        "selectedCount": len(sel["selected"]),
        "corpus": {k: sel[k] for k in ("pages", "grounded", "gateCleared")},
        "protectedSuites": list(PROTECTED_SUITES),
        # The dynamics + audit wiring — see projectionHonestyNote + RESEARCH_FOLLOWUP.
        # The selected set is UNCHANGED by the dynamics (the gate stays the floor);
        # this records that the selection happened, tamper-evidently.
        "beliefDynamics": {
            "level3Evidence": False,
            "decayPlan": sel["decayPlan"],
            "auditChainValid": sel["auditChainValid"],
            "auditRecordCount": len(sel["auditChain"]),
            "projectionHonestyNote": sel["projectionHonestyNote"],
            "unrecordedPlaceholders": sel["unrecordedPlaceholders"],
        },
        "nextStep": ("Train a LoRA on the selected facts on a GPU box, evaluate it, then route the "
                     "candidate through agent.continual_plasticity.evaluate_update — promote ONLY if "
                     "no protected suite regresses (the anti-forgetting gate). No metrics are "
                     "fabricated here; this manifest is the input to that gated training step."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Append the tamper-evident lifecycle ledger as JSONL (one hash-chained event per
    # line). Each run appends; the chain's prev_hash links to the prior head.
    audit_out = Path(args.audit_out)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    with audit_out.open("a", encoding="utf-8") as fh:
        for event in sel["auditChain"]:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(json.dumps({"selectedCount": manifest["selectedCount"], "corpus": manifest["corpus"],
                      "auditChainValid": manifest["beliefDynamics"]["auditChainValid"],
                      "auditRecordCount": manifest["beliefDynamics"]["auditRecordCount"],
                      "out": args.out, "auditOut": args.audit_out}, indent=2))


if __name__ == "__main__":
    main()
