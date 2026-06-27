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
    GENESIS_EPOCH, UNRECORDED_REINFORCEMENT, project_corpus,
)
from okf.decay_okf import plan_decay  # noqa: E402
from okf.forgetting_audit import ForgettingAudit  # noqa: E402
from okf.page import load_pages  # noqa: E402
from okf.surprise_signal import corpus_surprise  # noqa: E402
from tools.audit_cpqa_recall import classify_source  # noqa: E402

# A belief is flagged "surprising" (a novelty-consolidation candidate) when its MEASURED
# surprise exceeds the corpus average (the z-score->logistic midpoint). Fixed, documented
# semantics — see okf/surprise_signal.py; NOT tuned to a target count.
SURPRISE_FLAG_THRESHOLD = 0.5

DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "cls-consolidation-manifest.json"
DEFAULT_AUDIT_OUT = ROOT / "agi-proof" / "benchmark-results" / "cls-consolidation-audit.jsonl"


def build_selection(wiki_dir, *, min_stable_snapshots: int = 1) -> "dict":
    """Select stable, grounded, answer-bearing facts to consolidate (offline, no metrics).

    This also runs the belief-dynamics layer (``plan_decay``) over the projected corpus
    and records every consolidation selection in a tamper-evident ``ForgettingAudit``
    ledger. ``surprise`` is now a MEASURED signal (``okf.surprise_signal``, leave-one-out
    retrieval-likelihood), so the projection is no longer a placeholder on that channel.
    ``written_at``/``reinforcement_count`` remain unrecorded, so time-decay and usage-
    reinforcement are still no-ops (with the corpus at the GENESIS_EPOCH "arrival unknown"
    marker, every age is 0 and nothing decays for time). Any suppression present is driven
    by recorded ``authorConfidence`` (source discipline), NOT by an unmeasured signal.
    The SELECTED set is still stability-driven and unchanged by the dynamics — the anti-
    forgetting gate (``evaluate_update``) remains the floor downstream; surprise only
    surfaces an auditable novelty set, it never bypasses the gate.
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
    gate_cleared_set = set(gate_cleared)
    selected = select_consolidation_set(streaks, gate_cleared, min_stable_snapshots=min_stable_snapshots)

    # ---- Belief dynamics over the real corpus ----
    # surprise is MEASURED (okf.surprise_signal); written_at/reinforcement_count are still
    # unrecorded. With the corpus at GENESIS_EPOCH (age 0) plan_decay does not time-decay,
    # and with reinforcement_count=0 the surprise+usage reinforce path stays dormant; any
    # suppression is recorded-confidence-driven (epistemic_hygiene), not unmeasured-signal.
    surprise_scores = corpus_surprise(pages, graph=graph)
    surprise_norm = {nid: s.surprise for nid, s in surprise_scores.items()}
    beliefs = project_corpus(pages, surprise_by_id=surprise_norm)
    decay_plan = plan_decay(beliefs, now=GENESIS_EPOCH).to_dict()

    # MEASURED novelty set: surprising (above corpus average) AND gate-cleared. This is the
    # real, auditable effect of the surprise channel — empty under the old placeholder
    # (surprise=0 for all), non-empty now. It does NOT alter ``selected``.
    surprising = sorted(nid for nid, s in surprise_scores.items()
                        if s.surprise > SURPRISE_FLAG_THRESHOLD and nid in gate_cleared_set)

    # Tamper-evident ledger: every selected fact is a `consolidate` event (recorded first,
    # so the genesis record is a consolidate), then every surprising belief a surprise-
    # gated `reinforce` event. The chain is hash-chained and verifiable.
    audit = ForgettingAudit()
    for fid in selected:
        audit.append(_lifecycle_event("consolidate", fid, "cls_selection"))
    for fid in surprising:
        audit.append(_lifecycle_event("reinforce", fid, "surprise_gated"))
    audit_chain_valid = audit.verify()

    svals = sorted(surprise_norm.values())
    n = len(svals)
    surprise_dist = {
        "min": round(svals[0], 6), "max": round(svals[-1], 6),
        "median": round(svals[n // 2] if n % 2 else (svals[n // 2 - 1] + svals[n // 2]) / 2.0, 6),
        "mean": round(sum(svals) / n, 6),
    } if n else {"min": None, "max": None, "median": None, "mean": None}

    return {"pages": len(pages), "grounded": len(state), "gateCleared": len(gate_cleared),
            "selected": selected,
            "decayPlan": decay_plan,
            "auditChain": audit.to_list(),
            "auditChainValid": audit_chain_valid,
            # surprise is now MEASURED and projected into the live belief states.
            "surpriseSignal": {
                "measured": True,
                "method": "leave-one-out retrieval-likelihood over the OKF graph (okf.surprise_signal)",
                "distribution": surprise_dist,
                "novelCount": len(surprising),
                "novel": surprising,
                "auditReinforceEvents": len(surprising),
                "honestyCaveat": (
                    "leave-one-out P(belief | rest of memory), NOT temporal 'surprise at "
                    "first observation' (needs written_at). Normalised RELATIVE to the corpus."
                ),
            },
            "projectionHonestyNote": (
                "surprise is MEASURED (okf.surprise_signal, leave-one-out retrieval-"
                "likelihood). STILL UNRECORDED: written_at/last_reinforced_at=GENESIS_EPOCH "
                "(arrival time unknown -> age 0 -> time-decay is a no-op) and "
                "reinforcement_count=0 (unmeasured, not 'never used'). Any suppression is "
                "authorConfidence-driven (source discipline), not time. See "
                "okf/belief_state_projection HONESTY CONTRACT and RESEARCH_FOLLOWUP.md."
            ),
            "measuredSignals": {"surprise": True},
            "unrecordedPlaceholders": {
                "writtenAtEpoch": GENESIS_EPOCH,
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
        # surprise is now MEASURED and projected into the live belief states; the selected
        # set is still UNCHANGED by the dynamics (the gate stays the floor). The broader
        # layer stays level3:false while written_at/reinforcement_count remain unrecorded;
        # the surprise SIGNAL's scoped evidence is agi-proof/okf-consistency/
        # surprise-signal.public-report.json (tools/eval_okf_surprise.py).
        "beliefDynamics": {
            "level3Evidence": False,
            "decayPlan": sel["decayPlan"],
            "auditChainValid": sel["auditChainValid"],
            "auditRecordCount": len(sel["auditChain"]),
            "surpriseSignal": sel["surpriseSignal"],
            "measuredSignals": sel["measuredSignals"],
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
