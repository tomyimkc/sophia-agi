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
from okf.page import load_pages  # noqa: E402
from tools.audit_cpqa_recall import classify_source  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "cls-consolidation-manifest.json"


def build_selection(wiki_dir, *, min_stable_snapshots: int = 1) -> "dict":
    """Select stable, grounded, answer-bearing facts to consolidate (offline, no metrics)."""
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
    return {"pages": len(pages), "grounded": len(state), "gateCleared": len(gate_cleared),
            "selected": selected}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default=str(ROOT / "wiki"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-stable", type=int, default=1)
    args = ap.parse_args()

    sel = build_selection(args.wiki, min_stable_snapshots=args.min_stable)
    manifest = {
        "schema": "sophia.cls_consolidation_manifest.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "gpuRequired": True,
        "selectedFacts": sel["selected"],
        "selectedCount": len(sel["selected"]),
        "corpus": {k: sel[k] for k in ("pages", "grounded", "gateCleared")},
        "protectedSuites": list(PROTECTED_SUITES),
        "nextStep": ("Train a LoRA on the selected facts on a GPU box, evaluate it, then route the "
                     "candidate through agent.continual_plasticity.evaluate_update — promote ONLY if "
                     "no protected suite regresses (the anti-forgetting gate). No metrics are "
                     "fabricated here; this manifest is the input to that gated training step."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selectedCount": manifest["selectedCount"], "corpus": manifest["corpus"],
                      "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
