# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Surprise-gated consolidation selector — CANDIDATE-ONLY, not wired into the live path.

This module pairs the two consolidation signals that a memory system with belief
*lifecycle* dynamics would want to cooperate:

  (a) STABILITY: a fact grounded for >= min_stable_snapshots (hippocampal replay) —
      the existing CLS signal, computed by ``agent.cls_consolidation.stability_streaks``
      / ``select_consolidation_set``.
  (b) SURPRISE: a belief that surprised the system AND was reinforced (the engram
      consolidation signal), surfaced by ``okf.decay_okf.plan_decay().reinforce``.

It lives here — and NOT in ``agent.cls_consolidation`` — because it is a *candidate*
selector that has never been merged into the live consolidation loop. It is fully
unit-tested (``tests/test_cls_surprise_consolidation.py``) and exercised by the
dynamics-evidence driver (``tools/eval_okf_belief_dynamics.py``), but until a real run
clears the anti-forgetting gate with it wired in, it carries ``level3Evidence: false``.

Design invariant: surprise PROPOSES, stability DISPOSES. By default a surprise belief
must ALSO be stable to be consolidated; ``include_surprise_only=True`` lets a
high-surprise reinforced belief consolidate ahead of the stability floor (the "novel
result worth remembering immediately" case for frontier problems). In every case the
selected set is still routed through the anti-forgetting plasticity gate downstream —
this only changes WHAT is selected, never whether it survives the gate.
"""

from __future__ import annotations


def surprise_gated_consolidation_set(
    streaks: "dict[str, int]",
    gate_cleared,
    decay_plan: "dict | None",
    *,
    min_stable_snapshots: int = 3,
    include_surprise_only: bool = False,
) -> "list[str]":
    """Augment the stability-selected set with surprise-gated beliefs.

    See module docstring for the two-signal cooperation and the default-deny stance.
    """
    cleared = set(gate_cleared)
    stable = {fid for fid, k in streaks.items() if k >= min_stable_snapshots and fid in cleared}
    reinforce = set((decay_plan or {}).get("reinforce", []))
    if include_surprise_only:
        selected = stable | (reinforce & cleared)
    else:
        # default: surprise must also be stable (surprise proposes, stability disposes)
        selected = stable | (reinforce & stable)
    return sorted(selected)


__all__ = ["surprise_gated_consolidation_set"]
