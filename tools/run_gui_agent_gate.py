#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Demo the fail-closed screenshot-grounded GUI agent gate (offline).

Every proposed GUI action is re-verified against the screenshot's ground-truth
elements before dispatch: a click is allowed only if its named element exists and
the coordinate lands on it; otherwise it is withheld fail-closed and escalated.
Shows a grounded action stream (all dispatched) vs a hallucinated one (a phantom
control, a coordinate that hits the wrong element, a mutating action with no
coordinate to ground) — all withheld.

    python tools/run_gui_agent_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multimodal_bench import gui_agent  # noqa: E402


def main(argv=None) -> int:
    if "--json" in (argv or sys.argv[1:]):
        print(json.dumps(gui_agent.demo(), indent=2))
        return 0
    out = gui_agent.demo()
    print("\nFail-closed GUI action gate (grounded vs hallucinated action streams):")
    for name, s in out.items():
        print(f"  {name:<13} proposed={s['proposed']} dispatched={s['dispatched']} "
              f"withheld={s['withheld']} (rate {s['withholdRate']:.2f}) escalations={s['escalations']}")
        if s["reasons"]:
            print(f"                withhold reasons: {', '.join(s['reasons'])}")
    g, b = out["grounded"], out["hallucinated"]
    print(f"\n  grounded actions all dispatched : {g['withheld'] == 0}")
    print(f"  hallucinated actions all withheld: {b['dispatched'] == 0}")
    print("  -> a confident-but-wrong click never reaches the OS; it escalates to a human.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
