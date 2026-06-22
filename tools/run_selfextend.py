#!/usr/bin/env python3
"""Run the self-extending verification flywheel and print its falsifiable metrics.

Demonstrates the path-to-AGI engine end to end, offline:
  coverage rises as verifiers are synthesized + held-out-validated (without gaming),
  the loop transfers across domains, the causal model beats correlation under
  confounding, and a long-horizon plan recovers then halts on drift.

    python tools/run_selfextend.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selfextend import CausalGraph, run_flywheel, run_long_horizon, run_transfer  # noqa: E402

_DOMAINS = {
    "danger": [("delete the database", True), ("delete user files", True),
               ("please delete records", True), ("delete everything now", True),
               ("read the database", False), ("read user files", False),
               ("please read records", False), ("read everything now", False)],
    "question": [("what is this", True), ("what happened here", True), ("what time is it", True),
                 ("what do you mean", True), ("this is fine", False), ("it happened here", False),
                 ("the time is noon", False), ("you mean well", False)],
}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    fly = run_flywheel(_DOMAINS)
    transfer = run_transfer()
    g = CausalGraph().add_edge("C", "X", 2.0).add_edge("C", "Y", 3.0).add_edge("X", "Y", 1.0)
    lh = run_long_horizon([
        {"id": "s1", "content": "step one", "sources": ["src1"]},
        {"id": "s2", "content": "step two", "sources": [], "repair_sources": ["src2"]},
        {"id": "s3", "content": "step three", "sources": [{"id": "bad", "status": "refuted"}]},
    ])
    report = {
        "flywheel": {k: fly[k] for k in ("coverageBefore", "coverageAfter", "heldoutFalseAcceptRate", "coveredDomains")},
        "transfer": {"transferred": transfer["transferred"], "perDomain": transfer["perDomain"]},
        "causal": {"causal_effect_X_on_Y": g.causal_effect("X", "Y"),
                   "observational_coef_X_on_Y": g.observational_coef("X", "Y"),
                   "confounded": g.confounded("X", "Y")},
        "longHorizon": {k: lh[k] for k in ("effectiveHorizon", "recoveries", "driftedAt")},
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print("Self-extending flywheel:")
    print(f"  coverage {report['flywheel']['coverageBefore']:.0%} -> {report['flywheel']['coverageAfter']:.0%}"
          f"  (held-out false-accept {report['flywheel']['heldoutFalseAcceptRate']:.0%})")
    print(f"  transfer across {list(transfer['perDomain'])}: {report['transfer']['transferred']}")
    print(f"  causal: do(X)->Y = {report['causal']['causal_effect_X_on_Y']} vs "
          f"observational {report['causal']['observational_coef_X_on_Y']} (confounded: {report['causal']['confounded']})")
    print(f"  long-horizon: effective {report['longHorizon']['effectiveHorizon']} steps, "
          f"{report['longHorizon']['recoveries']} recovery, drift at {report['longHorizon']['driftedAt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
