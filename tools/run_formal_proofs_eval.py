#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the formal-proofs eval split through the self-extending proof loop.

Loads `formal_proofs/eval/{closed-smoke,open-problems}.jsonl`, enforces the
contamination rule (open problems are EVAL-ONLY — never train/held-out), and runs:

  - `closed_loop_on_proofs` on the smoke split (the closed-loop measurement).
  - An abstention measurement on the open-problems split: the loop MUST abstain on every
    open problem (reward 0.0, routeAfter "abstain"). A non-zero reward on an open
    problem would be either a real mathematical breakthrough OR a contamination/bug —
    both flag, never silently promote.

Honest scope: `candidateOnly: true`. A real Lean toolchain is required for reward > 0 on
the smoke split; without it the smoke loop fail-closed abstains (which is itself the
correct, tested outcome). The open-problem abstentions are kernel-independent: there is
no proof text, so no kernel can accept them regardless of availability.

Usage:
    python tools/run_formal_proofs_eval.py [--out agi-proof/benchmark-results/formal-proofs-eval.public-report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.lean_verifier import lean_available  # noqa: E402
from selfextend.proof_verifier import (  # noqa: E402
    ProofAttempt,
    close_loop_on_proofs,
    proof_reward,
)

EVAL_DIR = ROOT / "formal_proofs" / "eval"
SMOKE = EVAL_DIR / "closed-smoke.jsonl"
OPEN = EVAL_DIR / "open-problems.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "formal-proofs-eval.public-report.json"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _attempts(rows: list[dict]) -> list[ProofAttempt]:
    return [ProofAttempt(r["claim_id"], r["proposition"], r.get("proof_text", "")) for r in rows]


def _contamination_check(smoke: list[dict], open_rows: list[dict]) -> dict:
    """No open problem may appear in the smoke (train-eligible) split. Same claim_id is
    the contract; same proposition is defence-in-depth."""
    smoke_ids = {r["claim_id"] for r in smoke}
    smoke_props = {r["proposition"] for r in smoke}
    leaks = [r["claim_id"] for r in open_rows
             if r["claim_id"] in smoke_ids or r["proposition"] in smoke_props]
    return {"clean": not leaks, "leakedClaimIds": leaks}


def run() -> dict:
    smoke_rows = _load(SMOKE)
    open_rows = _load(OPEN)

    contamination = _contamination_check(smoke_rows, open_rows)
    if not contamination["clean"]:
        # Fail closed HARD: a contaminated split invalidates every downstream number.
        return {
            "schema": "sophia.formal_proofs_eval.v1",
            "ok": False,
            "candidateOnly": True,
            "level3Evidence": False,
            "canClaimAGI": False,
            "claimBoundary": "Formal-proofs eval split contaminated; open problem leaked "
                             "into train-eligible smoke split. No result emitted.",
            "contamination": contamination,
        }

    smoke_att = _attempts(smoke_rows)
    open_att = _attempts(open_rows)

    # 1) Closed loop on the smoke split. Without a kernel this fail-closed abstains
    #    (the correct, tested outcome). With a kernel it can close on the easy rungs.
    closed = close_loop_on_proofs("formal-proofs-smoke", smoke_att, smoke_att, smoke_att,
                                  threshold=1.0)

    # 2) Abstention measurement on the open-problems split. There is no proof text, so
    #    proof_reward is 0.0 by construction on every open problem regardless of kernel.
    #    The invariant: ZERO open problems earn reward > 0. If one did, that is either a
    #    breakthrough or a bug — we FLAG it, never promote it.
    open_rewards = {a.claim_id: proof_reward(a) for a in open_att}
    any_open_rewarded = any(r > 0.0 for r in open_rewards.values())

    return {
        "schema": "sophia.formal_proofs_eval.v1",
        "ok": True,
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "claimBoundary": (
            "Formal-proofs eval smoke + abstention measurement. candidateOnly: this is the "
            "loop MACHINERY and its fail-closed behavior, not a capability claim. A headline "
            "result needs the no-overclaim gate (≥3 runs, CI excludes 0, independent review)."
        ),
        "kernelPresent": lean_available(),
        "contamination": contamination,
        "smokeSplit": {"n": len(smoke_att)},
        "openSplit": {"n": len(open_att)},
        "closedLoop": closed,
        "openAbstention": {
            "perProblemReward": open_rewards,
            "allAbstained": not any_open_rewarded,
            "interpretation": (
                "Every open problem earned reward 0.0 — the loop abstained on each. This is "
                "the designed, wisdom-before-intelligence output. (A reward > 0 here would be "
                "flagged as a breakthrough-or-bug, never auto-promoted.)"
            ) if not any_open_rewarded else (
                "AT LEAST ONE OPEN PROBLEM EARNED REWARD > 0 — FLAGGED. This is either a "
                "genuine mathematical breakthrough or a contamination/bug; it is NOT "
                "auto-promoted and must be human-reviewed before any claim."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    report = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # Human-readable summary to stdout (the JSON is the artefact).
    print(f"formal-proofs eval -> {out}")
    print(f"  kernel present: {report.get('kernelPresent')}")
    if not report.get("ok"):
        print(f"  CONTAMINATION: {report['contamination']}")
        return 2
    cl = report["closedLoop"]
    print(f"  smoke closed-loop: loop_closed={cl['loop_closed']} "
          f"routeAfter={cl['routeAfter']} heldoutReward={cl['heldoutReward']}")
    oa = report["openAbstention"]
    print(f"  open-problems abstained: {oa['allAbstained']} "
          f"(n={report['openSplit']['n']})")
    if not oa["allAbstained"]:
        print("  *** FLAG: an open problem earned reward > 0 — review required ***")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
