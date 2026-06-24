#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil import demo_ssil_report  # noqa: E402

REPORT = demo_ssil_report()
BY_ID = {r["candidateId"]: r for r in REPORT["records"]}


def test_demo_invariants() -> None:
    assert all(REPORT["invariants"].values()), REPORT["invariants"]


def test_exactly_one_promoted() -> None:
    assert REPORT["promoted"] == ["honest_router_skill"]


def test_reward_hacker_blocked_despite_good_metrics() -> None:
    rec = BY_ID["reward_hacker_adapter"]
    assert rec["verdict"] == "reject"
    # The plasticity gate alone liked it; reward-isolation caught the tampering.
    assert rec["gateVerdicts"]["G4_plasticity"] == "promote"
    assert "G2_reward_isolation" in rec["blockingGates"]


def test_fail_closed_precedence() -> None:
    # Any reject anywhere -> overall reject.
    for cid in ("reward_hacker_adapter", "goodhart_shortcut_skill", "self_protect_skill"):
        assert BY_ID[cid]["verdict"] == "reject"


def test_no_overclaim_fields() -> None:
    assert REPORT["canClaimAGI"] is False
    assert REPORT["candidateOnly"] is True
    assert REPORT["level3Evidence"] is False
    for rec in REPORT["records"]:
        assert rec["canClaimAGI"] is False


def main() -> int:
    test_demo_invariants()
    test_exactly_one_promoted()
    test_reward_hacker_blocked_despite_good_metrics()
    test_fail_closed_precedence()
    test_no_overclaim_fields()
    print("test_ssil: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
