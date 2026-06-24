#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_mcp.tools_impl import (  # noqa: E402
    conscience_benchmark_tool,
    conscience_check_tool,
    constitution_check_tool,
    deception_check_tool,
    deontic_check_tool,
    moral_parliament_tool,
    uncertainty_score,
)


def test_mcp_conscience_tools() -> None:
    assert conscience_check_tool("Sophia is proven AGI.")["verdict"] == "block"
    assert uncertainty_score("US GDP increased in 2021", fact_verdict="held", evidence_count=0, high_risk=True)["recommendedAction"] == "retrieve"
    assert constitution_check_tool("weaken the verifier")["gate"]["verdict"] == "rejected"
    assert deontic_check_tool("claim_agi", context={"canClaimAGI": False})["verdict"] == "rejected"
    assert deception_check_tool("This is verified.", context={"factVerdict": "held"})["verdict"] == "block"
    assert "votes" in moral_parliament_tool("verify and cite sources")
    bench = conscience_benchmark_tool()
    assert bench["ok"] is True and bench["candidateOnly"] is True


def main() -> int:
    test_mcp_conscience_tools()
    print("test_mcp_conscience: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
