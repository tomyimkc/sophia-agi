#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.layered_memory import LayeredMemory, demo_memory_report  # noqa: E402


def test_trusted_memory_requires_accepted_evidence() -> None:
    mem = LayeredMemory()
    held = mem.write(layer="semantic", content="unsupported", verdict="held")
    assert held["ok"] is False
    ok = mem.write(layer="semantic", content="supported claim", verdict="accepted", confidence=0.9, evidence=[{"id": "src"}])
    assert ok["ok"] is True


def test_memory_retrieval_scores_trusted_records() -> None:
    mem = LayeredMemory()
    mem.write(layer="working", content="supported claim scratch", verdict="held", confidence=0.2)
    sem = mem.write(layer="semantic", content="supported claim", verdict="accepted", confidence=0.9, evidence=[{"id": "src"}])
    rows = mem.retrieve("supported claim")
    assert rows[0]["id"] == sem["id"]


def test_memory_demo_invariants() -> None:
    rep = demo_memory_report()
    assert all(rep["invariants"].values())


def main() -> int:
    test_trusted_memory_requires_accepted_evidence()
    test_memory_retrieval_scores_trusted_records()
    test_memory_demo_invariants()
    print("test_layered_memory: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
