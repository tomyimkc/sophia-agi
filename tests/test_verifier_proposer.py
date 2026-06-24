#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/verifier_proposer.py (offline, deterministic).

Covers the missing-coverage gap for step 3: an LLM may *propose* predicates, but
trust is conferred ONLY by the AST sandbox + held-out meta-verification in
agent.verifier_synthesis. These tests assert:

  - the parser extracts predicates from JSON, fenced, and raw forms;
  - the ``mock`` proposer is an offline no-op (no network);
  - a good proposed predicate is admitted while a useless one is rejected;
  - an UNSAFE proposed source is dropped (never admitted, never executed).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import verifier_synthesis as vs  # noqa: E402
from agent.verifier_proposer import _extract_predicates, make_model_proposer  # noqa: E402

_EVEN = "def check(answer):\n    return int(answer) % 2 == 0"


def _even_examples() -> list[dict]:
    return [{"answer": str(n), "label": n % 2 == 0, "_idx": i} for i, n in enumerate(range(2, 22))]


def test_parser_handles_json_fence_and_raw() -> None:
    assert _extract_predicates('{"predicates": ["%s"]}' % _EVEN.replace("\n", "\\n")) == [_EVEN]
    fenced = f"Here:\n```python\n{_EVEN}\n```\ndone"
    assert _extract_predicates(fenced) == [_EVEN]
    assert _extract_predicates(_EVEN + "\n") == [_EVEN]
    # garbage / no predicate -> empty
    assert _extract_predicates("no code here") == []


def test_mock_proposer_is_offline_noop() -> None:
    assert make_model_proposer("mock")({"task_id": "t"}, [1], [2]) == []


def test_good_predicate_admitted_useless_rejected() -> None:
    def propose(task, corrects, incorrects):
        return [_EVEN, "def check(answer):\n    return True"]

    res = vs.synthesize({"task_id": "even", "examples": _even_examples()}, seed=0, propose_fn=propose)
    names = [c.name for c, _ in res.admitted]
    assert not res.abstained
    assert "proposed:0" in names          # the good even-predicate is admitted
    assert "proposed:1" not in names      # the useless 'always True' is rejected by meta-verification


def test_unsafe_proposed_source_is_dropped() -> None:
    def propose_unsafe(task, corrects, incorrects):
        return ["def check(answer):\n    return __import__('os').system('echo hacked')"]

    res = vs.synthesize({"task_id": "even2", "examples": _even_examples()}, seed=0, propose_fn=propose_unsafe)
    # no 'proposed:*' candidate may be admitted (the AST allowlist rejects it before scoring)
    assert not any(c.name.startswith("proposed:") for c, _ in res.admitted)


def main() -> int:
    test_parser_handles_json_fence_and_raw()
    test_mock_proposer_is_offline_noop()
    test_good_predicate_admitted_useless_rejected()
    test_unsafe_proposed_source_is_dropped()
    print("test_verifier_proposer: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
