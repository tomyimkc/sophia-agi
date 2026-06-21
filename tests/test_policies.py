#!/usr/bin/env python3
"""Tests for runtime verifier policies in the guarded loop.

The guarded spine is gate-agnostic: the same retrieve→generate→gate→repair/abstain
loop must enforce provenance, arithmetic, citation, code, or a synthesised gate —
selected at runtime — while the provenance default path is unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import guarded, policies  # noqa: E402
from agent import verifier_synthesis as vs  # noqa: E402


@dataclass
class _FakeResult:
    text: str
    ok: bool = True
    error: str | None = None


def _scripted_generate(responses):
    """Return a generate(system, user) that yields responses in order."""
    state = {"i": 0}

    def gen(system, user):
        i = min(state["i"], len(responses) - 1)
        state["i"] += 1
        return _FakeResult(responses[i])

    return gen


def _no_retrieval(*a, **k):
    return []


def _empty_context(chunks):
    return ""


def _complete(query, **kw):
    return guarded.guarded_complete(
        query, retrieve_fn=_no_retrieval, format_context_fn=_empty_context, **kw)


def test_get_policy_known_and_unknown() -> None:
    for name in ("provenance", "arithmetic", "citation", "code"):
        p = policies.get_policy(name)
        assert p.name == name and callable(p.verifier)
    try:
        policies.get_policy("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_arithmetic_policy_clean_pass() -> None:
    res = _complete("compute", policy="arithmetic",
                    generate=_scripted_generate(["The total is 2 + 2 = 4."]))
    assert res.action == "clean" and res.passed is True


def test_arithmetic_policy_repairs() -> None:
    # first answer is arithmetically wrong; repair fixes it.
    res = _complete("compute", policy="arithmetic", on_fail="repair",
                    generate=_scripted_generate(["2 + 2 = 5", "Corrected: 2 + 2 = 4."]))
    assert res.action == "repaired" and res.passed is True and res.attempts == 2


def test_arithmetic_policy_abstains_with_passing_abstention() -> None:
    res = _complete("compute", policy="arithmetic", on_fail="abstain",
                    generate=_scripted_generate(["2 + 2 = 5"]))
    assert res.action == "abstained" and res.ok is True
    assert res.passed is True            # generic abstention has no equations → passes arithmetic gate


def test_synthesized_gate_as_policy() -> None:
    # Synthesise an "even" gate, then use it to guard a generation.
    from agent.synthesis_eval import build_suite

    tasks, _, _ = build_suite(0)
    even = next(t for t in tasks if t["task_id"] == "even")
    gate = vs.synthesize(even, seed=0).gate
    assert gate is not None
    res = _complete("give an even number", verifier=gate, on_fail="abstain",
                    generate=_scripted_generate(["7"]))   # odd → fails synthesised gate
    assert res.passed is False or res.action in {"abstained", "repaired"}
    res2 = _complete("give an even number", verifier=gate,
                     generate=_scripted_generate(["8"]))   # even → clean
    assert res2.action == "clean" and res2.passed is True


def test_provenance_default_unchanged() -> None:
    # No policy/verifier → provenance gate still fires on a forbidden attribution.
    bad = "Confucius wrote the Dao De Jing."
    res = _complete("who wrote it", on_fail="passthrough",
                    generate=_scripted_generate([bad]))
    assert res.passed is False and res.action == "passthrough"


def main() -> int:
    test_get_policy_known_and_unknown()
    test_arithmetic_policy_clean_pass()
    test_arithmetic_policy_repairs()
    test_arithmetic_policy_abstains_with_passing_abstention()
    test_synthesized_gate_as_policy()
    test_provenance_default_unchanged()
    print("test_policies: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
