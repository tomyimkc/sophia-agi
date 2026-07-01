#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Regression for review item #5: generated forge skills compile model-proposed
predicate source at RUNTIME via ``_compile_predicate``. Assert that path re-applies
the AST allowlist on every call, so a malicious ``src`` is refused at load time, not
just at synthesis time."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.verifier_synthesis import _compile_predicate  # noqa: E402

SAFE = "def check(answer):\n    return len(str(answer)) > 3\n"
UNSAFE = [
    "def check(answer):\n    return __import__('os').system('echo pwned') == 0\n",   # import
    "def check(answer):\n    return answer.__class__.__mro__ is not None\n",          # attribute/dunder
    "def check(answer):\n    return [x for x in range(10)]\n",                        # comprehension/loop
    "def check(answer):\n    return open('/etc/passwd').read()\n",                    # disallowed call
    "def evil(answer):\n    return True\n",                                          # wrong function name
]


def test_safe_predicate_compiles_and_runs():
    fn = _compile_predicate(SAFE)
    assert callable(fn)
    assert fn("hello") is True
    assert fn("ab") is False


def test_unsafe_predicates_refused_at_compile():
    for src in UNSAFE:
        assert _compile_predicate(src) is None, f"sandbox admitted unsafe src:\n{src}"


def test_generated_verifier_uses_the_sandbox():
    # The generated verifier.py (tools/sophia_skill_forge) imports _compile_predicate for
    # "proposed:" rules, so the runtime path IS this sandbox. Confirm the import target exists.
    import agent.verifier_synthesis as vs
    assert hasattr(vs, "_compile_predicate") and hasattr(vs, "_is_prime")


def main() -> int:
    test_safe_predicate_compiles_and_runs()
    test_unsafe_predicates_refused_at_compile()
    test_generated_verifier_uses_the_sandbox()
    print("test_proposed_predicate_runtime_sandbox: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
