#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the executable, gated, forgetting-resistant skill library.

These prove the Factor #1 "Done when": a skill learned in one episode is reused
later with 0 regression of prior skills, and admitting/upgrading a skill can never
silently break a previously-learned skill. All cases are hand-built with tiny
verifier cases; nothing here touches a clock, a network, or randomness.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.skill_library import (  # noqa: E402
    SCHEMA,
    Skill,
    SkillLibrary,
    skill_retention_benchmark,
    verify_skill,
    version_tag,
)


def _double() -> Skill:
    return Skill(
        id="double",
        source="def solve(x): return x * 2",
        verifier_cases=({"input": 3, "expected": 6}, {"input": 0, "expected": 0}),
    )


def test_learn_safe_skill_admitted():
    lib = SkillLibrary()
    decision = lib.learn(_double())
    assert decision["decision"] == "admit", decision
    assert decision["schema"] == SCHEMA
    assert decision["candidateOnly"] is True
    assert decision["version"] == 1
    assert decision["regressedDependents"] == []
    assert lib.is_verified("double")
    assert lib.invoke("double", 9) == 18


def test_unsafe_source_rejected_fail_closed():
    lib = SkillLibrary()
    # Import is not in the AST allowlist => unsafe => reject (fail-closed).
    bad_import = Skill(
        id="evil_import",
        source="def solve(x):\n    import os\n    return x",
        verifier_cases=({"input": 1, "expected": 1},),
    )
    d1 = lib.learn(bad_import)
    assert d1["decision"] == "reject", d1
    assert "evil_import" not in lib.ids()

    # eval is rejected too.
    bad_eval = Skill(
        id="evil_eval",
        source="def solve(x): return eval('x')",
        verifier_cases=({"input": 1, "expected": 1},),
    )
    d2 = lib.learn(bad_eval)
    assert d2["decision"] == "reject", d2
    assert "evil_eval" not in lib.ids()

    # A dunder/attribute escape is rejected.
    bad_attr = Skill(
        id="evil_attr",
        source="def solve(x): return x.__class__",
        verifier_cases=({"input": 1, "expected": 1},),
    )
    d3 = lib.learn(bad_attr)
    assert d3["decision"] == "reject", d3


def test_skill_failing_own_verifier_rejected():
    lib = SkillLibrary()
    wrong = Skill(
        id="wrong_double",
        source="def solve(x): return x * 3",
        verifier_cases=({"input": 2, "expected": 4},),  # 6 != 4
    )
    d = lib.learn(wrong)
    assert d["decision"] == "reject", d
    assert "wrong_double" not in lib.ids()


def test_composite_over_admitted_dep_works():
    lib = SkillLibrary()
    assert lib.learn(_double())["decision"] == "admit"
    quad = Skill(
        id="quadruple",
        source="def solve(x): return double(double(x))",
        deps=("double",),
        verifier_cases=({"input": 3, "expected": 12}, {"input": 1, "expected": 4}),
    )
    d = lib.learn(quad)
    assert d["decision"] == "admit", d
    assert lib.invoke("quadruple", 5) == 20
    # Composition of composition.
    oct_ = Skill(
        id="octuple",
        source="def solve(x): return double(quadruple(x))",
        deps=("double", "quadruple"),
        verifier_cases=({"input": 1, "expected": 8},),
    )
    assert lib.learn(oct_)["decision"] == "admit"
    assert lib.invoke("octuple", 2) == 16


def test_composite_with_missing_dep_rejected():
    lib = SkillLibrary()
    # `double` was never admitted => verify fails closed => reject.
    quad = Skill(
        id="quadruple",
        source="def solve(x): return double(double(x))",
        deps=("double",),
        verifier_cases=({"input": 3, "expected": 12},),
    )
    d = lib.learn(quad)
    assert d["decision"] == "reject", d
    assert "quadruple" not in lib.ids()
    report = verify_skill(quad, lib)
    assert report["ok"] is False
    assert report["passed"] == 0


def test_breaking_upgrade_rejected_dependent_stays_passing():
    lib = SkillLibrary()
    assert lib.learn(_double())["decision"] == "admit"
    quad = Skill(
        id="quadruple",
        source="def solve(x): return double(double(x))",
        deps=("double",),
        verifier_cases=({"input": 3, "expected": 12}, {"input": 2, "expected": 8}),
    )
    assert lib.learn(quad)["decision"] == "admit"
    before = verify_skill(lib.get("quadruple"), lib)["passRate"]
    assert before == 1.0

    # Breaking upgrade: double now triples. quadruple expects 4*x => regresses.
    breaking = lib.learn(Skill(
        id="double",
        version=2,
        source="def solve(x): return x * 3",
        verifier_cases=({"input": 3, "expected": 9},),
    ))
    assert breaking["decision"] == "reject", breaking
    assert "quadruple" in breaking["regressedDependents"], breaking
    # Prior version kept; dependent unharmed.
    assert lib.get("double").version == 1
    assert lib.invoke("double", 4) == 8
    after = verify_skill(lib.get("quadruple"), lib)["passRate"]
    assert after == 1.0


def test_non_breaking_upgrade_admitted_version_bumps():
    lib = SkillLibrary()
    assert lib.learn(_double())["decision"] == "admit"
    quad = Skill(
        id="quadruple",
        source="def solve(x): return double(double(x))",
        deps=("double",),
        verifier_cases=({"input": 3, "expected": 12},),
    )
    assert lib.learn(quad)["decision"] == "admit"
    # Same 2*x behavior via x + x; richer spec, version bump.
    nonbreaking = lib.learn(Skill(
        id="double",
        version=2,
        source="def solve(x): return x + x",
        verifier_cases=({"input": 5, "expected": 10},),
    ))
    assert nonbreaking["decision"] == "admit", nonbreaking
    assert lib.get("double").version == 2
    assert lib.invoke("quadruple", 3) == 12
    assert verify_skill(lib.get("quadruple"), lib)["passRate"] == 1.0


def test_upgrade_must_bump_version():
    lib = SkillLibrary()
    assert lib.learn(_double())["decision"] == "admit"
    # Same version re-admit is refused (cannot silently replace at same version).
    d = lib.learn(_double())
    assert d["decision"] == "reject", d
    assert lib.get("double").version == 1


def test_retention_benchmark_ok_and_zero_forgotten():
    report = skill_retention_benchmark()
    assert report["ok"] is True, report
    assert report["forgottenSkills"] == 0
    assert report["breakingUpgradeRejected"] is True
    assert report["nonBreakingUpgradeAdmitted"] is True
    assert report["reusedAcrossEpisodes"] is True
    assert report["candidateOnly"] is True
    # Every recorded retention-matrix rate is a perfect pass (no silent forgetting).
    for row in report["retentionMatrix"]:
        for sid, rate in row["rates"].items():
            assert rate == 1.0, (row, sid, rate)


def test_retention_benchmark_deterministic_across_runs():
    a = skill_retention_benchmark()
    b = skill_retention_benchmark()
    assert a == b


def test_version_tag_deterministic():
    s = _double()
    t1 = version_tag(s)
    t2 = version_tag(_double())
    assert t1 == t2
    assert t1.startswith("double@v1#")
    # Library tag matches the module-level tag (process-independent).
    lib = SkillLibrary()
    lib.learn(s)
    assert lib.version_tag("double") == t1
    # Different source => different sha.
    other = Skill(id="double", source="def solve(x): return x + x",
                  verifier_cases=({"input": 1, "expected": 2},))
    assert version_tag(other) != t1


def test_to_update_candidate_parity_adapter():
    lib = SkillLibrary()
    lib.learn(_double())
    cand = lib.to_update_candidate("double", {
        "skill:double": (0.0, 1.0),
        "skill:quadruple": (1.0, 1.0),
    })
    target = [m for m in cand.metrics if m.suite == "skill:double"]
    protected = [m for m in cand.metrics if m.protected]
    assert target and target[0].protected is False
    assert protected and protected[0].suite == "skill:quadruple"


def test_invoke_unverified_fails_closed():
    lib = SkillLibrary()
    try:
        lib.invoke("nope", 1)
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
