# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Spec C — personality-diverse council A/B (plain-script, no pytest)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import council_personas as cp  # noqa: E402


class _R:
    def __init__(self, text, ok=True):
        self.text = text; self.ok = ok


class _StubBase:
    spec = "stub"
    def __init__(self, answer):
        self.answer = answer; self.systems = []
    def generate(self, system, user):
        self.systems.append(system)
        return _R(self.answer)


def test_persona_prefix_prepends_and_passes_through() -> None:
    base = _StubBase("hello")
    pc = cp.PersonaClient(base, "O+high", {"O": 0.9})
    r = pc.generate("SEAT SYSTEM", "q")
    assert r.ok and r.text == "hello"
    assert base.systems[-1].startswith("PERSONA")          # persona prepended
    assert "SEAT SYSTEM" in base.systems[-1]                # seat system preserved
    assert pc.spec.endswith("persona:O+high")              # reported to SeatResult.model


def test_persona_prompt_bands() -> None:
    hi = cp.ocean_persona_prompt("x", {"O": 0.9, "C": 0.9, "E": 0.9, "A": 0.9, "N": 0.9})
    lo = cp.ocean_persona_prompt("y", {"O": 0.1, "C": 0.1, "E": 0.1, "A": 0.1, "N": 0.1})
    assert hi != lo and "imaginative" in hi and "conventional" in lo


def test_council_diversity_runs_and_computes_dq() -> None:
    # A stub that always answers correctly so passrates are well-defined; the
    # point is the A/B plumbing + ΔQ, not a positive result.
    base = _StubBase("No. Confucius did not write the Dao De Jing; it is a Daoist text. "
                     "中文：孔子並未撰寫道德經。")
    profiles = [("O+", {"O": 0.9}), ("O-", {"O": 0.1}), ("E+", {"E": 0.9})]
    out = cp.council_diversity("philosophy", client=base, profiles=profiles)
    assert set(out) >= {"single", "homogeneous", "diverse", "dq", "dq_ci", "profiles"}
    assert isinstance(out["dq"], float)
    assert out["diverse"]["seat_families"] != out["homogeneous"]["seat_families"]  # diversity present


def main() -> int:
    tests = [test_persona_prefix_prepends_and_passes_through, test_persona_prompt_bands,
             test_council_diversity_runs_and_computes_dq]
    for t in tests:
        t(); print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} council tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
