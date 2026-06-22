"""Spec A — personality measurement gate tests (plain-script style, no pytest)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import personality_map as pm


def test_mbti_to_ocean_all_types() -> None:
    for code in pm.SIXTEEN_TYPES:
        out = pm.mbti_to_ocean(code)
        assert "error" not in out, f"{code}: {out}"
        assert set(["O", "C", "E", "A", "N"]).issubset(out), out
        assert out["N"] is None, f"Neuroticism must be undetermined for {code}"
        for k in ("O", "C", "E", "A"):
            assert out[k] in ("high", "low"), (code, k, out[k])
    assert len(pm.SIXTEEN_TYPES) == 16


def test_mbti_to_ocean_intj() -> None:
    out = pm.mbti_to_ocean("intj")  # case-insensitive
    assert out["O"] == "high" and out["C"] == "high"
    assert out["E"] == "low" and out["A"] == "low" and out["N"] is None


def test_mbti_to_ocean_invalid() -> None:
    out = pm.mbti_to_ocean("XXXX")
    assert "error" in out and len(out["available"]) == 16


def test_ocean_to_mbti_letters_roundtrip() -> None:
    for code in pm.SIXTEEN_TYPES:
        ocean = pm.mbti_to_ocean(code)
        assert pm.ocean_to_mbti_letters(ocean) == code


def test_build_type_records() -> None:
    recs = pm.build_type_records()
    assert set(recs) == set(pm.SIXTEEN_TYPES)
    assert recs["INTJ"]["ocean"]["E"] == "low"
    assert recs["INTJ"]["ocean"]["N"] is None


def main() -> int:
    tests = [
        test_mbti_to_ocean_all_types,
        test_mbti_to_ocean_intj,
        test_mbti_to_ocean_invalid,
        test_ocean_to_mbti_letters_roundtrip,
        test_build_type_records,
    ]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} personality tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
