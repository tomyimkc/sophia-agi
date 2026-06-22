"""Spec A — personality measurement gate tests (plain-script style, no pytest)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import personality_map as pm
from agent import personality_measure as pmeasure


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


def test_parse_rating() -> None:
    assert pmeasure.parse_rating("5") == 5
    assert pmeasure.parse_rating("Answer: 3") == 3
    assert pmeasure.parse_rating("B") == 4  # A=5, B=4, C=3, D=2, E=1
    assert pmeasure.parse_rating("nonsense") is None


def test_score_items_uniform_high() -> None:
    bank = pmeasure.load_bank()
    # Rate every item so it reverse-keys to 5 (pos->5, neg-raw->1 -> 6-1=5).
    resp = {it["id"]: (5 if it["keyed"] == 1 else 1) for it in bank["items"]}
    scored = pmeasure.score_items(resp, bank)
    for dim in ("O", "C", "E", "A", "N"):
        assert abs(scored["dimensions"][dim]["mean"] - 5.0) < 1e-9, scored


def test_score_items_reverse_key() -> None:
    bank = pmeasure.load_bank()
    # Answer 5 to BOTH the O+ and O- items: O- reverses to 1 -> O mean = 3.0.
    resp = {it["id"]: 3 for it in bank["items"]}
    resp["o_pos"] = 5
    resp["o_neg"] = 5  # raw 5 -> reverse 6-5=1
    scored = pmeasure.score_items(resp, bank)
    assert abs(scored["dimensions"]["O"]["mean"] - 3.0) < 1e-9, scored["dimensions"]["O"]


def test_score_items_missing() -> None:
    bank = pmeasure.load_bank()
    resp = {it["id"]: 4 for it in bank["items"]}
    resp["o_pos"] = None
    scored = pmeasure.score_items(resp, bank)
    assert scored["missing"] == 1


class _StubResult:
    def __init__(self, text):
        self.ok = True
        self.text = text


class _StubClient:
    """Returns a scripted rating per item id, in administration order."""
    def __init__(self, ratings_by_text):
        self.ratings_by_text = ratings_by_text
        self.calls = []

    def generate(self, system, user, **kw):
        self.calls.append((system, user))
        for frag, rating in self.ratings_by_text.items():
            if frag in user:
                return _StubResult(str(rating))
        return _StubResult("3")


def test_measure_ocean_with_stub() -> None:
    bank = pmeasure.load_bank()
    # Make every keyed item resolve to 5 after reverse-keying.
    ratings = {it["text"]: (5 if it["keyed"] == 1 else 1) for it in bank["items"]}
    client = _StubClient(ratings)
    out = pmeasure.measure_ocean(client, bank=bank, seed=7)
    assert out["missing"] == 0
    assert len(client.calls) == len(bank["items"])  # one stateless call per item
    for dim in ("O", "C", "E", "A", "N"):
        assert abs(out["dimensions"][dim]["mean"] - 5.0) < 1e-9


def test_measure_ocean_mock_smoke() -> None:
    os.environ["SOPHIA_MOCK_RESPONSE"] = "4"
    try:
        from agent.model import default_client
        client = default_client("mock")
        out = pmeasure.measure_ocean(client, seed=1)
        assert out["missing"] == 0  # every item parsed "4"
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)


def test_measure_ocean_persona_in_system() -> None:
    bank = pmeasure.load_bank()
    client = _StubClient({it["text"]: 3 for it in bank["items"]})
    pmeasure.measure_ocean(client, bank=bank, persona="You are very extraverted.", seed=0)
    assert all("very extraverted" in sys for sys, _ in client.calls)


def test_mcp_mbti_type_record() -> None:
    from sophia_mcp.tools_impl import mbti_type_record
    rec = mbti_type_record("intj")
    assert rec["code"] == "INTJ" and rec["ocean"]["N"] is None
    bad = mbti_type_record("ZZZZ")
    assert "error" in bad and len(bad["sampleIds"]) == 16


def test_mcp_personality_target_mock() -> None:
    from sophia_mcp.tools_impl import personality_target
    out = personality_target("ENFP", {"E": "high"}, "Say hello.", model="mock", gate=False)
    assert out["mbti"] == "ENFP" and out["response"]
    assert "error" not in out


def test_mcp_personality_faithful_score() -> None:
    from sophia_mcp.tools_impl import personality_faithful_score
    out = personality_faithful_score("MBTI is just the Big Five renamed.", "INTJ", {}, model="mock")
    assert out["passed"] is False  # framework merge -> contradicted


def test_verifier_corpus_merge_cases() -> None:
    import json as _json
    corpus = _json.loads((ROOT / "benchmark" / "personality_faithful.json").read_text(encoding="utf-8"))
    ver = pm_ver()
    for case in corpus["cases"]:
        if case["kind"] != "merge":
            continue  # Spec A deterministically checks the merge/myth cases only
        verdict = ver(case["proposition"], None, {})
        expect_pass = case["expectFaithful"]
        assert verdict["passed"] == expect_pass, (case["id"], verdict)


def pm_ver():
    from agent.verifiers import personality_faithful
    return personality_faithful()


def test_skill_frontmatter_valid() -> None:
    md = (ROOT / "skills" / "portable" / "sophia-personality-faithful" / "SKILL.md").read_text(encoding="utf-8")
    assert md.startswith("---")
    head = md.split("---", 2)[1].lower()
    assert "name:" in head and "description:" in head
    assert "claude" not in head and "anthropic" not in head  # naming rule
    assert "<" not in head and ">" not in head  # no angle brackets in frontmatter


def main() -> int:
    tests = [
        test_mbti_to_ocean_all_types,
        test_mbti_to_ocean_intj,
        test_mbti_to_ocean_invalid,
        test_ocean_to_mbti_letters_roundtrip,
        test_build_type_records,
        test_parse_rating,
        test_score_items_uniform_high,
        test_score_items_reverse_key,
        test_score_items_missing,
        test_measure_ocean_with_stub,
        test_measure_ocean_mock_smoke,
        test_measure_ocean_persona_in_system,
        test_mcp_mbti_type_record,
        test_mcp_personality_target_mock,
        test_mcp_personality_faithful_score,
        test_verifier_corpus_merge_cases,
        test_skill_frontmatter_valid,
    ]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} personality tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
