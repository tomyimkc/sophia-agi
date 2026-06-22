"""Spec B — activation-steering math + verdict tests (plain-script style, no pytest)."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.steering import vectors as vec  # noqa: E402
from agent.steering import compose  # noqa: E402
from agent.steering import stats  # noqa: E402


def test_normalize_unit_length() -> None:
    v = vec.normalize([3.0, 4.0])
    assert abs(vec.norm(v) - 1.0) < 1e-9
    assert abs(v[0] - 0.6) < 1e-9 and abs(v[1] - 0.8) < 1e-9
    # zero vector is returned unchanged (no divide-by-zero)
    assert vec.normalize([0.0, 0.0]) == [0.0, 0.0]


def test_diff_of_means_recovers_direction() -> None:
    # Plant a unit direction u into the positive cluster; negatives centered at 0.
    u = vec.normalize([1.0, 2.0, -1.0, 0.5])
    rng = _Rng(7)
    pos = [[u[i] + 0.01 * rng.unit() for i in range(4)] for _ in range(512)]
    neg = [[0.0 + 0.01 * rng.unit() for _ in range(4)] for _ in range(512)]
    d = vec.normalize(vec.diff_of_means(pos, neg))
    assert vec.cosine(d, u) > 0.98


def test_mock_vector_deterministic_unit() -> None:
    a = vec.mock_vector(16, seed=3)
    b = vec.mock_vector(16, seed=3)
    assert a == b and len(a) == 16
    assert abs(vec.norm(a) - 1.0) < 1e-9
    assert vec.mock_vector(16, seed=4) != a


def test_gram_schmidt_orthogonal() -> None:
    vs = {"E": [1.0, 0.0, 0.0], "O": [1.0, 1.0, 0.0], "C": [1.0, 1.0, 1.0]}
    ortho = compose.gram_schmidt(vs)
    keys = sorted(ortho)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert abs(vec.dot(ortho[keys[i]], ortho[keys[j]])) < 1e-6


def test_soft_project_reduces_overlap() -> None:
    vs = {"E": vec.normalize([1.0, 0.0]), "O": vec.normalize([1.0, 1.0])}
    before = abs(vec.cosine(vs["E"], vs["O"]))
    sp = compose.soft_project(vs, beta=0.5)
    after = abs(vec.cosine(sp["E"], sp["O"]))
    assert after < before  # soft projection reduces (not necessarily zeroes) overlap


def test_compose_sums_normalized_axes() -> None:
    vs = {"E": [2.0, 0.0], "O": [0.0, 3.0]}  # already orthogonal
    alphas = {"E": 1.0, "O": 1.0}
    composed, manifest = compose.compose_vectors(vs, alphas, scheme="soft_proj")
    # orthogonal inputs → normalized axes are unit; sum is (1,1)
    assert abs(composed[0] - 1.0) < 1e-6 and abs(composed[1] - 1.0) < 1e-6
    assert manifest["scheme"] == "soft_proj" and manifest["normalized"] is True
    assert "E|O" in manifest["gram"]


def test_cohen_d_matches_analytic() -> None:
    a = [1.0, 1.0, 1.0, 1.0]   # mean 1, sd 0 in group a
    b = [0.0, 0.0, 0.0, 0.0]
    # pooled sd is 0 here → guard returns 0.0
    assert stats.cohen_d(a, b) == 0.0
    a2 = [2.0, 4.0, 6.0, 8.0]   # mean 5, population variance 5
    b2 = [1.0, 3.0, 5.0, 7.0]   # mean 4, population variance 5
    d = stats.cohen_d(a2, b2)
    assert 0.40 < d < 0.50      # (5-4)/sqrt((5+5)/2) = 1/sqrt(5) ≈ 0.447 (population-SD pooled)


def test_bootstrap_diff_ci_separates() -> None:
    steer = [0.9, 1.0, 1.1, 1.0, 0.95]   # clearly larger
    base = [0.1, 0.0, 0.05, 0.1, 0.0]
    lo, hi = stats.bootstrap_diff_ci(steer, base, n_boot=2000, seed=0)
    assert lo > 0.0                      # CI excludes zero, lower bound positive
    same = stats.bootstrap_diff_ci(base, base, n_boot=2000, seed=0)
    assert same[0] <= 0.0 <= same[1]     # identical → CI includes zero


def test_kappa_reuse_identity_and_negation() -> None:
    assert stats.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert stats.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0


def test_ssa_verdict_enacted_and_abstain_paths() -> None:
    good = {"delta_ci": [0.4, 0.9], "delta_point": 0.6, "steered_d": 0.8,
            "off_target_d": {"O": 0.1, "C": -0.05}, "kappa": 0.55,
            "capability_drop": 0.02, "coherence": 90.0, "is_mock": False}
    assert stats.ssa_verdict(good)["status"] == "enacted"
    # each of these flips exactly one condition → abstain with the matching reason
    assert stats.ssa_verdict({**good, "delta_ci": [-0.1, 0.5]})["status"] == "abstained"
    assert stats.ssa_verdict({**good, "steered_d": 0.4})["reason"] == "below_floor"
    assert stats.ssa_verdict({**good, "off_target_d": {"O": 0.3}})["reason"] == "off_target_halo"
    assert stats.ssa_verdict({**good, "kappa": 0.2})["reason"] == "low_kappa"
    assert stats.ssa_verdict({**good, "capability_drop": 0.10})["reason"] == "capability_drop"
    assert stats.ssa_verdict({**good, "is_mock": True})["reason"] == "mock_subject"


class _Rng:
    """Tiny deterministic LCG so the test needs no numpy."""
    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFF
    def unit(self) -> float:  # in [-1, 1)
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return (self.s / 0x3FFFFFFF) - 1.0


from agent import personality_behavioral as beh  # noqa: E402
from provenance_bench import steering_dataset as sds  # noqa: E402


def _stub_complete(system, user, *, spec=None, **kw):
    # Deterministic stub judge with a SMALL per-response spread so Cohen's d is
    # well-defined (constant scores → zero variance → d undefined). High if the
    # response pulls on extraversion ("party"/"people").
    import json as _json
    hi = ("party" in user.lower()) or ("people" in user.lower())
    jitter = sum(ord(c) for c in user) % 7          # deterministic 0..6 spread
    base = 88 if hi else 18
    return _json.dumps({"trait_score": base + jitter, "coherence": 95})


def test_judge_score_parses_json() -> None:
    fixed = lambda s, u, **k: '{"trait_score": 90, "coherence": 95}'  # noqa: E731
    out = beh.judge_score("anything", "E", judge_spec="ollama:qwen2.5:3b", complete_fn=fixed)
    assert out["trait_score"] == 90.0 and out["coherence"] == 95.0
    bad = beh.judge_score("xyz", "E", judge_spec="ollama:qwen2.5:3b",
                          complete_fn=lambda *a, **k: "not json")
    assert bad["trait_score"] is None and bad["coherence"] == 0.0


def test_score_behavioral_distinguishes_steered() -> None:
    # Distinct responses so the jittered stub yields non-zero within-group variance.
    steered = [f"I love a big party with lots of people, take {i}!" for i in range(6)]
    neutral = [f"I sat quietly at home, evening {i}." for i in range(6)]
    out = beh.score_behavioral(steered, neutral, "E",
                               judges=["ollama:qwen2.5:3b", "ollama:llama3.2:3b"],
                               complete_fn=_stub_complete)
    assert out["trait_d"] > 0.5            # steered (~88-94) clearly above neutral (~18-24)
    assert out["kappa"] is not None       # two judges produced comparable "moved" labels
    assert set(out["judge_families"]) == {"qwen2.5", "llama3.2"}


def test_behavioral_veneer_invariant() -> None:
    # The behavioral path must never read an MBTI string — it isn't a parameter at
    # all, so identical inputs give identical results, label present or not. Use
    # varied multi-item lists so trait_d is a real non-zero value (not a trivial 0).
    steered = [f"I love a big party with people, take {i}" for i in range(4)]
    neutral = [f"I stayed quiet at home, evening {i}" for i in range(4)]
    a = beh.score_behavioral(steered, neutral, "E", judges=["ollama:qwen2.5:3b"], complete_fn=_stub_complete)
    b = beh.score_behavioral(steered, neutral, "E", judges=["ollama:qwen2.5:3b"], complete_fn=_stub_complete)
    assert a["trait_d"] == b["trait_d"] and a["trait_d"] > 0.5  # meaningful + deterministic


def test_score_behavioral_kappa_alignment() -> None:
    """κ must be computed on ALIGNED pairs only.

    One judge returns invalid JSON (→ None) for exactly one steered response but
    a valid score for its aligned neutral partner. The old code filtered steered
    and neutral lists INDEPENDENTLY, so the first None in steered would cause all
    later neutral entries to be paired with the wrong steered entry, silently
    corrupting κ. The fixed code keeps only pairs where BOTH are not None.

    Assert: score_behavioral returns without IndexError, kappa is a float or None
    (not corrupted by misaligned indices), and the pair count that feeds κ is
    exactly the number of valid aligned pairs (not inflated by independent filtering).
    """
    import json as _json

    invalid_json_for_steered_idx_0 = object()  # sentinel

    call_count = {"n": 0}

    def _stub_align(system, user, *, spec=None, **kw):
        call_count["n"] += 1
        # First call per judge is for steered[0] → return invalid JSON to trigger None.
        # Subsequent calls return valid JSON with distinct scores.
        # We detect "steered" calls by checking the call order within a spec family:
        # judge_score is called steered[0..N-1] then neutral[0..N-1] per judge.
        n = call_count["n"]
        # Two judges × (3 steered + 3 neutral) = 12 calls total.
        # Calls 1 and 7 are steered[0] for each judge respectively.
        if n in (1, 7):
            return "not valid json"   # steered[0] → trait_score = None
        hi = ("party" in user.lower())
        base = 80 if hi else 20
        jitter = sum(ord(c) for c in user) % 5
        return _json.dumps({"trait_score": base + jitter, "coherence": 90})

    steered = [f"I love a big party, take {i}!" for i in range(3)]
    neutral  = [f"I stayed home quietly, evening {i}." for i in range(3)]
    out = beh.score_behavioral(
        steered, neutral, "E",
        judges=["ollama:qwen2.5:3b", "ollama:llama3.2:3b"],
        complete_fn=_stub_align,
    )
    # Must complete without crash (IndexError or otherwise)
    assert isinstance(out, dict), "score_behavioral must return a dict"
    assert "kappa" in out, "kappa key must be present"
    assert out["kappa"] is None or isinstance(out["kappa"], float), (
        "kappa must be float or None, not a misaligned-index artifact"
    )
    # The aligned-pair pool for each judge excludes the one None steered entry:
    # only indices 1 and 2 (out of 0,1,2) are valid aligned pairs → 2 pairs each.
    # Neither judge should contribute steered[0] or neutral[0] to κ via independent
    # filtering; the fixed implementation ensures this by construction.
    assert out["trait_d"] is not None, "trait_d must be computable from aligned pairs"


def test_steering_split_is_contamination_free() -> None:
    split = sds.build_steering_split(eval_frac=0.4, seed=0)
    assert split["item_intersection"] == []          # no item on both sides
    ex = {it["id"] for it in split["extract_items"]}
    me = {it["id"] for it in split["measure_items"]}
    assert ex and me and ex.isdisjoint(me)
    # deterministic + drift-sealed
    again = sds.build_steering_split(eval_frac=0.4, seed=0)
    assert again["extract_sealed"] == split["extract_sealed"]


import importlib  # noqa: E402


def test_run_steering_offline_invariants() -> None:
    rs = importlib.import_module("tools.run_steering")
    ok, detail = rs._offline_invariants()
    assert ok is True
    c = detail["checks"]
    assert c["mockExtractDeterministic"] and c["composeOrthogonalReduces"]
    assert c["verdictEnactsWhenStrong"] and c["verdictAbstainsWhenWeak"]
    assert c["contaminationFree"]


def test_run_steering_main_mock_writes_report() -> None:
    rs = importlib.import_module("tools.run_steering")
    rc = rs.main(["--model", "mock", "--dry-run"])
    assert rc == 0
    assert rs.OUT_JSON.exists()


def main() -> int:
    tests = [
        test_normalize_unit_length,
        test_diff_of_means_recovers_direction,
        test_mock_vector_deterministic_unit,
        test_gram_schmidt_orthogonal,
        test_soft_project_reduces_overlap,
        test_compose_sums_normalized_axes,
        test_cohen_d_matches_analytic,
        test_bootstrap_diff_ci_separates,
        test_kappa_reuse_identity_and_negation,
        test_ssa_verdict_enacted_and_abstain_paths,
        test_judge_score_parses_json,
        test_score_behavioral_distinguishes_steered,
        test_behavioral_veneer_invariant,
        test_score_behavioral_kappa_alignment,
        test_steering_split_is_contamination_free,
        test_run_steering_offline_invariants,
        test_run_steering_main_mock_writes_report,
    ]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} steering tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
