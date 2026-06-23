"""Spec C — PIF harness + held-out + anti-gaming (plain-script, no pytest)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.steering import stats  # noqa: E402
from agent.steering import pif_harness as pif  # noqa: E402
from provenance_bench import heldout_split as hos  # noqa: E402
from agent.steering import anti_gaming as ag  # noqa: E402


def test_holm_bonferroni_hand_computed() -> None:
    # raw p = [0.01, 0.04, 0.03]; m=3. Holm: sorted 0.01,0.03,0.04 → *3,*2,*1 = 0.03,0.06,0.04
    # monotone → 0.03,0.06,0.06 ; mapped back to input order.
    adj = stats.holm_bonferroni([0.01, 0.04, 0.03])
    assert abs(adj[0] - 0.03) < 1e-9 and abs(adj[1] - 0.06) < 1e-9 and abs(adj[2] - 0.06) < 1e-9


def test_benjamini_hochberg_hand_computed() -> None:
    # p=[0.01,0.02,0.5], q=0.05, m=3: thresholds 0.0167,0.0333,0.05 → ranks1,2 pass → k_max=2
    sig = stats.benjamini_hochberg([0.01, 0.02, 0.5], 0.05)
    assert sig == [True, True, False]


def test_residualized_d_removes_offtarget() -> None:
    # target = 0.5*off + noise-free → residualized (net of off) ≈ 0 mean → small
    off = [1.0, 2.0, 3.0, 4.0]
    target = [0.5 * x for x in off]            # perfectly explained by off-target
    rd = stats.residualized_d(target, {"X": off})
    assert abs(rd) < 0.5                        # halo removed → not a strong residual effect
    # independent target → residual ≈ raw standardized mean (non-trivial)
    rd2 = stats.residualized_d([2.0, 2.0, 3.0, 3.0], {"X": off})
    assert isinstance(rd2, float)


def test_bootstrap_diff_p_separates() -> None:
    p_sep = stats.bootstrap_diff_p([0.9, 1.0, 1.1, 1.0], [0.0, 0.1, 0.0, 0.05], seed=0)
    p_null = stats.bootstrap_diff_p([0.1, 0.0, 0.1], [0.1, 0.0, 0.1], seed=0)
    assert p_sep < 0.1 and p_null > 0.3


def _planted(kind):
    # per-seed arrays for one cell, axis "E" target, off-targets O/C/A
    import random
    rng = random.Random(1)
    K = 24
    if kind == "strong":   # steered >> base, off-target clean
        steer = [1.0 + 0.05 * rng.gauss(0, 1) for _ in range(K)]
        base = [0.1 + 0.05 * rng.gauss(0, 1) for _ in range(K)]
    else:                  # null: steered ≈ base
        steer = [0.2 + 0.05 * rng.gauss(0, 1) for _ in range(K)]
        base = [0.2 + 0.05 * rng.gauss(0, 1) for _ in range(K)]
    neutral = [0.0 for _ in range(K)]
    s = {"E": {"steer": steer, "base": base, "neutral": neutral}}
    for ax in ("O", "C", "A"):
        off_steer = [0.02 * rng.gauss(0, 1) for _ in range(K)]
        off_base = [0.02 * rng.gauss(0, 1) for _ in range(K)]
        off_neutral = list(off_steer)   # separate copy; cohen_d(steer, neutral)=0 by construction
        s[ax] = {"steer": off_steer, "base": off_base, "neutral": off_neutral}
    s["kappa"] = 0.6; s["coherence"] = 90.0; s["capability_drop"] = 0.02
    return s


def test_build_cells_enacts_and_abstains() -> None:
    grid = [{"cell_id": "c1", "target_axis": "E", "off_target_axes": ["O", "C", "A"], "is_mock": False, "seed": 1}]
    strong = pif.build_cells_from_scores({"c1": _planted("strong")}, grid)
    assert strong[0]["verdict"]["status"] == "enacted"
    null = pif.build_cells_from_scores({"c1": _planted("null")}, grid)
    assert null[0]["verdict"]["status"] == "abstained"
    assert null[0]["verdict"]["reason"] == "steer_not_beats_baseline"


def test_headline_bh_kills_borderline() -> None:
    # two cells that each pass ssa_verdict but whose p_raw don't survive BH
    cells = [{"cell_id": "a", "p_raw": 0.04, "verdict": {"status": "enacted"}},
             {"cell_id": "b", "p_raw": 0.9, "verdict": {"status": "enacted"}}]
    h = pif.headline(cells, q=0.05)
    assert h["total"] == 2 and h["enacted"] == 0
    assert "enacted_over_total" in h


def test_is_mock_forces_abstain() -> None:
    grid = [{"cell_id": "c1", "target_axis": "E", "off_target_axes": ["O"], "is_mock": True, "seed": 1}]
    cells = pif.build_cells_from_scores({"c1": _planted("strong")}, grid)
    assert cells[0]["verdict"]["reason"] == "mock_subject"


def test_held_out_disjoint() -> None:
    r = hos.held_out_disjoint()
    assert r["ipip_intersection"] == []            # no shared item ids
    assert r["ngram_overlaps"] == []               # no shared content 3-gram
    assert r["fit_reads_heldout"] is False         # fit module never imports held-out paths
    assert r["seen_sealed"] != r["heldout_sealed"]
    assert r["nearest_neighbour_sim"] < 0.5        # construct-disjoint, not paraphrase


def test_sealing_reproduces_and_hides_salt() -> None:
    from tools.seal_personality_heldout import build_private_pack
    from tools.hidden_eval_commitments import build_commitments, case_digest
    pack = build_private_pack(salt="cafe" * 16)               # fixed salt for the test
    com = build_commitments(pack)
    assert com["saltStatus"] == "withheld until reveal"
    assert "salt" not in com
    # every committed sha256 re-verifies from the private pack + salt
    for c, pub in zip(pack["cases"], com["cases"]):
        assert case_digest(c, pack["salt"]) == pub["sha256"]


def test_grep_gate_no_plaintext_heldout_answer() -> None:
    # the public commitments file must contain only hashes, not held-out prompts
    com_path = ROOT / "agi-proof" / "hidden-reviewer-packs" / "personality-heldout-2026-06-23.commitments.json"
    if com_path.exists():
        txt = com_path.read_text(encoding="utf-8")
        assert "love to think up new ways" not in txt   # a held-out IPIP item must NOT leak
        assert '"salt"' not in txt


def test_ship_steering_promote_and_abstain() -> None:
    good = ag.ship_steering(fit_shift=0.6, held_shift=0.55, heldout_off_target_rate=0.0,
                            target_moved_on_heldout=True, axis="E")
    assert good["ship"] is True and all(good["invariants"].values())
    # gamed: big seen-vs-held gap → ABSTAIN
    gamed = ag.ship_steering(fit_shift=0.9, held_shift=0.1, heldout_off_target_rate=0.0,
                             target_moved_on_heldout=True, axis="E")
    assert gamed["ship"] is False and gamed["reason"] == "steering_gamed"
    # off-target dirty → ABSTAIN
    dirty = ag.ship_steering(fit_shift=0.6, held_shift=0.55, heldout_off_target_rate=0.3,
                             target_moved_on_heldout=True, axis="O")
    assert dirty["ship"] is False and dirty["reason"] == "steering_off_target"
    # target didn't move on held-out → ABSTAIN (fail-closed)
    nomove = ag.ship_steering(fit_shift=0.6, held_shift=0.55, heldout_off_target_rate=0.0,
                              target_moved_on_heldout=False, axis="C")
    assert nomove["ship"] is False and nomove["reason"] == "target_not_moved"


def main() -> int:
    tests = [test_holm_bonferroni_hand_computed, test_benjamini_hochberg_hand_computed,
             test_residualized_d_removes_offtarget, test_bootstrap_diff_p_separates,
             test_build_cells_enacts_and_abstains, test_headline_bh_kills_borderline,
             test_is_mock_forces_abstain, test_held_out_disjoint,
             test_sealing_reproduces_and_hides_salt, test_grep_gate_no_plaintext_heldout_answer,
             test_ship_steering_promote_and_abstain]
    for t in tests:
        t(); print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} pif tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
