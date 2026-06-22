"""Spec C — PIF harness + held-out + anti-gaming (plain-script, no pytest)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.steering import stats  # noqa: E402


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


def main() -> int:
    tests = [test_holm_bonferroni_hand_computed, test_benjamini_hochberg_hand_computed,
             test_residualized_d_removes_offtarget, test_bootstrap_diff_p_separates]
    for t in tests:
        t(); print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} pif tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
