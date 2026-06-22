"""Spec B — activation-steering math + verdict tests (plain-script style, no pytest)."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.steering import vectors as vec  # noqa: E402
from agent.steering import compose  # noqa: E402


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


class _Rng:
    """Tiny deterministic LCG so the test needs no numpy."""
    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFF
    def unit(self) -> float:  # in [-1, 1)
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return (self.s / 0x3FFFFFFF) - 1.0


def main() -> int:
    tests = [
        test_normalize_unit_length,
        test_diff_of_means_recovers_direction,
        test_mock_vector_deterministic_unit,
        test_gram_schmidt_orthogonal,
        test_soft_project_reduces_overlap,
        test_compose_sums_normalized_axes,
    ]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} steering tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
