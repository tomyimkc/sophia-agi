# Activation-Steering Engine + Behavioral PIF (Spec B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Level-3 activation-steering engine (CAA difference-of-means axis vectors → `register_forward_hook` on a local Phi-3.5-mini on MPS) + a behavioral PIF channel judged by local Ollama, and answer one falsifiable question — does steering beat Spec A's Level-1 persona baseline (SSA)?

**Architecture:** Two tiers, mirroring `tools/run_rlvr.py`. A **deterministic CI core** in **pure stdlib** (vector math, composition, Cohen's d, bootstrap CI, κ, the SSA verdict — `list[float]`, `math`, `statistics`, and the existing stdlib `provenance_bench` helpers) needs no torch/numpy and runs in bare CI. **torch is isolated to `agent/steering/hooks.py`** (the real path) and lazy-imported. Real Phi-3.5 runs are opt-in (`--model phi3.5`), never a CI assertion. Reuses Spec A's `score_items` / `measure_ocean` / `personality` benchmark domain verbatim. Note: `personality_faithful` is NOT used by the behavioral channel — `agent/personality_behavioral.py` is a pure independent LLM-judge panel; `personality_faithful` is available as a future optional pre-filter but is not wired in Spec B.

**Tech Stack:** Python 3.12 (CI) / 3.10.6 (local). Stdlib for the CI core. `torch>=2.3` + `transformers>=4.46.2` (installed local env: torch 2.4.0, transformers 5.5.3) for the real path only. Ollama (`qwen2.5:3b`, `llama3.2:3b`) for judges via `agent/model.py`.

## Global Constraints

Every task's requirements implicitly include these (copied from the spec):

- **NO pytest.** Tests are plain scripts (style A): `def test_*() -> None` with bare `assert`; a `main() -> int` that runs a `tests=[...]` list printing `ok {name}` then `PASS N steering tests`; ending `if __name__ == "__main__": raise SystemExit(main())`. Every test file starts with `ROOT = Path(__file__).resolve().parents[1]; if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))`.
- **CI core is pure stdlib — NO `torch`, NO `numpy` imports in `vectors.py`, `compose.py`, `stats.py`, `personality_behavioral.py`, or `tests/test_steering.py`.** Vectors are `list[float]`. `torch` is imported ONLY inside `agent/steering/hooks.py` (and the real-path branch of `run_steering.py`), lazily. CI has no `pip install` step — anything the CI tests import must already be importable (stdlib + the repo).
- **The torch hook test (`tests/test_personality_steering.py`) is SKIP-GUARDED:** `try: import torch except Exception: print("SKIP test_personality_steering (torch unavailable)"); return 0`. It runs locally (torch present), skips in CI. This mirrors `run_rlvr.py`, whose GPU path is also never CI-tested.
- **MBTI veneer-invariance (inherited from Spec A):** `mbti_to_ocean` is consumed *upstream* to pick axis signs; **no compose/verdict/judge/effect-size path reads the MBTI string.** A `veneerInvariant` test asserts the composition verdict is identical with/without an MBTI label.
- **Neuroticism:** `mbti_to_ocean` returns `N: None` always — **never steer N from an MBTI code.** N axis steering requires an explicit OCEAN sign.
- **Normalize each axis vector, steer `h ← h + alpha·v̂`** (the corrected convention). Record `normalized: true` in vector provenance.
- **`SSA = 0/N` is a legitimate result.** All SSA thresholds are **pre-registered, fixed before any run**: N=8 personas, K=20 seeds, steered residualized `d > 0.5`, superiority `Δd` point ≥ **+0.3** with bootstrap 95% CI lower bound > 0, off-target `|d| < 0.2`, `κ ≥ 0.40` (= `KAPPA_FLOOR`), capability ε = 5% relative + coherence floor 75. Within-system deltas only — never human-norm percentiles, never "MBTI type achieved".
- **Reuse, don't fork:** `provenance_bench/consensus.py:cohen_kappa(a,b)->float|None`, `provenance_bench/aggregate.py:_ci(xs,alpha=0.05)->[lo,hi]` (percentile — caller bootstraps), `KAPPA_FLOOR=0.40`; Spec A's `score_items`, `measure_ocean`, `load_bank`. (`personality_faithful` is available but is NOT called by the behavioral channel in Spec B — the behavioral channel is a pure LLM-judge panel.)
- **Security:** the OpenRouter key lives only in gitignored `.env` (`OPENROUTER_API_KEY`); never in code, spec, report, manifest, or CI. Missing key → that judge channel ABSTAINS (no error, no silent local fallback).
- Branch is `feat/activation-steering-pif` (worktree `/Users/tom/Documents/GitHub/sophia-agi-spec-b`, stacked on Spec A / PR #64). Commit after every task.

---

### Task 1: Steering vectors (pure-stdlib math + package scaffold + CI wiring)

**Files:**
- Create: `agent/steering/__init__.py`
- Create: `agent/steering/vectors.py`
- Create: `tests/test_steering.py`
- Modify: `.github/workflows/ci.yml` (add two test invocations)

**Interfaces:**
- Produces:
  - `Vector = list[float]` (type alias).
  - `dot(a, b) -> float`, `norm(a) -> float`, `scale(a, s) -> Vector`, `sub(a, b) -> Vector`, `add(a, b) -> Vector`.
  - `mean_vectors(vs: list[Vector]) -> Vector`.
  - `diff_of_means(pos: list[Vector], neg: list[Vector]) -> Vector`.
  - `normalize(v: Vector) -> Vector` (unit vector; zero vector returned unchanged).
  - `cosine(a, b) -> float`.
  - `mock_vector(dim: int, seed: int) -> Vector` (deterministic seeded unit vector).

- [ ] **Step 1: Write the failing test** — create `tests/test_steering.py`:

```python
"""Spec B — activation-steering math + verdict tests (plain-script style, no pytest)."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.steering import vectors as vec  # noqa: E402


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
    ]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} steering tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.steering'`

- [ ] **Step 3: Create the package + vectors module** — `agent/steering/__init__.py`:

```python
"""Level-3 activation steering (Spec B).

Pure-stdlib vector math + composition live here; torch is imported ONLY inside
hooks.py (the real path) and lazily, so importing this package never requires torch.
"""
from agent.steering.vectors import (
    Vector, add, cosine, diff_of_means, dot, mean_vectors, mock_vector, norm,
    normalize, scale, sub,
)

__all__ = [
    "Vector", "add", "cosine", "diff_of_means", "dot", "mean_vectors",
    "mock_vector", "norm", "normalize", "scale", "sub",
]
```

`agent/steering/vectors.py`:

```python
"""Pure-stdlib residual-stream vector math (no torch, no numpy).

A Vector is a list[float]. The real path (hooks.py) converts torch hidden states
to plain lists before calling these, so this module stays CI-testable.
"""
from __future__ import annotations

import math
import random

Vector = list  # list[float]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def scale(a: Vector, s: float) -> Vector:
    return [x * s for x in a]


def sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def mean_vectors(vs: "list[Vector]") -> Vector:
    if not vs:
        return []
    n = len(vs)
    dim = len(vs[0])
    acc = [0.0] * dim
    for v in vs:
        for i in range(dim):
            acc[i] += v[i]
    return [x / n for x in acc]


def diff_of_means(pos: "list[Vector]", neg: "list[Vector]") -> Vector:
    """CAA Eq. 1: mean(positive activations) − mean(negative activations)."""
    return sub(mean_vectors(pos), mean_vectors(neg))


def normalize(v: Vector) -> Vector:
    n = norm(v)
    return v[:] if n == 0.0 else scale(v, 1.0 / n)


def cosine(a: Vector, b: Vector) -> float:
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def mock_vector(dim: int, seed: int) -> Vector:
    """Deterministic seeded unit vector — the offline extractor stand-in."""
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    return normalize(v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: `PASS 3 steering tests`

- [ ] **Step 5: Wire both new test files into CI** — in `.github/workflows/ci.yml`, in the "OKF + provenance verifier tests" step, after the `python tests/test_personality.py` line add:

```yaml
          python tests/test_steering.py
          python tests/test_personality_steering.py
```

(`tests/test_personality_steering.py` is created in Task 4; until then it does not exist. To keep CI green between tasks, create a minimal skip-guarded stub now so the CI line resolves — Task 4 fills it in.) Create `tests/test_personality_steering.py`:

```python
"""Spec B — toy-decoder steering-hook tests (skip-guarded; torch-only)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import torch  # noqa: F401
    except Exception:
        print("SKIP test_personality_steering (torch unavailable)")
        return 0
    print("PASS 0 hook tests (toy hook tests added in Task 4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Verify both run and ci.yml parses:
`cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py && python tests/test_personality_steering.py && python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: both PASS/SKIP lines print, no YAML error.

- [ ] **Step 6: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
git add agent/steering/__init__.py agent/steering/vectors.py tests/test_steering.py tests/test_personality_steering.py .github/workflows/ci.yml
git commit -m "feat(steering): pure-stdlib vector math + package scaffold + CI wiring (Spec B Task 1)"
```

---

### Task 2: Composition + orthogonalization (`compose.py`)

**Files:**
- Create: `agent/steering/compose.py`
- Modify: `tests/test_steering.py` (append + register in `main()`)

**Interfaces:**
- Consumes: `vectors` (Task 1).
- Produces:
  - `gram_matrix(vs: dict[str, Vector]) -> dict` — pairwise cosines, keyed `"AXIS1|AXIS2"`.
  - `soft_project(vs: dict[str, Vector], beta: float = 0.5) -> dict[str, Vector]` — `d_i ← d_i − β·⟨d_i, d̂_j⟩·d̂_j` for j≠i.
  - `gram_schmidt(vs: dict[str, Vector]) -> dict[str, Vector]` — order = sorted keys.
  - `compose_vectors(vs: dict[str, Vector], alphas: dict[str, float], *, scheme: str = "soft_proj") -> tuple[Vector, dict]` — orthogonalize per scheme, normalize each, return `Σ alpha_i·v̂_i` + a manifest `{scheme, axes, gram, per_axis_norm, normalized: True}`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_steering.py` (and register all three in `main()`):

```python
from agent.steering import compose  # noqa: E402


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.steering.compose'`

- [ ] **Step 3: Write the implementation** — `agent/steering/compose.py`:

```python
"""Axis-vector composition + orthogonalization (pure stdlib).

Takes OCEAN target signs only — never an MBTI string (veneer-invariance).
Default scheme is C4 soft-projection (best signal retention per arXiv:2602.15847).
Orthogonalization reduces, it does NOT eliminate, behavioral cross-trait
interference — validate each axis behaviorally after composition.
"""
from __future__ import annotations

from agent.steering.vectors import Vector, add, cosine, dot, normalize, scale, sub


def gram_matrix(vs: "dict[str, Vector]") -> dict:
    keys = sorted(vs)
    out: dict = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            out[f"{keys[i]}|{keys[j]}"] = round(cosine(vs[keys[i]], vs[keys[j]]), 6)
    return out


def soft_project(vs: "dict[str, Vector]", beta: float = 0.5) -> "dict[str, Vector]":
    keys = sorted(vs)
    hats = {k: normalize(vs[k]) for k in keys}
    out: dict = {}
    for i in keys:
        d = vs[i][:]
        for j in keys:
            if j == i:
                continue
            d = sub(d, scale(hats[j], beta * dot(d, hats[j])))
        out[i] = d
    return out


def gram_schmidt(vs: "dict[str, Vector]") -> "dict[str, Vector]":
    keys = sorted(vs)
    basis: list = []
    out: dict = {}
    for k in keys:
        u = vs[k][:]
        for b in basis:
            u = sub(u, scale(b, dot(u, b)))
        u = normalize(u)
        basis.append(u)
        out[k] = u
    return out


def compose_vectors(vs: "dict[str, Vector]", alphas: "dict[str, float]", *,
                    scheme: str = "soft_proj") -> "tuple[Vector, dict]":
    gram_before = gram_matrix(vs)
    if scheme == "soft_proj":
        ortho = soft_project(vs)
    elif scheme == "gram_schmidt":
        ortho = gram_schmidt(vs)
    elif scheme == "raw":
        ortho = {k: v[:] for k, v in vs.items()}
    else:
        raise ValueError(f"unknown scheme {scheme!r}; use soft_proj|gram_schmidt|raw")
    keys = sorted(ortho)
    dim = len(next(iter(ortho.values()))) if ortho else 0
    composed: Vector = [0.0] * dim
    per_axis_norm: dict = {}
    for k in keys:
        vhat = normalize(ortho[k])
        per_axis_norm[k] = round(sum(x * x for x in ortho[k]) ** 0.5, 6)
        composed = add(composed, scale(vhat, alphas.get(k, 0.0)))
    manifest = {
        "scheme": scheme, "axes": keys, "gram": gram_before,
        "per_axis_norm": per_axis_norm, "alphas": {k: alphas.get(k, 0.0) for k in keys},
        "normalized": True,
    }
    return composed, manifest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: `PASS 6 steering tests`

- [ ] **Step 5: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
git add agent/steering/compose.py tests/test_steering.py
git commit -m "feat(steering): axis composition + soft-projection/Gram-Schmidt orthogonalization (Spec B Task 2)"
```

---

### Task 3: Stats + the SSA verdict (`stats.py`)

**Files:**
- Create: `agent/steering/stats.py`
- Modify: `tests/test_steering.py` (append + register)

**Interfaces:**
- Consumes: `provenance_bench.aggregate._ci`, `provenance_bench.consensus.cohen_kappa`, `KAPPA_FLOOR`.
- Produces:
  - `cohen_d(a: list[float], b: list[float]) -> float` — `(mean_a − mean_b) / pooled_sd` (pooled two-group SD; 0.0 if pooled SD == 0).
  - `bootstrap_diff_ci(steer: list[float], base: list[float], *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> list[float]` — bootstrap CI of `Δd = d(steer,neutral) − d(base,neutral)`; here `steer`/`base` are per-seed d values already, so it bootstraps the paired differences and returns `_ci`.
  - `binarize_moved(scores: list[float], neutral: list[float]) -> list[int]` — `1` if `scores[i] > neutral[i]` else `0` (for κ).
  - `SSA_THRESHOLDS: dict` — the pre-registered constants.
  - `ssa_verdict(cell: dict) -> dict` — returns `{"status": "enacted"|"abstained", "reason": str|None, "checks": {...}}`. `cell` carries `{delta_ci, delta_point, steered_d, off_target_d (dict), kappa, capability_drop, coherence, is_mock}`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_steering.py` (register in `main()`):

```python
from agent.steering import stats  # noqa: E402


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.steering.stats'`

- [ ] **Step 3: Write the implementation** — `agent/steering/stats.py`:

```python
"""Effect-size, bootstrap CI, kappa, and the pre-registered SSA verdict (pure stdlib).

Reuses provenance_bench's stdlib helpers verbatim — no new stats deps.
"""
from __future__ import annotations

import random
import statistics

from provenance_bench.aggregate import _ci, KAPPA_FLOOR
from provenance_bench.consensus import cohen_kappa  # re-exported for callers

# Pre-registered SSA thresholds — fixed before any run (spec §"Locked decisions").
SSA_THRESHOLDS = {
    "delta_point_min": 0.30,   # Δd point estimate floor
    "steered_d_min": 0.50,     # absolute residualized d floor
    "off_target_max": 0.20,    # off-target |d| null band
    "kappa_floor": KAPPA_FLOOR,  # 0.40
    "capability_eps": 0.05,    # ≤5% relative capability drop
    "coherence_floor": 75.0,
}


def cohen_d(a: "list[float]", b: "list[float]") -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = statistics.pvariance(a), statistics.pvariance(b)
    pooled = ((va + vb) / 2.0) ** 0.5
    if pooled == 0.0:
        return 0.0
    return (statistics.fmean(a) - statistics.fmean(b)) / pooled


def bootstrap_diff_ci(steer: "list[float]", base: "list[float]", *,
                      n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> "list[float]":
    """Bootstrap CI of the paired difference (steer_i − base_i). steer/base are
    per-seed effect sizes already, paired by index. Returns _ci([..]) = [lo, hi]."""
    diffs = [s - b for s, b in zip(steer, base)]
    n = len(diffs)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot.append(statistics.fmean(sample))
    return _ci(boot, alpha)


def binarize_moved(scores: "list[float]", neutral: "list[float]") -> "list[int]":
    return [1 if s > nt else 0 for s, nt in zip(scores, neutral)]


def ssa_verdict(cell: dict) -> dict:
    """Apply the six pre-registered SSA conditions; ABSTAIN on the first failure.
    Order matters only for the reported reason; all must hold to be 'enacted'."""
    T = SSA_THRESHOLDS
    checks = {}
    lo, hi = cell["delta_ci"]
    checks["superiority"] = lo > 0.0 and cell["delta_point"] >= T["delta_point_min"]
    checks["floor"] = cell["steered_d"] > T["steered_d_min"]
    checks["orthogonality"] = all(abs(d) < T["off_target_max"] for d in cell["off_target_d"].values())
    checks["corroboration"] = cell["kappa"] >= T["kappa_floor"]
    checks["capability"] = (cell["capability_drop"] < T["capability_eps"]
                            and cell["coherence"] >= T["coherence_floor"])
    checks["non_mock"] = not cell["is_mock"]
    reason_for = {
        "superiority": "steer_not_beats_baseline", "floor": "below_floor",
        "orthogonality": "off_target_halo", "corroboration": "low_kappa",
        "capability": "capability_drop", "non_mock": "mock_subject",
    }
    for key in ("non_mock", "superiority", "floor", "orthogonality", "corroboration", "capability"):
        if not checks[key]:
            return {"status": "abstained", "reason": reason_for[key], "checks": checks}
    return {"status": "enacted", "reason": None, "checks": checks}
```

> NOTE on reason precedence: the test sets exactly one condition false at a time, so the `for` order doesn't change those assertions. `non_mock` is checked first so a mock run is reported as `mock_subject` even if other (synthetic) fields look passing.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: `PASS 10 steering tests`

- [ ] **Step 5: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
git add agent/steering/stats.py tests/test_steering.py
git commit -m "feat(steering): Cohen's d, bootstrap CI, kappa reuse + pre-registered SSA verdict (Spec B Task 3)"
```

---

### Task 4: The torch hook + `SteeredClient` (`hooks.py`) + toy-decoder test

**Files:**
- Create: `agent/steering/hooks.py`
- Replace: `tests/test_personality_steering.py` (the Task-1 stub → the real toy-hook suite, skip-guarded)

**Interfaces:**
- Consumes: `vectors.Vector`.
- Produces:
  - `make_steering_hook(vec_f32, alpha)` — a `register_forward_hook` callback adding `alpha·v` to `output[0]` (tuple, 4.x) or the bare tensor (5.x); casts `v` to `hidden.dtype/device` inside.
  - `attach_hooks(model, vector: Vector, alpha: float, layers: list[int])` — context manager; converts the `list[float]` vector to an fp32 tensor, registers on `model.model.layers[L]` for each L, guarantees `handle.remove()` in `finally`.
  - `SteeredClient` — wraps an in-process HF model + tokenizer + an active steering vector; exposes `generate(system, user) -> _Result` with `.ok`/`.text` (the duck-type `measure_ocean` requires).

- [ ] **Step 1: Write the failing test** — replace `tests/test_personality_steering.py` with the real suite (skip-guard preserved; the toy module is verified-green):

```python
"""Spec B — toy-decoder steering-hook tests (skip-guarded; torch-only)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build():
    import torch
    import torch.nn as nn

    class ToyDecoderLayer(nn.Module):
        def __init__(self, d_model, seed):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.proj = nn.Linear(d_model, d_model)
            with torch.no_grad():
                self.proj.weight.copy_(torch.empty(d_model, d_model).normal_(generator=g))
                self.proj.bias.copy_(torch.empty(d_model).normal_(generator=g))

        def forward(self, hidden, *args, **kwargs):
            return (hidden + torch.tanh(self.proj(hidden)),)

    class ToyDecoder(nn.Module):
        def __init__(self, d_model=16, n_layers=3, seed=0):
            super().__init__()
            self.d_model = d_model
            # name the attribute `layers` AND nest under `.model` so attach_hooks'
            # `model.model.layers[L]` path works against the toy too.
            inner = nn.Module()
            inner.layers = nn.ModuleList([ToyDecoderLayer(d_model, seed + i) for i in range(n_layers)])
            self.model = inner

        def forward(self, hidden):
            per_layer = []
            for layer in self.model.layers:
                hidden = layer(hidden)[0]
                per_layer.append(hidden)
            return hidden, per_layer

    return torch, ToyDecoder


def test_hook_adds_alpha_v_surgically_and_removes() -> None:
    torch, ToyDecoder = _build()
    from agent.steering.hooks import attach_hooks

    model = ToyDecoder().eval()
    g = torch.Generator().manual_seed(123)
    x = torch.empty(1, 4, model.d_model).normal_(generator=g)
    with torch.no_grad():
        clean_out, clean_layers = model(x)

    L, alpha = 1, 2.5
    v = [1.0] * model.d_model
    with attach_hooks(model, v, alpha, [L]):
        with torch.no_grad():
            _, steered_layers = model(x)
    vt = torch.tensor(v)
    assert torch.allclose(steered_layers[L], clean_layers[L] + alpha * vt, atol=1e-5)
    for i in range(L):  # earlier layers unchanged
        assert torch.allclose(steered_layers[i], clean_layers[i], atol=1e-5)
    assert not torch.allclose(steered_layers[L + 1], clean_layers[L + 1], atol=1e-5)  # propagates
    # context manager removed the hook → clean forward restored
    with torch.no_grad():
        restored_out, _ = model(x)
    assert torch.allclose(restored_out, clean_out, atol=1e-5)


def test_capture_residual_on_toy() -> None:
    torch, ToyDecoder = _build()
    from agent.steering.hooks import capture_residual

    model = ToyDecoder().eval()
    g = torch.Generator().manual_seed(1)
    x1 = torch.empty(1, 4, model.d_model).normal_(generator=g)
    x2 = x1 + 1.0
    v1 = capture_residual(model, 1, lambda: model(x1))
    v2 = capture_residual(model, 1, lambda: model(x2))
    assert len(v1) == model.d_model and len(v2) == model.d_model
    assert v1 != v2                                           # input-sensitive
    assert capture_residual(model, 1, lambda: model(x1)) == v1  # deterministic


def main() -> int:
    try:
        import torch  # noqa: F401
    except Exception:
        print("SKIP test_personality_steering (torch unavailable)")
        return 0
    tests = [test_hook_adds_alpha_v_surgically_and_removes, test_capture_residual_on_toy]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} hook tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_personality_steering.py`
Expected (torch present locally): FAIL — `ModuleNotFoundError: No module named 'agent.steering.hooks'`. (If torch is somehow absent it would SKIP — but locally torch 2.4.0 is installed, so it must reach the import error.)

- [ ] **Step 3: Write the implementation** — `agent/steering/hooks.py`:

```python
"""Residual-stream steering hooks + SteeredClient (the ONLY torch module).

torch is imported lazily inside functions so importing agent.steering never
requires torch. Steering math (the vector) stays a plain list[float] until the
hook boundary, where it becomes an fp32 tensor cast to the hidden-state dtype.
"""
from __future__ import annotations

import contextlib
import os

from agent.steering.vectors import Vector

# MPS safety: opt into CPU fallback for any unimplemented op before torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def make_steering_hook(vec_f32, alpha: float):
    """register_forward_hook callback: add alpha*v to the residual-stream output.
    Handles transformers 4.x (tuple output) and 5.x (bare tensor). Casts v to the
    hidden state's dtype/device inside the hook (fp32 master → fp16 add on MPS)."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            v = vec_f32.to(device=hs.device, dtype=hs.dtype)
            return (hs + alpha * v,) + tuple(output[1:])
        hs = output
        v = vec_f32.to(device=hs.device, dtype=hs.dtype)
        return hs + alpha * v
    return hook


@contextlib.contextmanager
def attach_hooks(model, vector: Vector, alpha: float, layers: "list[int]"):
    """Register a steering hook on each model.model.layers[L]; remove all in finally."""
    import torch
    vec_f32 = torch.tensor(list(vector), dtype=torch.float32)
    handles = []
    try:
        for L in layers:
            layer = model.model.layers[L]
            handles.append(layer.register_forward_hook(make_steering_hook(vec_f32, alpha)))
        yield
    finally:
        for h in handles:
            h.remove()


def capture_residual(model, layer_idx: int, run) -> Vector:
    """Register a capturing hook on model.model.layers[layer_idx], call run() (which
    triggers a forward), and return the captured residual stream mean-pooled over all
    positions as a plain list[float]. The pure-stdlib diff_of_means/normalize then turn
    these into the steering direction (those are CI-tested in Task 1)."""
    import torch  # noqa: F401
    captured: dict = {}

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        # mean over every dim except the last (hidden) → a [hidden] vector
        captured["v"] = hs.detach().float().mean(dim=tuple(range(hs.dim() - 1))).cpu().tolist()

    h = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        run()
    finally:
        h.remove()
    return captured["v"]


def extract_persona_vector(model, tokenizer, pos_prompts, neg_prompts, layer: int,
                           *, normalize: bool = True) -> Vector:
    """CAA difference-of-means axis vector (real path): mean residual on positive
    trait prompts minus mean on negatives, at `layer`, then normalize."""
    from agent.steering.vectors import diff_of_means, normalize as _normalize

    def _vecs(prompts):
        out = []
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt").input_ids.to(model.device)
            out.append(capture_residual(model, layer, lambda ids=ids: model(ids)))
        return out

    raw = diff_of_means(_vecs(pos_prompts), _vecs(neg_prompts))
    return _normalize(raw) if normalize else raw


class _Result:
    """Duck-types agent.model.ModelResult for measure_ocean (.ok / .text)."""
    def __init__(self, text: str, ok: bool = True):
        self.text = text
        self.ok = ok


class SteeredClient:
    """In-process HF model with an active steering vector. Duck-types
    measure_ocean's client: generate(system, user) -> object with .ok/.text.
    When vector/alpha are None it is the unsteered baseline client."""

    def __init__(self, model, tokenizer, *, vector: "Vector | None" = None,
                 alpha: float = 0.0, layers: "list[int] | None" = None,
                 max_new_tokens: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.vector = vector
        self.alpha = alpha
        self.layers = layers or []
        self.max_new_tokens = max_new_tokens

    def _run(self, system: str, user: str) -> str:
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    def generate(self, system: str, user: str):
        try:
            if self.vector is not None and self.layers:
                with attach_hooks(self.model, self.vector, self.alpha, self.layers):
                    return _Result(self._run(system, user).strip())
            return _Result(self._run(system, user).strip())
        except Exception as exc:  # never crash the measurement loop
            return _Result("", ok=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_personality_steering.py`
Expected (torch present): `PASS 2 hook tests`

- [ ] **Step 5: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
git add agent/steering/hooks.py tests/test_personality_steering.py
git commit -m "feat(steering): register_forward_hook apply + SteeredClient + toy-decoder test (Spec B Task 4)"
```

---

### Task 5: Behavioral PIF channel (`personality_behavioral.py`)

**Files:**
- Create: `agent/personality_behavioral.py`
- Create: `data/behavioral_battery.json`
- Modify: `tests/test_steering.py` (append stub-judge + veneer-invariance tests)

**Interfaces:**
- Consumes: `agent.model.complete` (judges, real path), `agent.steering.stats` (cohen_d, binarize_moved, cohen_kappa). Note: does NOT consume `agent.verifiers.personality_faithful` — the behavioral channel is a pure independent LLM-judge panel; `personality_faithful` is available as a future optional pre-filter but is not wired in Spec B.
- Produces:
  - `load_battery(path=None) -> dict` — open-ended prompts per OCEAN axis.
  - `JUDGE_RUBRIC: str`, `judge_score(response: str, axis: str, *, judge_spec: str, complete_fn=complete) -> dict` — returns `{"trait_score": float, "coherence": float}` (parsed strict JSON; out-of-format → `{"trait_score": None, "coherence": 0.0}`).
  - `score_behavioral(steered_responses, neutral_responses, axis, *, judges: list[str], complete_fn=complete) -> dict` — judges both conditions, computes per-axis behavioral `trait_d`, coherence, inter-judge `kappa`; returns `{"trait_d", "coherence", "kappa", "judge_families"}`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_steering.py` (register in `main()`). Uses a deterministic stub judge so no network/Ollama is needed in CI:

```python
from agent import personality_behavioral as beh  # noqa: E402


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.personality_behavioral'`

- [ ] **Step 3: Create the battery** — `data/behavioral_battery.json` (open-ended prompts that never name the trait):

```json
{
  "instrument": "ocean-behavioral-battery-v0",
  "prompts": {
    "E": ["You arrive at a party where you know no one. Describe your next hour.",
          "Your weekend is completely free. Walk me through your ideal Saturday."],
    "O": ["You are handed a tool you have never seen. What do you do with it?",
          "Describe a place you would most want to visit and why."],
    "C": ["You have a week to finish a big project. Describe how you start.",
          "Your desk is a mess before a deadline. What happens next?"],
    "A": ["A coworker takes credit for your idea in a meeting. What do you say and do?",
          "A stranger asks you for directions while you are in a hurry."],
    "N": ["You get an unexpected, vaguely worded message from your boss. How do you feel and react?",
          "Your plans fall through at the last minute. Describe the next ten minutes."]
  }
}
```

- [ ] **Step 4: Write the implementation** — `agent/personality_behavioral.py`:

```python
"""Behavioral PIF channel — sibling to the self-report personality_measure.py (Spec A).

Open-ended, trait-name-free elicitation; an independent LLM judge panel scores
trait expression 0-100 at temp 0. NEVER reads an MBTI string (veneer-invariance):
the MBTI label is not a parameter here at all. Judges are passed as agent.model
spec strings (e.g. "ollama:qwen2.5:3b"); a complete_fn seam lets tests inject a
deterministic stub so CI needs no network.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.model import complete
from agent.steering import stats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATTERY = ROOT / "data" / "behavioral_battery.json"

_AXIS_NAME = {"O": "openness", "C": "conscientiousness", "E": "extraversion",
              "A": "agreeableness", "N": "neuroticism"}

JUDGE_RUBRIC = (
    "You are a blind personality rater. Read ONE response and rate how strongly it "
    "expresses the trait {trait} on a 0-100 scale (0 = strongly low, 100 = strongly "
    "high), and separately rate coherence 0-100. Reply with STRICT JSON only: "
    '{{"trait_score": <0-100>, "coherence": <0-100>}}'
)


def load_battery(path: "Path | None" = None) -> dict:
    return json.loads(Path(path or DEFAULT_BATTERY).read_text(encoding="utf-8"))


def _family(judge_spec: str) -> str:
    # "ollama:qwen2.5:3b" -> "qwen2.5" ; "openrouter:meta-llama/llama-3.2" -> "meta-llama"
    model = judge_spec.partition(":")[2] or judge_spec
    return model.split(":")[0].split("/")[0]


def judge_score(response: str, axis: str, *, judge_spec: str, complete_fn=complete) -> dict:
    system = JUDGE_RUBRIC.format(trait=_AXIS_NAME.get(axis, axis))
    try:
        raw = complete_fn(system, response, spec=judge_spec)
        obj = json.loads(raw)
        return {"trait_score": float(obj["trait_score"]), "coherence": float(obj["coherence"])}
    except Exception:
        return {"trait_score": None, "coherence": 0.0}


def score_behavioral(steered_responses: "list[str]", neutral_responses: "list[str]",
                     axis: str, *, judges: "list[str]", complete_fn=complete) -> dict:
    """Judge both conditions with each judge family; behavioral d from the mean
    judge trait scores (steered vs neutral); inter-judge kappa from binarized
    'moved up' labels across paired prompts."""
    per_judge_steered: dict = {}
    per_judge_neutral: dict = {}
    coher: list = []
    for spec in judges:
        fam = _family(spec)
        s = [judge_score(r, axis, judge_spec=spec, complete_fn=complete_fn) for r in steered_responses]
        n = [judge_score(r, axis, judge_spec=spec, complete_fn=complete_fn) for r in neutral_responses]
        per_judge_steered[fam] = [x["trait_score"] for x in s if x["trait_score"] is not None]
        per_judge_neutral[fam] = [x["trait_score"] for x in n if x["trait_score"] is not None]
        coher += [x["coherence"] for x in s if x["coherence"] is not None]
    fams = list(per_judge_steered)
    # behavioral effect size: pool judge means per condition
    steered_pool = [v for fam in fams for v in per_judge_steered[fam]]
    neutral_pool = [v for fam in fams for v in per_judge_neutral[fam]]
    trait_d = stats.cohen_d(steered_pool, neutral_pool)
    coherence = sum(coher) / len(coher) if coher else 0.0
    kappa = None
    if len(fams) >= 2:
        a = stats.binarize_moved(per_judge_steered[fams[0]], per_judge_neutral[fams[0]])
        b = stats.binarize_moved(per_judge_steered[fams[1]], per_judge_neutral[fams[1]])
        m = min(len(a), len(b))
        kappa = stats.cohen_kappa(a[:m], b[:m])
    return {"trait_d": trait_d, "coherence": coherence, "kappa": kappa, "judge_families": fams}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: `PASS 13 steering tests`

- [ ] **Step 6: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
git add agent/personality_behavioral.py data/behavioral_battery.json tests/test_steering.py
git commit -m "feat(steering): behavioral PIF channel (open-ended battery + judge panel + kappa) (Spec B Task 5)"
```

---

### Task 6: Contamination-free steering split (`steering_dataset.py`)

**Files:**
- Create: `provenance_bench/steering_dataset.py`
- Modify: `tests/test_steering.py` (append + register)

**Interfaces:**
- Produces:
  - `build_steering_split(*, eval_frac: float = 0.3, seed: int = 0) -> dict` — splits the IPIP items (Spec A bank) into an **extract** set (used to fit axis vectors) and a disjoint **measure** set (used to score), so a vector is never evaluated on its own fitting items. Returns `{"extract_items", "measure_items", "extract_sealed", "measure_sealed", "item_intersection"}`. `item_intersection` MUST be `[]`.
  - `_sealed(items) -> str` (16-char sha256 over sorted item ids, mirroring `rl_dataset.sealed_hash`).

- [ ] **Step 1: Write the failing test** — append to `tests/test_steering.py` (register in `main()`):

```python
from provenance_bench import steering_dataset as sds  # noqa: E402


def test_steering_split_is_contamination_free() -> None:
    split = sds.build_steering_split(eval_frac=0.4, seed=0)
    assert split["item_intersection"] == []          # no item on both sides
    ex = {it["id"] for it in split["extract_items"]}
    me = {it["id"] for it in split["measure_items"]}
    assert ex and me and ex.isdisjoint(me)
    # deterministic + drift-sealed
    again = sds.build_steering_split(eval_frac=0.4, seed=0)
    assert again["extract_sealed"] == split["extract_sealed"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'provenance_bench.steering_dataset'`

- [ ] **Step 3: Write the implementation** — `provenance_bench/steering_dataset.py`:

```python
"""Contamination-free extract/measure split of the IPIP item bank (Spec B).

Mirrors provenance_bench/rl_dataset.py's entity-disjoint split + sealed_hash, but
the unit is an IPIP item id (a vector is never measured on the items it was fit
on). Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import random

from agent.personality_measure import load_bank


def _sealed(items: list) -> str:
    payload = sorted(it["id"] for it in items)
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def build_steering_split(*, eval_frac: float = 0.3, seed: int = 0) -> dict:
    items = load_bank()["items"]
    # split per OCEAN domain so each side keeps both poles where possible
    by_dim: dict = {}
    for it in items:
        by_dim.setdefault(it["domain"], []).append(it)
    rng = random.Random(seed)
    extract, measure = [], []
    for dim in sorted(by_dim):
        group = sorted(by_dim[dim], key=lambda it: it["id"])
        rng.shuffle(group)
        n_meas = max(1, min(len(group) - 1, int(round(len(group) * eval_frac))))
        measure.extend(group[:n_meas])
        extract.extend(group[n_meas:])
    ex_ids = {it["id"] for it in extract}
    me_ids = {it["id"] for it in measure}
    return {
        "extract_items": extract,
        "measure_items": measure,
        "extract_sealed": _sealed(extract),
        "measure_sealed": _sealed(measure),
        "item_intersection": sorted(ex_ids & me_ids),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: `PASS 14 steering tests`

- [ ] **Step 5: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
git add provenance_bench/steering_dataset.py tests/test_steering.py
git commit -m "feat(steering): contamination-free extract/measure split (Spec B Task 6)"
```

---

### Task 7: `run_steering.py` CLI (twin of `run_rlvr.py`)

**Files:**
- Create: `tools/run_steering.py`
- Modify: `tests/test_steering.py` (append a CLI-offline test + register)

**Interfaces:**
- Consumes: all steering modules + `steering_dataset`.
- Produces: `python tools/run_steering.py --model mock --dry-run` runs `_offline_invariants()` through the **shipping** functions (mock extractor → compose → SSA verdict on synthetic cells → contamination-free split), writes `agi-proof/benchmark-results/steering.public-report.json`, prints `STEERING WIRING VERIFIED ✓`, returns 0. `--model phi3.5` calls `_run_real(args)` (the gated real path — Task 9 exercises it).
- `_offline_invariants() -> tuple[bool, dict]` and `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_steering.py` (register in `main()`):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.run_steering'`

- [ ] **Step 3: Write the implementation** — `tools/run_steering.py` (structural twin of `run_rlvr.py`):

```python
"""Spec B — Level-3 activation-steering runner.

OFFLINE (default, no torch/GPU/network): --model mock / --dry-run runs the
steering-machinery invariants through the shipping functions.
REAL (gated, MPS): --model phi3.5 downloads + steers microsoft/Phi-3.5-mini-instruct
and runs the Ollama-judged battery. LIVE SSA is OPEN in agi-proof/failure-ledger.md
until a gated run (entry id: steering-live-run-not-yet-gated-2026-06-23).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "steering.public-report.json"
DEFAULT_MODEL = "microsoft/Phi-3.5-mini-instruct"
FALLBACK_CHAIN = [
    "microsoft/Phi-3.5-mini-instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "ibm-granite/granite-3.1-2b-instruct", "stabilityai/stablelm-2-1_6b-chat",
]


def _offline_invariants() -> "tuple[bool, dict]":
    """Steering-machinery invariants (no torch, no GPU, no network)."""
    from agent.steering import vectors as vec
    from agent.steering import compose, stats
    from provenance_bench import steering_dataset as sds

    # mock extractor is deterministic + unit
    m1 = vec.mock_vector(3072, seed=1)
    m2 = vec.mock_vector(3072, seed=1)
    mock_det = (m1 == m2) and abs(vec.norm(m1) - 1.0) < 1e-9

    # composition with soft-projection reduces pairwise overlap vs raw
    vs = {"E": vec.normalize([1.0, 0.0]), "O": vec.normalize([1.0, 1.0])}
    raw_cos = abs(vec.cosine(vs["E"], vs["O"]))
    sp = compose.soft_project(vs)
    compose_reduces = abs(vec.cosine(sp["E"], sp["O"])) < raw_cos

    # SSA verdict enacts on a strong synthetic cell, abstains on a weak one
    strong = {"delta_ci": [0.4, 0.9], "delta_point": 0.6, "steered_d": 0.8,
              "off_target_d": {"O": 0.1}, "kappa": 0.55, "capability_drop": 0.02,
              "coherence": 90.0, "is_mock": False}
    weak = {**strong, "delta_ci": [-0.1, 0.5], "delta_point": 0.1}
    enacts = stats.ssa_verdict(strong)["status"] == "enacted"
    abstains = stats.ssa_verdict(weak)["status"] == "abstained"

    split = sds.build_steering_split(eval_frac=0.3, seed=0)

    checks = {
        "mockExtractDeterministic": mock_det,
        "composeOrthogonalReduces": compose_reduces,
        "verdictEnactsWhenStrong": enacts,
        "verdictAbstainsWhenWeak": abstains,
        "contaminationFree": split["item_intersection"] == [],
    }
    detail = {
        "checks": checks,
        "extractItems": len(split["extract_items"]),
        "measureItems": len(split["measure_items"]),
        "extractSealed": split["extract_sealed"],
        "measureSealed": split["measure_sealed"],
        "ssaThresholds": stats.SSA_THRESHOLDS,
        "fallbackChain": FALLBACK_CHAIN,
    }
    return all(checks.values()), detail


def _write_report(detail: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _run_real(args) -> int:
    """Gated real Phi-3.5 MPS run (Task 9 wires the full battery). Bails cleanly
    if torch/MPS is unavailable, mirroring run_rlvr's cuda guard."""
    try:
        import torch
    except Exception:
        print("real run needs torch: pip install -r requirements-steering.txt", file=sys.stdout)
        return 1
    if not torch.backends.mps.is_available():
        print("MPS not available; steering real run is Apple-Silicon only.", file=sys.stdout)
        return 1
    # Full real pipeline is filled in by Task 9 (load-and-smoke probe → extract →
    # steer → measure → judge → SSA). Until then, record the OPEN live claim.
    report = {
        "benchmark": "steering", "model": args.model, "visibility": "public-aggregate",
        "claimStatus": "Open — capability claim requires a gated run; "
                       "this artifact records config only",
        "ssaThresholds": __import__("agent.steering.stats", fromlist=["SSA_THRESHOLDS"]).SSA_THRESHOLDS,
        "fallbackChain": FALLBACK_CHAIN,
    }
    _write_report(report, args.out)
    print("Real steering run scaffolded. Full battery + SSA is the gated step.")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mock",
                    help=f'subject (default "mock"; real: "phi3.5" → {DEFAULT_MODEL})')
    ap.add_argument("--dry-run", action="store_true", help="offline invariants only (no torch)")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.model == "mock" or args.dry_run:
        ok, detail = _offline_invariants()
        detail["benchmark"] = "steering"
        detail["mode"] = "mock-offline"
        detail["claim"] = "steering-machinery invariants (NOT a capability claim)"
        detail["liveClaimStatus"] = (
            "Open — see agi-proof/failure-ledger.md steering-live-run-not-yet-gated-2026-06-23"
        )
        _write_report(detail, args.out)
        print("STEERING WIRING VERIFIED ✓" if ok else "STEERING INVARIANTS NOT MET ✗")
        return 0 if ok else 1

    return _run_real(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)
```

- [ ] **Step 4: Run test + the CLI to verify they pass**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tests/test_steering.py`
Expected: `PASS 16 steering tests`
Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-b && python tools/run_steering.py --model mock --dry-run`
Expected: prints `wrote .../steering.public-report.json` then `STEERING WIRING VERIFIED ✓`, exit 0.

- [ ] **Step 5: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
git add tools/run_steering.py tests/test_steering.py agi-proof/benchmark-results/steering.public-report.json
git commit -m "feat(steering): run_steering.py CLI (twin of run_rlvr) + offline invariants (Spec B Task 7)"
```

---

### Task 8: Requirements, docs, failure-ledger entry, full regression

**Files:**
- Create: `requirements-steering.txt`
- Create: `docs/09-Agent/Steering-Experiment.md`
- Modify: `agi-proof/failure-ledger.md` (append the OPEN live-run entry)

**Interfaces:** none new — packaging + docs + the pre-registered OPEN claim.

- [ ] **Step 1: Create `requirements-steering.txt`** (header mirrors `requirements-rl.txt`):

```
# Spec B activation steering. The offline `--model mock` invariants + the CI
# steering tests need NONE of these — the deterministic core is pure stdlib.
# Install only for a REAL run (in-process Phi-3.5 on Apple Silicon MPS).
torch>=2.3
transformers>=4.46.2   # Phi-3.5 LongRoPE needs a recent transformers (local env: 5.5.3)
accelerate>=1.14.0
safetensors>=0.4
# bitsandbytes 4-bit is CUDA-only; not used on MPS
bitsandbytes>=0.43; platform_system != "Darwin"
# No vLLM: incompatible with in-process forward hooks.
```

- [ ] **Step 2: Create `docs/09-Agent/Steering-Experiment.md`** (companion to `RLVR-Experiment.md`):

```markdown
# Activation-Steering Experiment (Spec B)

**The falsifiable claim, offline (CI-gated):** the steering *machinery* is
arithmetically correct — `register_forward_hook` adds `alpha·v̂` surgically,
difference-of-means recovers a planted direction, composition orthogonalizes,
and the SSA verdict abstains fail-closed. Verified by `tests/test_steering.py`
(pure stdlib) + `tests/test_personality_steering.py` (toy torch hook, skip-guarded
in CI) + `python tools/run_steering.py --model mock --dry-run`.

**The pre-registered live claim (OPEN until a gated run):** SSA — for N≥8 personas,
Level-3 steering produces a residualized OCEAN shift strictly larger than Spec A's
Level-1 persona baseline, behavior-corroborated (≥2 judge families distinct from
the subject, κ≥0.40) and capability-preserving. `SSA = 0/N` is a legitimate result.
Subject = local Phi-3.5-mini (fallback chain in `tools/run_steering.py`); judges =
local Ollama `qwen2.5:3b` + `llama3.2:3b`. Determinism is best-effort on MPS/Ollama;
only the pure scorer is bitwise-deterministic.

**Two-channel cross-validation:** the self-report channel reuses Spec A's
`measure_ocean`/`score_items`; the behavioral channel (`agent/personality_behavioral.py`)
judges open-ended, trait-name-free output. A self-report shift without a behavioral
shift → ABSTAIN ("claims, does not enact").

Run: `python tools/run_steering.py --model mock --dry-run` (offline) ·
`python tools/run_steering.py --model phi3.5` (gated real run, downloads ~7.6 GB).
```

- [ ] **Step 3: Append the failure-ledger entry** — add to `agi-proof/failure-ledger.md` (find the end of the file; append a new entry following the existing format). The entry id must match the string in `run_steering.py`:

```markdown

## steering-live-run-not-yet-gated-2026-06-23

**Status:** OPEN. The Spec B activation-steering engine is built and its machinery
invariants pass offline (`python tools/run_steering.py --model mock --dry-run` →
`STEERING WIRING VERIFIED ✓`; `tests/test_steering.py` green in CI). The live SSA
claim — that Level-3 steering beats Spec A's Level-1 persona baseline,
behavior-corroborated and capability-preserving — is **pre-registered and OPEN**:
it requires a gated real run (Phi-3.5 on MPS + the Ollama-judged battery at
N≥8/K≥20). `SSA = 0/N` would be a legitimate honest result. Thresholds are fixed
in `agent/steering/stats.py:SSA_THRESHOLDS` before any run.
```

- [ ] **Step 4: Full regression — confirm the whole CI tier is green**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
python tests/test_steering.py
python tests/test_personality_steering.py
python tests/test_personality.py
python tests/test_verifiers.py
python tools/run_steering.py --model mock --dry-run
```
Expected: `PASS 16 steering tests`; `PASS 2 hook tests` (local torch); Spec A tests still green; `STEERING WIRING VERIFIED ✓`.

- [ ] **Step 5: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
git add requirements-steering.txt docs/09-Agent/Steering-Experiment.md agi-proof/failure-ledger.md
git commit -m "feat(steering): requirements + experiment doc + OPEN live-run ledger entry (Spec B Task 8)"
```

---

### Task 9: The gated real demo run (download Phi-3.5 → one illustrative SSA run)

> **This task runs heavy, non-deterministic work on the M3 and is NOT a CI task.** It is the owner-approved "build the engine + execute ONE real demo run." It downloads ~7.6 GB (gated — confirm free disk first) and runs real MPS steering + Ollama judging on a **reduced scope** (2–3 axes, a handful of personas/seeds) to produce illustrative Δd numbers. It never makes a headline SSA claim (that needs the full N=8/K=20 run).

**Files:**
- Modify: `tools/run_steering.py` (`_run_real` → the full pipeline: preflight → load-and-smoke probe → extract → steer → measure (self-report + behavioral) → SSA Δd table → emit the leaderboard artifact + report)
- Create (generated artifact, committed): `benchmark/model_runs/local-phi3.5-steer-personality.json` + `.report.json`, and the updated `agi-proof/benchmark-results/steering.public-report.json`

**Interfaces:** consumes every prior task. Produces the illustrative Δd table.

- [ ] **Step 1: Preflight (gated) — confirm the environment before any download**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
df -h . | tail -1                                   # need ~10GB free for the model
ollama list | grep -E "qwen2.5:3b|llama3.2:3b"      # both judges present
python -c "import torch; print('mps', torch.backends.mps.is_available())"
pip install -r requirements-steering.txt            # ensure transformers/accelerate present
```
Expected: ≥10 GB free; both Ollama tags listed; `mps True`. If any fails, STOP and report (do not download).

- [ ] **Step 2: Implement the load-and-smoke probe** in `tools/run_steering.py`. Add this concrete function and call it first in `_run_real` (it walks the `FALLBACK_CHAIN` and records which model actually loaded; any failure → next entry; all fail → return None and ABSTAIN):

```python
def _load_and_smoke(seed: int = 0):
    """Try the FALLBACK_CHAIN in order; return (model, tokenizer, model_id) for the
    first that loads on MPS and passes an 8-token greedy + hidden-shape + fp32-hook
    + no-silent-CPU-fallback probe. Returns (None, None, None) if all fail."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agent.steering.hooks import capture_residual
    for model_id in FALLBACK_CHAIN:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, attn_implementation="eager",
            ).to("mps").eval()
            # 1) the model is actually on MPS (no silent CPU fallback)
            assert next(model.parameters()).device.type == "mps", "model not on mps"
            # 2) an 8-token greedy generation runs
            ids = tok("Hello.", return_tensors="pt").input_ids.to("mps")
            with torch.no_grad():
                model.generate(ids, max_new_tokens=8, do_sample=False)
            # 3) a residual capture at the target layer has the right hidden size
            L = min(21, model.config.num_hidden_layers - 1)
            v = capture_residual(model, L, lambda: model(ids))
            assert len(v) == model.config.hidden_size, "hidden-size mismatch"
            # 4) a steering vector is fp32-castable and addable at the hook
            _ = torch.tensor(v, dtype=torch.float32).to("mps")
            print(f"load-and-smoke OK: {model_id} on mps, L={L}, hidden={len(v)}")
            return model, tok, model_id
        except Exception as exc:  # try the next fallback
            print(f"load-and-smoke FAILED for {model_id}: {exc!r}")
            continue
    return None, None, None
```

Gate `_run_real` on it: `model, tok, model_id = _load_and_smoke(); if model is None: write an ABSTAIN report (reason "all fallback models failed to load on mps") and return 1.`

- [ ] **Step 3: Implement the reduced-scope real pipeline** in `_run_real`: for axes `["E", "O"]` (the two highest-confidence mappings; **skip T/F→A — pre-registered ABSTAIN**), per axis: build contrastive IPIP pairs from the `extract_items` split, `extract_persona_vector` at L=21, normalize; for each of a few personas/seeds, build a `SteeredClient` (steered, α swept over a small grid) and a Level-1 baseline client (persona prompt, no hook); run `measure_ocean` (self-report) + the behavioral battery on the `measure_items`; compute residualized `d`, `Δd = d_steer − d_baseline`, the off-target vector, behavioral `trait_d`/`kappa` via the Ollama judges (`ollama:qwen2.5:3b`, `ollama:llama3.2:3b`); apply `ssa_verdict`. Emit the leaderboard artifact (`local-phi3.5-steer-personality.json` + `.report.json`, the exact Spec A shape) and the Δd table into `steering.public-report.json` with `claimStatus: "Illustrative — reduced-scope demo; headline SSA requires the gated N=8/K=20 run"`.

- [ ] **Step 4: Execute the demo run**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
PYTORCH_ENABLE_MPS_FALLBACK=1 python tools/run_steering.py --model phi3.5
```
Expected: the probe passes (records `microsoft/Phi-3.5-mini-instruct` loaded on mps); the run prints a per-axis Δd table (steered d, baseline d, Δd, behavioral d, κ, verdict) and writes the three artifacts. Capture the actual numbers — **whatever they are** (including `SSA = 0/N` or abstains) — they are the honest illustrative result.

- [ ] **Step 5: Verify the offline tier still green + leaderboard picks up the artifact**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
python tests/test_steering.py && python tools/run_steering.py --model mock --dry-run
python tools/update_leaderboards.py && ls benchmark/results/leaderboard-personality.json
```
Expected: offline still `STEERING WIRING VERIFIED ✓`; the steered run artifact is present and leaderboards regenerate without error.

- [ ] **Step 6: Commit (engine final state + the illustrative artifact)**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-b
git add tools/run_steering.py agi-proof/benchmark-results/steering.public-report.json \
        benchmark/model_runs/local-phi3.5-steer-personality.json \
        benchmark/model_runs/local-phi3.5-steer-personality.report.json
git commit -m "feat(steering): real Phi-3.5 MPS demo run — illustrative Δd vs Level-1 baseline (Spec B Task 9)"
```

> If the real run reveals a model-load or MPS issue that the fallback chain can't resolve, STOP and report with the probe output — do not fake numbers. The engine (Tasks 1–8) stands on its own as a CI-green, mergeable deliverable even if the demo run abstains.

---

## Spec coverage check

| Spec § | Requirement | Task |
|---|---|---|
| §2 vectors.py | CAA diff-of-means extraction + mock | 1 |
| §2 hooks.py | register_forward_hook + SteeredClient duck-type | 4 |
| §2 compose.py | soft-projection + Gram-Schmidt + gram pre-flight | 2 |
| §2 personality_behavioral.py | behavioral judge channel | 5 |
| §2 run_steering.py | CLI twin of run_rlvr (offline + real) | 7, 9 |
| §2 steering_dataset.py | contamination-free split | 6 |
| §3 mechanics | fp32-on-MPS hook, eager attn, tuple/tensor output | 4, 9 |
| §4 composition | normalize + soft-proj default + interference posture | 2 |
| §5 behavioral PIF | battery, rubric, Ollama judging, Cohen d + κ | 5 |
| §7 SSA metric | the 6 cell conditions + abstain | 3 |
| §8 testing | pure-stdlib CI core + skip-guarded torch test + --model mock | 1–7 |
| §9 integration | reuse Spec A surface; requirements; leaderboard artifact | 5, 8, 9 |
| §10 risks | fallback chain + load-smoke probe + security (.env) | 9 (+ 7 chain) |
| Veneer-invariance | composition/behavioral never read MBTI string | 2 (test), 5 (test) |

**Deferred (correctly absent):** held-out sealed hidden-eval (Spec C), trained GRPO policy, human-rater judge calibration, cross-model transfer (all §12).
