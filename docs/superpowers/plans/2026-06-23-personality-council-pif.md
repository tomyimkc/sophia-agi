# Personality Council + Held-Out Anti-Gaming + Headline PIF (Spec C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, all CI-green, the three Spec C pieces — C1 a personality-diverse council A/B, C2 a held-out family + sealed eval + selfextend-style anti-gaming ABSTAIN, C3 the headline PIF/SSA harness — with the full K≥20 real run deferred and cheap/reduced real runs executed.

**Architecture:** Two-tier like `run_rlvr`/Spec B: a deterministic **pure-stdlib CI core** (the shipped contribution) + opt-in real runs. C1 reuses `council_deliberate.deliberate`'s `seat_clients` seam with persona-prefixed clients, judged by the deterministic `score_case`. C2 mirrors the selfextend false-accept→ABSTAIN contract and seals a held-out family via `hidden_eval_commitments`. C3 extends `agent/steering/stats.py` with `residualized_d`/Holm/BH and a pure `build_cells_from_scores()` seam.

**Tech Stack:** Python 3.12 (CI) / 3.10.6 (local). Pure stdlib for the CI core (`statistics`, `re`, `json`, `hashlib`, `secrets`, `random`). `torch`/`transformers` + Ollama only on the opt-in real path.

## Global Constraints

- **NO pytest.** Plain-script style A: `def test_*() -> None` + `main() -> int` running a `tests=[...]` list, printing `ok {name}` then `PASS N <suite> tests`, ending `if __name__ == "__main__": raise SystemExit(main())`; each file starts with the `ROOT`/`sys.path` guard. Wire new test files into `.github/workflows/ci.yml`.
- **Pure-stdlib CI core — NO torch/numpy** in `agent/steering/stats.py`, `agent/steering/pif_harness.py`, `provenance_bench/heldout_split.py`, `agent/council_personas.py`, or their tests. (CI has no pip step.)
- **A NULL result is a legitimate, pre-registered outcome** (C1 ΔQ and C3 SSA are both expected null). Never relax a threshold to manufacture a positive.
- **MBTI veneer-invariance** (inherited): personas are chosen from OCEAN signs; no judge/verdict/effect-size path reads an MBTI string.
- **Anti-theatre sealing:** the salt is generated once and **never committed**; only per-case sha256 hashes are public; a CI **grep-gate** forbids plaintext held-out answers in the tree/logs. The unsealed held-out pack + salt live under gitignored `private/`.
- **Anti-gaming firewall (pre-registered):** ship a steering vector only if `(fit_shift − held_shift) ≤ 0.20` AND `heldoutOffTargetRate ≤ 0.10` AND `target_moved_on_heldout`, where **`fit_shift` is measured on the fit split and `held_shift` on the disjoint held-out split** — else ABSTAIN + record. Fail-closed on missing/degenerate held-out data.
- **SSA thresholds** (reused from Spec B `SSA_THRESHOLDS`, unchanged): d>0.5, Δd≥0.30, off<0.20, κ≥0.40, capability<5%, coherence≥75. A cell counts "enacted" only if `ssa_verdict=="enacted"` AND it survives BH at q=0.05.
- **Verified reuse (do NOT fork):** `council_deliberate.deliberate(query, *, client, seat_clients=…)` (seat client called via `_gen` → `client.generate(system,user)->result(.ok,.text)`); `benchmark_checks.score_case(case, response, traditions)`, `load_json`, `DOMAIN_BENCH`, `load_traditions`; `hidden_eval_commitments.case_digest`/`build_commitments`; `selfextend.abstention_ledger.AbstentionLedger.record(*, domain, query="", reason=…)`; `agent/steering/stats.py` existing fns; `personality_measure.load_bank/measure_ocean` + `personality_behavioral.load_battery/score_behavioral` (both accept `path=`).
- Branch `feat/personality-council-pif` (worktree `/Users/tom/Documents/GitHub/sophia-agi-spec-c`, stacked on Spec B PR #66). Commit after every task.

---

### Task 1 (C1): Personality-diverse council A/B

**Files:**
- Create: `agent/council_personas.py`, `tools/run_council_diversity.py`, `tests/test_council_diversity.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `ocean_persona_prompt(name, traits) -> str`; `PersonaClient(base, persona_name, traits)` with `.generate(system,user)`, `.spec`, `.model`; `arm_passrate(domain, *, client, seat_clients=None, traditions) -> float`; `council_diversity(domain, *, client, profiles) -> dict` returning `{single, homogeneous, diverse, dq, dq_ci, profiles}`.

- [ ] **Step 1: Write the failing test** — `tests/test_council_diversity.py` (deterministic: a stub client whose `.generate` returns a canned answer that passes a tiny gold case, so the harness is exercised without a model):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-c && python tests/test_council_diversity.py`
Expected: FAIL — `No module named 'agent.council_personas'`

- [ ] **Step 3: Write `agent/council_personas.py`** (PersonaClient verbatim from the grounding; arm/diversity logic reuses `score_case` + `deliberate`):

```python
"""C1 — personality-diverse council A/B (Spec C). Pure stdlib + the council engine.

Seats carry an OCEAN persona via a prefix wrapper passed through deliberate()'s
seat_clients seam (verified: each seat is called via _gen -> client.generate
(system,user) -> result with .ok/.text). The deterministic score_case judge
cannot collude with the seats. A NULL ΔQ is the expected, honest result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_case, load_traditions
from agent.council_deliberate import deliberate
from provenance_bench.aggregate import _ci


def ocean_persona_prompt(name: str, traits: dict) -> str:
    def band(x, hi, lo):
        return hi if x >= 0.5 else lo
    o = band(traits.get("O", 0.5), "imaginative, open to unconventional framings",
             "conventional, prefers established framings")
    c = band(traits.get("C", 0.5), "methodical, detail-checking, risk-averse",
             "exploratory, big-picture, tolerant of loose ends")
    e = band(traits.get("E", 0.5), "assertive and decisive in stating a view",
             "reserved, hedged, careful to qualify")
    a = band(traits.get("A", 0.5), "cooperative, seeks common ground",
             "skeptical, willing to dissent and challenge")
    n = band(traits.get("N", 0.5), "highly alert to downside risks and failure modes",
             "calm, unbothered by tail risks")
    return (f"PERSONA ({name}): Adopt this cognitive style throughout. You are {o}; {c}; "
            f"{e}; {a}; {n}. Let this style shape WHICH considerations you surface and how "
            f"you weigh them — but never fabricate facts or citations to fit the persona.")


@dataclass
class PersonaClient:
    base: object
    persona_name: str
    traits: dict
    spec: str = field(default="", init=False)
    model: str = field(default="", init=False)

    def __post_init__(self):
        base_spec = getattr(self.base, "spec", "") or getattr(self.base, "model", "")
        self.spec = f"{base_spec}|persona:{self.persona_name}" if base_spec else f"persona:{self.persona_name}"
        self.model = self.spec

    def generate(self, system: str, user: str):
        merged = f"{ocean_persona_prompt(self.persona_name, self.traits)}\n\n{system}"
        return self.base.generate(merged, user)


def arm_passrate(domain: str, *, client, seat_clients=None, traditions=None) -> dict:
    traditions = traditions if traditions is not None else load_traditions()
    cases = load_json(DOMAIN_BENCH[domain]).get("cases", [])
    passed = 0
    per_case = []
    for case in cases:
        d = deliberate(case["question"], client=client, seat_clients=seat_clients)
        ok, _ = score_case(case, d.synthesis, traditions)
        passed += int(ok)
        per_case.append(int(ok))
    fams = sorted({getattr(c, "spec", "") for c in (seat_clients or [])}) or ["<homogeneous>"]
    return {"passrate": passed / len(cases) if cases else 0.0, "n": len(cases),
            "per_case": per_case, "seat_families": fams}


def council_diversity(domain: str, *, client, profiles: "list[tuple[str, dict]]") -> dict:
    """Three matched arms on the same gold cases:
    single (bare client), homogeneous-persona (N copies of one profile),
    diverse-persona (N distinct profiles). ΔQ = diverse − homogeneous (paired-bootstrap CI)."""
    traditions = load_traditions()
    diverse_clients = [PersonaClient(client, name, t) for name, t in profiles]
    homo_name, homo_t = profiles[0]
    homo_clients = [PersonaClient(client, homo_name, homo_t) for _ in profiles]

    single = arm_passrate(domain, client=client, seat_clients=None, traditions=traditions)
    homogeneous = arm_passrate(domain, client=client, seat_clients=homo_clients, traditions=traditions)
    diverse = arm_passrate(domain, client=client, seat_clients=diverse_clients, traditions=traditions)

    # paired ΔQ across cases: per-case (diverse_ok − homo_ok), bootstrap its mean CI
    diffs = [d - h for d, h in zip(diverse["per_case"], homogeneous["per_case"])]
    import random as _r, statistics as _s
    rng = _r.Random(0)
    boot = [_s.fmean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(2000)] if diffs else [0.0]
    dq_ci = _ci(boot)
    return {"domain": domain, "single": single, "homogeneous": homogeneous, "diverse": diverse,
            "dq": round(diverse["passrate"] - homogeneous["passrate"], 4), "dq_ci": dq_ci,
            "profiles": [n for n, _ in profiles]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-c && python tests/test_council_diversity.py`
Expected: `PASS 3 council tests`

- [ ] **Step 5: Write `tools/run_council_diversity.py`** (thin CLI; `--model mock` uses `agent.model.default_client('mock')`, real uses a local Ollama spec):

```python
"""C1 driver — personality-diverse council A/B → ΔQ. NULL ΔQ is a legitimate result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "agi-proof" / "benchmark-results" / "council-diversity.public-report.json"
# 4 distinct OCEAN profiles spanning poles (pre-registered).
PROFILES = [("O+C-", {"O": 0.9, "C": 0.1}), ("O-C+", {"O": 0.1, "C": 0.9}),
            ("E+A-", {"E": 0.9, "A": 0.1}), ("E-A+", {"E": 0.1, "A": 0.9})]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mock", help='council base client spec (mock|ollama:qwen2.5:3b|…)')
    ap.add_argument("--domain", default="philosophy")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    from agent.model import default_client
    from agent.council_personas import council_diversity
    client = default_client(args.model)
    res = council_diversity(args.domain, client=client, profiles=PROFILES)
    res["model"] = args.model
    res["headline"] = ("PASS: diversity helps" if res["dq"] > 0 and res["dq_ci"][0] > 0
                       else "NULL: trait diversity does not improve council quality on this slice")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"[{args.domain}] single={res['single']['passrate']:.3f} "
          f"homo={res['homogeneous']['passrate']:.3f} diverse={res['diverse']['passrate']:.3f} "
          f"ΔQ={res['dq']} CI={res['dq_ci']} → {res['headline']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Wire CI** — add to `.github/workflows/ci.yml` (after `python tests/test_personality_steering.py`): `python tests/test_council_diversity.py`. Verify it runs + ci.yml parses.

- [ ] **Step 7: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-c
git add agent/council_personas.py tools/run_council_diversity.py tests/test_council_diversity.py .github/workflows/ci.yml
git commit -m "feat(council): personality-diverse council A/B + ΔQ (Spec C C1)"
```

---

### Task 2 (C3): Stats helpers — residualized-d, Holm, BH, bootstrap-p

**Files:**
- Modify: `agent/steering/stats.py` (add four pure-stdlib helpers + `_solve_linear`)
- Create: `tests/test_pif_harness.py` (the C3/C2 CI suite scaffold)

**Interfaces:**
- Produces: `residualized_d(target_per_seed, offtarget_per_seed_by_axis) -> float`; `holm_bonferroni(pvalues) -> list[float]`; `benjamini_hochberg(pvalues, q) -> list[bool]`; `bootstrap_diff_p(steer, base, *, n_boot=2000, seed=0) -> float`; `_solve_linear(A, c) -> list[float] | None`.

- [ ] **Step 1: Write the failing test** — `tests/test_pif_harness.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-c && python tests/test_pif_harness.py`
Expected: FAIL — `module 'agent.steering.stats' has no attribute 'holm_bonferroni'`

- [ ] **Step 3: Append the four helpers to `agent/steering/stats.py`** (verbatim from the grounding — `residualized_d`, `_solve_linear`, `holm_bonferroni`, `benjamini_hochberg`, `bootstrap_diff_p`). [Full code block — copy the five functions exactly as specified in the design grounding; they use only the already-imported `statistics` and `random`.]

```python
def residualized_d(target_per_seed, offtarget_per_seed_by_axis):
    y = list(target_per_seed); n = len(y)
    if n < 2:
        return 0.0
    axes = [k for k, v in offtarget_per_seed_by_axis.items() if len(v) == n]
    X = [[1.0] + [offtarget_per_seed_by_axis[k][i] for k in axes] for i in range(n)]
    p = 1 + len(axes)
    A = [[sum(X[i][r] * X[i][s] for i in range(n)) for s in range(p)] for r in range(p)]
    c = [sum(X[i][r] * y[i] for i in range(n)) for r in range(p)]
    beta = _solve_linear(A, c)
    if beta is None:
        sd = statistics.pstdev(y) if n > 1 else 0.0
        return 0.0 if sd == 0.0 else statistics.fmean(y) / sd
    resid = [y[i] - sum(beta[r] * X[i][r] for r in range(1, p)) for i in range(n)]
    sd = statistics.pstdev(resid)
    return 0.0 if sd == 0.0 else statistics.fmean(resid) / sd


def _solve_linear(A, c):
    n = len(A)
    M = [list(A[i]) + [c[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]; M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]; M[r] = [M[r][k] - f * M[col][k] for k in range(n + 1)]
    return [M[i][n] for i in range(n)]


def holm_bonferroni(pvalues):
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m; running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adj[idx] = min(1.0, running)
    return adj


def benjamini_hochberg(pvalues, q):
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    k_max = 0
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= (rank / m) * q:
            k_max = rank
    sig = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= k_max:
            sig[idx] = True
    return sig


def bootstrap_diff_p(steer, base, *, n_boot=2000, seed=0):
    diffs = [s - b for s, b in zip(steer, base)]; n = len(diffs)
    if n == 0:
        return 1.0
    rng = random.Random(seed)
    boot = [statistics.fmean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)]
    frac_le = sum(1 for m in boot if m <= 0.0) / n_boot
    return min(1.0, max(1.0 / n_boot, 2.0 * min(frac_le, 1.0 - frac_le)))
```

- [ ] **Step 4: Run test + confirm Spec B stats untouched**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-c && python tests/test_pif_harness.py && python tests/test_steering.py`
Expected: `PASS 4 pif tests` and `PASS 17 steering tests` (B's existing stats tests still pass).

- [ ] **Step 5: Wire CI** — add `python tests/test_pif_harness.py` after `python tests/test_council_diversity.py` in ci.yml.

- [ ] **Step 6: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-c
git add agent/steering/stats.py tests/test_pif_harness.py .github/workflows/ci.yml
git commit -m "feat(pif): residualized-d + Holm/BH + bootstrap-p stats helpers (Spec C C3)"
```

---

### Task 3 (C3): PIF harness `build_cells_from_scores` + `run_pif --dry-run`

**Files:**
- Create: `agent/steering/pif_harness.py`, `tools/run_pif.py`
- Modify: `tests/test_pif_harness.py` (append cell-assembly tests + register)

**Interfaces:**
- Consumes: `stats` (Task 2 + Spec B).
- Produces: `build_cells_from_scores(scores: dict, grid: list[dict]) -> list[dict]` (each cell has the `ssa_verdict` keys + `cell_id`, `p_raw`, `verdict`); `headline(cells, *, q=0.05) -> dict` (attaches BH significance grid-wide, returns `enacted/total`); `synthetic_scores(kind)` test fixtures.
- `run_pif.main(["--dry-run"]) -> 0` writes `agi-proof/benchmark-results/pif.public-report.json`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_pif_harness.py` (register in `main()`):

```python
from agent.steering import pif_harness as pif  # noqa: E402


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
    off = {ax: [0.02 * rng.gauss(0, 1) for _ in range(K)] for ax in ("O", "C", "A")}
    s = {"E": {"steer": steer, "base": base, "neutral": neutral}}
    for ax, arr in off.items():
        s[ax] = {"steer": arr, "base": arr, "neutral": neutral}
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
    assert h["total"] == 2 and 0 <= h["enacted"] <= 2
    assert "enacted_over_total" in h


def test_is_mock_forces_abstain() -> None:
    grid = [{"cell_id": "c1", "target_axis": "E", "off_target_axes": ["O"], "is_mock": True, "seed": 1}]
    cells = pif.build_cells_from_scores({"c1": _planted("strong")}, grid)
    assert cells[0]["verdict"]["reason"] == "mock_subject"
```

- [ ] **Step 2: Run to verify it fails** — Expected: `No module named 'agent.steering.pif_harness'`.

- [ ] **Step 3: Write `agent/steering/pif_harness.py`** (uses `build_cells_from_scores` from the grounding + a `headline` that applies BH grid-wide):

```python
"""C3 — PIF/SSA headline harness (pure stdlib). The build_cells_from_scores seam
takes pre-computed per-seed score arrays → fully CI-testable, no model. A near-null
headline confirming Spec B 0/2 is the expected, pre-registered result."""
from __future__ import annotations

import statistics

from agent.steering.stats import (bootstrap_diff_ci, bootstrap_diff_p, benjamini_hochberg,
                                   cohen_d, residualized_d, ssa_verdict)


def build_cells_from_scores(scores: dict, grid: "list[dict]") -> "list[dict]":
    cells = []
    for g in grid:
        cid, tgt = g["cell_id"], g["target_axis"]
        s = scores[cid]
        steer_t, base_t = s[tgt]["steer"], s[tgt]["base"]
        seed = g.get("seed", 0)
        delta_ci = bootstrap_diff_ci(steer_t, base_t, seed=seed)
        delta_point = statistics.fmean([a - b for a, b in zip(steer_t, base_t)]) if steer_t else 0.0
        p_raw = bootstrap_diff_p(steer_t, base_t, seed=seed)
        steered_d = abs(residualized_d(steer_t, {ax: s[ax]["steer"] for ax in g["off_target_axes"]}))
        off_target_d = {ax: cohen_d(s[ax]["steer"], s[ax]["neutral"]) for ax in g["off_target_axes"]}
        cell = {"cell_id": cid, "delta_ci": delta_ci, "delta_point": delta_point,
                "steered_d": steered_d, "off_target_d": off_target_d, "kappa": s["kappa"],
                "capability_drop": s["capability_drop"], "coherence": s["coherence"],
                "is_mock": g["is_mock"], "p_raw": p_raw}
        cell["verdict"] = ssa_verdict({k: cell[k] for k in (
            "delta_ci", "delta_point", "steered_d", "off_target_d",
            "kappa", "capability_drop", "coherence", "is_mock")})
        cells.append(cell)
    return cells


def headline(cells: "list[dict]", *, q: float = 0.05) -> dict:
    """A cell counts 'enacted' only if ssa_verdict=='enacted' AND survives BH at q."""
    pvals = [c.get("p_raw", 1.0) for c in cells]
    sig = benjamini_hochberg(pvals, q)
    enacted = sum(1 for c, ok in zip(cells, sig)
                  if c["verdict"]["status"] == "enacted" and ok)
    total = len(cells)
    return {"enacted": enacted, "total": total,
            "enacted_over_total": f"{enacted}/{total}",
            "bh_significant": sig}
```

- [ ] **Step 4: Write `tools/run_pif.py`** (two-tier twin of `run_steering.py`; `--dry-run`/`--model mock` feeds synthetic cells through the shipping functions; `--model <hf id>` is the deferred heavy path that records the OPEN claim):

```python
"""C3 driver — headline-grade PIF/SSA. --dry-run/--model mock = CI core (synthetic
cells through the shipping harness). --model <hf id> = opt-in heavy run (DEFERRED:
records the OPEN live claim; full N>=8/K>=20 on a downloaded model + held-out family)."""
from __future__ import annotations

import argparse, json, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "agi-proof" / "benchmark-results" / "pif.public-report.json"


def _offline_invariants() -> "tuple[bool, dict]":
    from agent.steering import pif_harness as pif
    import random
    rng = random.Random(2); K = 24
    def cell_scores(strong):
        steer = [(1.0 if strong else 0.2) + 0.05 * rng.gauss(0, 1) for _ in range(K)]
        base = [(0.1 if strong else 0.2) + 0.05 * rng.gauss(0, 1) for _ in range(K)]
        neu = [0.0] * K
        s = {"E": {"steer": steer, "base": base, "neutral": neu}}
        for ax in ("O", "C", "A"):
            s[ax] = {"steer": [0.02 * rng.gauss(0, 1) for _ in range(K)], "base": [0.0] * K, "neutral": neu}
        s["kappa"], s["coherence"], s["capability_drop"] = 0.6, 90.0, 0.02
        return s
    grid = [{"cell_id": "strong", "target_axis": "E", "off_target_axes": ["O", "C", "A"], "is_mock": False, "seed": 1},
            {"cell_id": "null", "target_axis": "E", "off_target_axes": ["O", "C", "A"], "is_mock": False, "seed": 2}]
    cells = pif.build_cells_from_scores({"strong": cell_scores(True), "null": cell_scores(False)}, grid)
    h = pif.headline(cells)
    checks = {"strongEnacts": cells[0]["verdict"]["status"] == "enacted",
              "nullAbstains": cells[1]["verdict"]["status"] == "abstained",
              "headlineCounts": h["total"] == 2}
    return all(checks.values()), {"checks": checks, "headline": h,
                                  "cells": [{"cell_id": c["cell_id"], "verdict": c["verdict"]["status"],
                                             "reason": c["verdict"]["reason"]} for c in cells]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.model == "mock" or args.dry_run:
        ok, detail = _offline_invariants()
        detail.update(benchmark="pif", mode="mock-offline",
                      liveClaimStatus="Open — see agi-proof/failure-ledger.md pif-headline-run-not-yet-gated-2026-06-23")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("PIF HARNESS VERIFIED ✓" if ok else "PIF HARNESS NOT MET ✗")
        return 0 if ok else 1
    print("Full N>=8/K>=20 PIF run is DEFERRED (OPEN in the ledger). Build CI-green; "
          "trigger only on a non-null reduced-slice trend.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)
```

- [ ] **Step 5: Run tests + CLI**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-c && python tests/test_pif_harness.py && python tools/run_pif.py --dry-run`
Expected: `PASS 7 pif tests`; `PIF HARNESS VERIFIED ✓`.

- [ ] **Step 6: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-c
git add agent/steering/pif_harness.py tools/run_pif.py tests/test_pif_harness.py agi-proof/benchmark-results/pif.public-report.json
git commit -m "feat(pif): build_cells_from_scores harness + run_pif --dry-run CI core (Spec C C3)"
```

---

### Task 4 (C2): Held-out family + `held_out_disjoint`

**Files:**
- Create: `data/personality_items_heldout.json`, `data/behavioral_battery_heldout.json`, `provenance_bench/heldout_split.py`
- Modify: `tests/test_pif_harness.py` (append disjointness tests)

**Interfaces:**
- Produces: `held_out_disjoint(seen_items_path=None, heldout_items_path=…, seen_battery_path=None, heldout_battery_path=…, fit_module="tools/run_steering.py") -> dict` returning `{ipip_intersection, ngram_overlaps, seen_sealed, heldout_sealed, fit_reads_heldout, nearest_neighbour_sim}`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_pif_harness.py`:

```python
from provenance_bench import heldout_split as hos  # noqa: E402


def test_held_out_disjoint() -> None:
    r = hos.held_out_disjoint()
    assert r["ipip_intersection"] == []            # no shared item ids
    assert r["ngram_overlaps"] == []               # no shared content 3-gram
    assert r["fit_reads_heldout"] is False         # fit module never imports held-out paths
    assert r["seen_sealed"] != r["heldout_sealed"]
    assert r["nearest_neighbour_sim"] < 0.5        # construct-disjoint, not paraphrase
```

- [ ] **Step 2: Run to verify it fails** — Expected: `No module named 'provenance_bench.heldout_split'`.

- [ ] **Step 3: Author the held-out data files.** `data/personality_items_heldout.json` — instrument `ipip-spec-c-heldout`, ≥4 markers/domain (≥2/pole), `ho_*` ids, **zero lexical overlap** with the seen 10 (seen: "have a vivid imagination", "get chores done right away", "am the life of the party", "sympathize with others' feelings", "get stressed out easily", and their reverses). Use distinct public-IPIP markers, e.g. O: "love to think up new ways of doing things" (+), "avoid philosophical discussions" (−); C: "pay attention to details" (+), "leave my belongings around" (−); E: "feel comfortable around people" (+), "keep in the background" (−); A: "take time out for others" (+), "insult people" (−); N: "worry about things" (+), "seldom feel blue" (−). Schema identical to `data/personality_items.json`. `data/behavioral_battery_heldout.json` — instrument `ocean-behavioral-battery-heldout-v0`, situationally-disjoint trait-name-free prompts (seen E uses a party; held-out E must use a *different* situation, e.g. "Your team just hit a deadline early — what do you do with the spare afternoon?").

- [ ] **Step 4: Write `provenance_bench/heldout_split.py`** (pure stdlib; mirrors `steering_dataset._sealed`):

```python
"""C2 — held-out family disjointness + sealing (pure stdlib). Construct-disjoint,
not just string-disjoint: hold out whole clusters, assert no shared content 3-gram,
and that the fit module never imports the held-out paths."""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

from agent.personality_measure import load_bank
from agent.personality_behavioral import load_battery

ROOT = Path(__file__).resolve().parents[1]
SEEN_ITEMS = ROOT / "data" / "personality_items.json"
HELDOUT_ITEMS = ROOT / "data" / "personality_items_heldout.json"
SEEN_BATTERY = ROOT / "data" / "behavioral_battery.json"
HELDOUT_BATTERY = ROOT / "data" / "behavioral_battery_heldout.json"
_STOP = {"the", "a", "an", "to", "of", "and", "i", "you", "your", "is", "are", "with", "for", "in", "on", "do"}


def _sealed(strings: "list[str]") -> str:
    return hashlib.sha256(json.dumps(sorted(strings)).encode()).hexdigest()[:16]


def _tokens(text: str) -> "list[str]":
    return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOP]


def _ngrams(texts: "list[str]", n=3) -> set:
    out = set()
    for t in texts:
        toks = _tokens(t)
        out |= {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}
    return out


def held_out_disjoint(*, fit_module="tools/run_steering.py") -> dict:
    seen_items = [it["text"] for it in load_bank(SEEN_ITEMS)["items"]]
    ho_items = [it["text"] for it in load_bank(HELDOUT_ITEMS)["items"]]
    seen_ids = {it["id"] for it in load_bank(SEEN_ITEMS)["items"]}
    ho_ids = {it["id"] for it in load_bank(HELDOUT_ITEMS)["items"]}
    seen_b = [p for v in load_battery(SEEN_BATTERY)["prompts"].values() for p in v]
    ho_b = [p for v in load_battery(HELDOUT_BATTERY)["prompts"].values() for p in v]
    overlaps = sorted(" ".join(g) for g in (_ngrams(seen_items + seen_b) & _ngrams(ho_items + ho_b)))
    # nearest-neighbour token-Jaccard between any seen and any held-out text
    def jac(a, b):
        sa, sb = set(_tokens(a)), set(_tokens(b))
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    nn = max((jac(s, h) for s in seen_items + seen_b for h in ho_items + ho_b), default=0.0)
    fit_src = (ROOT / fit_module).read_text(encoding="utf-8")
    fit_reads = ("personality_items_heldout" in fit_src) or ("behavioral_battery_heldout" in fit_src)
    return {"ipip_intersection": sorted(seen_ids & ho_ids), "ngram_overlaps": overlaps,
            "seen_sealed": _sealed(seen_items), "heldout_sealed": _sealed(ho_items),
            "fit_reads_heldout": fit_reads, "nearest_neighbour_sim": round(nn, 4)}
```

- [ ] **Step 5: Run test** — Expected: `PASS 8 pif tests`. (If `ngram_overlaps` or `nearest_neighbour_sim` fail, the held-out items are too close — revise them to be construct-disjoint, not paraphrases.)

- [ ] **Step 6: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-c
git add data/personality_items_heldout.json data/behavioral_battery_heldout.json provenance_bench/heldout_split.py tests/test_pif_harness.py
git commit -m "feat(heldout): construct-disjoint held-out family + held_out_disjoint check (Spec C C2)"
```

---

### Task 5 (C2): Sealing the held-out pack + anti-theatre grep-gate

**Files:**
- Create: `tools/seal_personality_heldout.py`, `private/hidden-evals/.gitkeep` (gitignored dir marker), `agi-proof/hidden-reviewer-packs/personality-heldout-2026-06-23.commitments.json`
- Modify: `.gitignore` (add `private/`), `tests/test_pif_harness.py` (append sealing + grep-gate tests)

**Interfaces:**
- Produces: `seal_heldout(salt=None) -> dict` (builds the private pack from the held-out data, returns public commitments; writes the unsealed pack + salt under `private/`); a CI grep-gate assertion that no plaintext held-out answer is in the public tree.

- [ ] **Step 1: Write the failing test** — append to `tests/test_pif_harness.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails** — `No module named 'tools.seal_personality_heldout'`.

- [ ] **Step 3: Write `tools/seal_personality_heldout.py`**:

```python
"""Seal the held-out personality family → public SHA-256 commitments only.
The salt + unsealed prompts are written under gitignored private/; only the
per-case hashes are published. Reuses tools/hidden_eval_commitments.py."""
from __future__ import annotations

import argparse, json, secrets, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.personality_measure import load_bank          # noqa: E402
from agent.personality_behavioral import load_battery     # noqa: E402
from tools.hidden_eval_commitments import build_commitments  # noqa: E402

PACK_ID = "personality-heldout-2026-06-23"
COMMIT_OUT = ROOT / "agi-proof" / "hidden-reviewer-packs" / f"{PACK_ID}.commitments.json"
PRIVATE_OUT = ROOT / "private" / "hidden-evals" / f"{PACK_ID}.private.json"
HELDOUT_ITEMS = ROOT / "data" / "personality_items_heldout.json"
HELDOUT_BATTERY = ROOT / "data" / "behavioral_battery_heldout.json"


def build_private_pack(salt: "str | None" = None) -> dict:
    salt = salt or secrets.token_hex(32)
    cases = []
    for it in load_bank(HELDOUT_ITEMS)["items"]:
        cases.append({"id": f"ipip_{it['id']}", "domain": "personality",
                      "prompt": f"I {it['text']}.",
                      "scoring": {"channel": "self-report", "domain": it["domain"], "keyed": it["keyed"],
                                  "method": "ipip-likert-1-5"},
                      "requiresToolLog": False, "requiresMemoryDiff": False})
    for axis, prompts in load_battery(HELDOUT_BATTERY)["prompts"].items():
        for i, p in enumerate(prompts):
            cases.append({"id": f"batt_{axis}_{i}", "domain": "personality", "prompt": p,
                          "scoring": {"channel": "behavioral", "axis": axis, "method": "ocean-judge-0-100"},
                          "requiresToolLog": False, "requiresMemoryDiff": False})
    return {"packId": PACK_ID, "createdAt": datetime.now(timezone.utc).isoformat(),
            "salt": salt, "cases": cases}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--salt", default=None, help="hex salt (default: fresh random 256-bit)")
    args = ap.parse_args(argv)
    pack = build_private_pack(args.salt)
    commitments = build_commitments(pack)
    COMMIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    COMMIT_OUT.write_text(json.dumps(commitments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    PRIVATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUT.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote public commitments {COMMIT_OUT} ({len(commitments['cases'])} cases)")
    print(f"wrote PRIVATE pack+salt {PRIVATE_OUT} (gitignored — store the salt securely)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`build_private_pack` returns the pack dict (so the test re-verifies each digest); `main()` writes the public commitments + the gitignored private pack.

- [ ] **Step 4: Add `private/` to `.gitignore`**, generate the commitments (`python tools/seal_personality_heldout.py` with a generated salt), and confirm the public file has hashes only.

- [ ] **Step 5: Run tests** — Expected: `PASS 10 pif tests`. Confirm `git status` does NOT show `private/` contents.

- [ ] **Step 6: Commit** (the public commitments + the tool + .gitignore; NOT the private pack/salt)

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-c
git add tools/seal_personality_heldout.py .gitignore agi-proof/hidden-reviewer-packs/personality-heldout-2026-06-23.commitments.json tests/test_pif_harness.py
git commit -m "feat(seal): salted SHA-256 commitments for the held-out personality pack + grep-gate (Spec C C2)"
```

---

### Task 6 (C2): Anti-gaming ship-or-ABSTAIN mirror

**Files:**
- Create: `agent/steering/anti_gaming.py`
- Modify: `tests/test_pif_harness.py` (append ship/abstain tests)

**Interfaces:**
- Consumes: `selfextend.abstention_ledger.AbstentionLedger`.
- Produces: `ship_steering(*, fit_shift, held_shift, heldout_off_target_rate, target_moved_on_heldout, axis, ledger=None) -> dict` returning `{ship, invariants, reason}`; mirrors the selfextend false-accept contract, fail-closed.

- [ ] **Step 1: Write the failing test** — append to `tests/test_pif_harness.py`:

```python
from agent.steering import anti_gaming as ag  # noqa: E402


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
```

- [ ] **Step 2: Run to verify it fails** — `No module named 'agent.steering.anti_gaming'`.

- [ ] **Step 3: Write `agent/steering/anti_gaming.py`** (the selfextend mirror, verbatim contract):

```python
"""C2 — ship-steering-or-ABSTAIN, mirroring the selfextend false-accept contract.
Ship a vector only if all three falsifiable invariants hold on the HELD-OUT split;
else ABSTAIN + record. fail-closed. An unvalidated steering vector is worse than none."""
from __future__ import annotations

GAP = 0.20            # mirrors verified_reward.reward_is_hackable gap
OFF_TARGET_MAX = 0.10  # mirrors flywheel heldoutFalseAcceptRate cap


def ship_steering(*, fit_shift: float, held_shift: float, heldout_off_target_rate: float,
                  target_moved_on_heldout: bool, axis: str, ledger=None) -> dict:
    invariants = {
        "not_gamed": (fit_shift - held_shift) <= GAP,
        "off_target_bounded": heldout_off_target_rate <= OFF_TARGET_MAX,
        "target_moved": bool(target_moved_on_heldout),
    }
    ship = all(invariants.values())
    reason = None
    if not ship:
        # reported reason: first failing invariant in this fixed order
        if not invariants["target_moved"]:
            reason = "target_not_moved"
        elif not invariants["not_gamed"]:
            reason = "steering_gamed"
        else:
            reason = "steering_off_target"
        if ledger is not None:
            ledger.record(domain=axis, reason=reason)
    return {"ship": ship, "invariants": invariants, "reason": reason}
```

> Reason precedence note: the test sets exactly one invariant false at a time. `target_moved` is checked first so a fail-closed missing-move is reported as `target_not_moved`; the gamed case (`not_gamed` false) reports `steering_gamed`; off-target reports `steering_off_target`.

- [ ] **Step 4: Run test** — Expected: `PASS 11 pif tests`.

- [ ] **Step 5: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-c
git add agent/steering/anti_gaming.py tests/test_pif_harness.py
git commit -m "feat(anti-gaming): ship-or-ABSTAIN held-out firewall (selfextend mirror) (Spec C C2)"
```

---

### Task 7: Wiring, docs, ledger, full regression

**Files:**
- Modify: `.github/workflows/ci.yml` (ensure all new test files wired)
- Create: `docs/09-Agent/Council-and-PIF-Experiment.md`, a new `agi-proof/failure-ledger.md` entry `pif-headline-run-not-yet-gated-2026-06-23`

- [ ] **Step 1: Ensure `ci.yml`** runs `python tests/test_council_diversity.py` and `python tests/test_pif_harness.py` (added in Tasks 1–2). Verify it parses.
- [ ] **Step 2: Write `docs/09-Agent/Council-and-PIF-Experiment.md`** — the C1 ΔQ claim (null OK), the C2 sealed-held-out + anti-gaming contract, the C3 harness (CI-green) + the OPEN headline run; two-tier discipline; "SSA=0/N and ΔQ≤0 are legitimate results."
- [ ] **Step 3: Append the ledger entry** `pif-headline-run-not-yet-gated-2026-06-23` to `agi-proof/failure-ledger.md` (OPEN; references `agent/steering/stats.py:SSA_THRESHOLDS` + the pre-registered grid; matches the existing entry format).
- [ ] **Step 4: Full regression**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-c
python tests/test_council_diversity.py && python tests/test_pif_harness.py && \
python tests/test_steering.py && python tests/test_personality.py && python tests/test_verifiers.py && \
python tools/run_pif.py --dry-run && python tools/run_council_diversity.py --model mock --domain philosophy
```
Expected: all PASS; `PIF HARNESS VERIFIED ✓`; the council mock A/B writes a report (ΔQ likely 0 with mock text — that's fine for the plumbing check).

- [ ] **Step 5: Commit**

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-c
git add .github/workflows/ci.yml docs/09-Agent/Council-and-PIF-Experiment.md agi-proof/failure-ledger.md agi-proof/benchmark-results/council-diversity.public-report.json
git commit -m "feat(spec-c): wiring + experiment doc + OPEN headline-PIF ledger entry (Spec C)"
```

---

### Task 8 (controller-driven): reduced real runs

> Heavy-ish, non-deterministic, NOT a CI task. Run the cheap council A/B on a local model and a reduced PIF/anti-gaming slice; commit the honest artifacts (expected: ΔQ≤0 and a near-null/abstain SSA).

- [ ] **Step 1: Council A/B for real** — `python tools/run_council_diversity.py --model ollama:qwen2.5:3b --domain philosophy` (cheap local 3B base client; deterministic `score_case` judge). Capture `single/homo/diverse/ΔQ/CI`. Repeat on `--domain personality` if time permits.
- [ ] **Step 2: Reduced PIF + anti-gaming slice** — a small driver run (or extend `run_pif.py --model granite` reduced) that: fits a CAA vector on SEEN carriers, FREEZES alpha, runs `held_out_disjoint()` gate, administers a reduced slice (2 axes, K~3) on the **held-out** IPIP + battery, computes `fit_shift` (on seen) vs `held_shift` (on held-out), calls `ship_steering(...)`, and assembles a few real cells through `build_cells_from_scores` + `headline`. Expected: ABSTAIN / near-null.
- [ ] **Step 3: Commit the artifacts** (council-diversity report with real ΔQ + the reduced PIF report) with an honest message capturing the null/abstain results. Verify the offline tier is still green.

---

## Spec coverage check

| Spec piece | Requirement | Task |
|---|---|---|
| C1 | personality-diverse council A/B, ΔQ + CI, deterministic judge | 1, 8 |
| C2 | held-out construct-disjoint family + `held_out_disjoint` | 4 |
| C2 | sealed commitments + anti-theatre grep-gate | 5 |
| C2 | ship-or-ABSTAIN selfextend mirror | 6, 8 |
| C3 | residualized-d + Holm/BH + bootstrap-p | 2 |
| C3 | `build_cells_from_scores` + headline + run_pif CI core | 3, 8 |
| both | two-tier (CI core + opt-in/deferred real); null OK | all |
| cross | ci.yml wiring, docs, OPEN ledger entry | 7 |

**Deferred to Spec D** (per spec §12): the full N≥8/K≥20 real headline run, validated Level-3 steered council seats, true external sealing, model×trait crossover, live GRPO, calibration tracking, capability-retention product gate + full FastMCP packaging.
