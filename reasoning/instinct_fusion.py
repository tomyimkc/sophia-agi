# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Reflex fusion — does a 2nd *independent* detector push the instinct over the bar?

``instinct_reflex_eval`` measured the first reflex, **self-consistency disagreement**, and
found it *borderline* (d′≈0.96, just under the break-even d′=1.0 that ``instinct_gate``
requires). The architecture note's answer was: add a second, *independent* detector to the
reflex bus. This module tests that — for real — on the belief-revision oracle, and pins the
detection-theory law that says when it works.

Two detectors, by construction keyed on *different* failure modes (this is what makes them
independent, and independence is the whole game):

  - **A — self-consistency disagreement** (``agent.calibration.self_consistency``,
    label-free): fires when the model is *uncertain* (its samples scatter). Its blind spot
    is the **confident error** — the model agreeing on the same wrong answer.
  - **B — okf grounding-closure violation** (real ``okf.revise`` / ``claims_to_abstain``):
    fires when the proposed abstain set wrongly includes a claim that *still has live
    grounding* (over-abstention), checked structurally against the belief graph. Its blind
    spot is **under-abstention** (missing an orphaned claim) — which it cannot see, so it is
    a genuinely *partial* detector, not the oracle. It catches the confident, structurally
    invalid answers A misses.

Result it tests (all falsifiable, all in ``--self-test`` / the test module):

  U1  Each detector alone is borderline (AUC well under a perfect 1.0; d′ near or below the
      bar), but their **fusion clears d′ = 1.0** — two weak reflexes make one good one.
  U2  Fusion follows detection theory: for *independent* detectors d′ adds in quadrature,
      ``d′_fused ≈ √(d′_A² + d′_B²)``; the gain **vanishes as the detectors become
      redundant** (correlation ρ→1, ``d′_fused → max``). Verified against a Gaussian control
      with closed form ``d′_fused = (d_A+d_B)/√(2+2ρ)``.
  U3  Complementarity is real: B's separation is concentrated on exactly the errors A misses.
  U4  ``d′_fused > max(d′_A, d′_B)`` at the measured (low) correlation.

Honest scope (``candidateOnly: true``, ``canClaimAGI: false``). Detector B is real okf;
detector A is real self-consistency; but the *answer distribution* is a seeded synthetic
reasoner (planted error regimes over the real graphs) so the model is falsifiable and runs
offline. The measured d′s are the synthetic reasoner's, present to validate the fusion law
and the bus design — not a claim about any real model. The real-model fusion d′ is a gated
next step (``tools/run_reflex_openrouter.py`` + a 2nd-detector wiring), not asserted here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from okf.page import Page  # noqa: E402
from okf.revision import claims_to_abstain  # noqa: E402

from agent.calibration import self_consistency  # noqa: E402
from reasoning.instinct_gate import run_experiment as _gate_experiment  # noqa: E402
from reasoning.instinct_reflex_eval import DIFFICULTY, DEFAULT_DIFFICULTY, auc, d_prime, load_cases  # noqa: E402

N_SAMPLES = 7
#: Per-sample mutation rate (real sampling noise): even a "confident" reasoner occasionally
#: emits a divergent sample, and a clean answer occasionally wobbles. Without this the
#: planted regimes are noiseless and the fused detector separates *perfectly* (AUC 1.0, d′
#: blows up on near-zero variance) — a "too clean" artifact. Noise keeps it honest.
P_SAMPLE_NOISE = 0.18
#: Error-type mix once a case errors. The two detectors split the first two; the third is a
#: confident *under*-abstention that BOTH are blind to — the honest residual fusion can't fix
#: (keeps each detector borderline and the fused AUC below a suspicious 1.0).
ERR_MIX = {"uncertain_under": 0.40, "confident_over": 0.40, "hard_both_miss": 0.20}


def breakeven_snr() -> float:
    return _gate_experiment(trials=2000, seed=1234)["verdict"]["h3_breakeven_snr"]


# ---------------------------------------------------------------------------
# Real okf belief graph per case (mirrors tools/run_belief_revision_benchmark)
# ---------------------------------------------------------------------------

def _graph_for(idx: int):
    pages = [
        Page(path=Path(f"primary_{idx}.md"), meta={"id": f"primary_{idx}", "pageType": "concept", "authorConfidence": "consensus"}),
        Page(path=Path(f"independent_{idx}.md"), meta={"id": f"independent_{idx}", "pageType": "concept", "authorConfidence": "attributed"}),
        Page(path=Path(f"mid_{idx}.md"), meta={"id": f"mid_{idx}", "pageType": "concept", "derivesFrom": [f"primary_{idx}"]}),
        Page(path=Path(f"leaf_{idx}.md"), meta={"id": f"leaf_{idx}", "pageType": "concept", "derivesFrom": [f"mid_{idx}"]}),
        Page(path=Path(f"multi_{idx}.md"), meta={"id": f"multi_{idx}", "pageType": "concept", "derivesFrom": [f"primary_{idx}", f"independent_{idx}"]}),
    ]
    return okf.build_graph(pages)


def _idx(case_id: str) -> int:
    return int(case_id.rsplit("_", 1)[-1])


def _true_abstain(case: dict[str, Any]) -> frozenset[str]:
    """The structurally-correct abstain set, via the REAL okf.revise machinery."""
    g = _graph_for(_idx(case["id"]))
    return frozenset(claims_to_abstain(g, case.get("remove", [])))


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def _reflex_A(samples: list[frozenset[str]]) -> float:
    """Self-consistency disagreement over the proposed abstain SETS (label-free)."""
    _ans, conf = self_consistency([tuple(sorted(s)) for s in samples])
    return 1.0 - float(conf)


def _reflex_B(majority: frozenset[str], true_set: frozenset[str], removed: set[str]) -> float:
    """okf grounding-closure violation: claims in the answer that STILL have grounding.

    Over-abstention only — a claim asserted-abstained that is not actually orphaned. This is
    checkable from the graph (``true_set`` is the structural orphan set from okf, not an ML
    answer key) and is deliberately *blind* to under-abstention, making B a partial detector.
    """
    over_included = [n for n in majority if n not in true_set and n not in removed]
    return float(len(over_included))


# ---------------------------------------------------------------------------
# Synthetic reasoner over the real graphs (planted error regimes)
# ---------------------------------------------------------------------------

def _mutate(s: frozenset[str], pool: list[str], rng: random.Random) -> frozenset[str]:
    """Sampling noise: flip one claim's membership (models a wobbly individual sample)."""
    if not pool:
        return s
    n = rng.choice(pool)
    return frozenset(s - {n}) if n in s else frozenset(s | {n})


def _noisy(sets: list[frozenset[str]], pool: list[str], rng: random.Random) -> list[frozenset[str]]:
    return [_mutate(s, pool, rng) if rng.random() < P_SAMPLE_NOISE else s for s in sets]


def _pick_regime(rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for name, p in ERR_MIX.items():
        acc += p
        if r < acc:
            return name
    return "hard_both_miss"


def _sample_case(case: dict[str, Any], rng: random.Random, true_set: frozenset[str],
                 *, p_err: float) -> list[frozenset[str]]:
    """Return N proposed abstain sets: a clean/error regime plus per-sample noise.

    clean           : samples = true set (correct, confident).
    uncertain_under : scatter by dropping true claims — A fires, B blind.
    confident_over  : all = true set + a survivor — A quiet, B fires.
    hard_both_miss  : confident under-abstention (drop one true claim, same set) —
                      A quiet (agreement) AND B blind (no over-inclusion) ⇒ residual error.
    """
    pool = sorted(set(true_set) | {s for s in case.get("expectSurvive", []) if s})
    survivors = [s for s in case.get("expectSurvive", []) if s]
    if rng.random() >= p_err or not true_set:
        return _noisy([true_set] * N_SAMPLES, pool, rng)
    regime = _pick_regime(rng)
    if regime == "confident_over" and survivors:
        wrong = frozenset(true_set | {rng.choice(survivors)})
        return _noisy([wrong] * N_SAMPLES, pool, rng)
    if regime == "hard_both_miss" and len(true_set) > 1:
        drop = rng.choice(sorted(true_set))
        wrong = frozenset(true_set - {drop})
        return _noisy([wrong] * N_SAMPLES, pool, rng)
    base = sorted(true_set)  # uncertain_under: scattered under-abstention (sorted ⇒ hash-stable)
    out = []
    for _ in range(N_SAMPLES):
        k = rng.randint(1, max(1, len(base) - 1))
        drop = set(rng.sample(base, k))
        out.append(frozenset(n for n in base if n not in drop))
    return _noisy(out, pool, rng)


def _majority(samples: "list[frozenset[str]]") -> "frozenset[str]":
    """Most common proposed set, ties broken lexicographically (hash-seed independent)."""
    counts = Counter(samples)
    return min(counts, key=lambda s: (-counts[s], tuple(sorted(s))))


@dataclass(frozen=True)
class FusionReport:
    schema: str = "sophia.reasoning.fusion.v1"
    n: int = 0
    base_error: float = 0.0
    d_prime_A: float = 0.0
    d_prime_B: float = 0.0
    d_prime_fused: float = 0.0
    auc_A: float = 0.5
    auc_B: float = 0.5
    auc_fused: float = 0.5
    correlation: float = 0.0
    breakeven_snr: float = float("nan")
    A_clears: bool = False
    B_clears: bool = False
    fused_clears: bool = False
    candidateOnly: bool = True
    level3Evidence: bool = False
    boundary: str = "synthetic reasoner over real okf graphs; fusion law, not a model claim."

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _zscores(xs: list[float]) -> list[float]:
    n = len(xs)
    if n == 0:
        return []
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / n) or 1.0
    return [(x - m) / sd for x in xs]


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0


def run_fusion(*, seed: int = 1234, p_err_scale: float = 0.85, bar: float | None = None,
               cases: list[dict[str, Any]] | None = None) -> FusionReport:
    cases = cases if cases is not None else load_cases()
    the_bar = bar if bar is not None else breakeven_snr()
    a_scores: list[float] = []
    b_scores: list[float] = []
    labels: list[bool] = []  # True == errored
    for i, case in enumerate(cases):
        crng = random.Random(seed * 1009 + i)
        true_set = _true_abstain(case)
        removed = set(case.get("remove", []))
        d = DIFFICULTY.get(case.get("caseType", ""), DEFAULT_DIFFICULTY)
        p_err = min(0.95, p_err_scale * d)
        samples = _sample_case(case, crng, true_set, p_err=p_err)
        majority = _majority(samples)
        a_scores.append(_reflex_A(samples))
        b_scores.append(_reflex_B(majority, true_set, removed))
        labels.append(majority != true_set)

    def split(scores: list[float]) -> tuple[list[float], list[float]]:
        err = [s for s, e in zip(scores, labels) if e]
        clean = [s for s, e in zip(scores, labels) if not e]
        return err, clean

    az, bz = _zscores(a_scores), _zscores(b_scores)
    fused = [a + b for a, b in zip(az, bz)]
    ae, ac = split(a_scores)
    be, bc = split(b_scores)
    fe, fc = split(fused)
    n = len(cases)
    return FusionReport(
        n=n,
        base_error=round(sum(labels) / n, 4) if n else 0.0,
        d_prime_A=round(d_prime(ae, ac), 4),
        d_prime_B=round(d_prime(be, bc), 4),
        d_prime_fused=round(d_prime(fe, fc), 4),
        auc_A=round(auc(ae, ac), 4),
        auc_B=round(auc(be, bc), 4),
        auc_fused=round(auc(fe, fc), 4),
        correlation=round(_pearson(a_scores, b_scores), 4),
        breakeven_snr=the_bar,
        A_clears=bool(d_prime(ae, ac) >= the_bar),
        B_clears=bool(d_prime(be, bc) >= the_bar),
        fused_clears=bool(math.isfinite(d_prime(fe, fc)) and d_prime(fe, fc) >= the_bar),
    )


# ---------------------------------------------------------------------------
# U3: complementarity — B's separation on exactly the errors A misses
# ---------------------------------------------------------------------------

def complementarity(seed: int = 1234, p_err_scale: float = 0.85) -> dict[str, float]:
    """Among errored items A scores LOW (≤ its median on errors), does B still separate?"""
    cases = load_cases()
    rows = []
    for i, case in enumerate(cases):
        crng = random.Random(seed * 1009 + i)
        true_set = _true_abstain(case)
        removed = set(case.get("remove", []))
        d = DIFFICULTY.get(case.get("caseType", ""), DEFAULT_DIFFICULTY)
        samples = _sample_case(case, crng, true_set, p_err=min(0.95, p_err_scale * d))
        majority = _majority(samples)
        rows.append({
            "err": majority != true_set,
            "A": _reflex_A(samples),
            "B": _reflex_B(majority, true_set, removed),
        })
    errs = [r for r in rows if r["err"]]
    cleans = [r for r in rows if not r["err"]]
    if not errs:
        return {"A_missed_errors": 0, "B_auc_on_A_missed": 0.5}
    a_err_sorted = sorted(r["A"] for r in errs)
    med = a_err_sorted[len(a_err_sorted) // 2]
    a_missed = [r for r in errs if r["A"] <= med]  # errors A would let through
    b_missed = [r["B"] for r in a_missed]
    b_clean = [r["B"] for r in cleans]
    return {
        "A_missed_errors": len(a_missed),
        "B_auc_on_A_missed": round(auc(b_missed, b_clean), 4),
    }


# ---------------------------------------------------------------------------
# U2: detection-theory law — Gaussian fusion closed form vs Monte Carlo
# ---------------------------------------------------------------------------

def gaussian_fusion_law(d_a: float, d_b: float, rho: float, trials: int = 40000,
                        seed: int = 7) -> dict[str, float]:
    """MC d′ of summing two correlated Gaussian detectors vs the closed form.

    Closed form (equal-variance signal/noise, correlation ρ in BOTH classes):
    ``d′_fused = (d_a + d_b) / sqrt(2 + 2ρ)``  → √(d_a²+d_b²) at ρ=0 (equal d); → mean at ρ=1.
    """
    rng = random.Random(seed)
    def draw(mean_a: float, mean_b: float) -> float:
        za = rng.gauss(0, 1)
        zb = rho * za + math.sqrt(max(0.0, 1 - rho * rho)) * rng.gauss(0, 1)
        return (mean_a + za) + (mean_b + zb)
    err = [draw(d_a, d_b) for _ in range(trials)]
    clean = [draw(0.0, 0.0) for _ in range(trials)]
    me, mc = sum(err) / trials, sum(clean) / trials
    ve = sum((x - me) ** 2 for x in err) / trials
    vc = sum((x - mc) ** 2 for x in clean) / trials
    mc_dprime = (me - mc) / math.sqrt(0.5 * (ve + vc))
    closed = (d_a + d_b) / math.sqrt(2 + 2 * rho)
    return {"rho": rho, "mc_dprime": round(mc_dprime, 4), "closed_form": round(closed, 4)}


def max_tolerable_rho(d_a: float, d_b: float, bar: float) -> float:
    """Largest correlation at which fused d′ still clears the bar (closed form)."""
    # (d_a+d_b)/sqrt(2+2ρ) >= bar  ⇒  ρ <= ((d_a+d_b)/bar)^2/2 - 1
    val = ((d_a + d_b) / bar) ** 2 / 2 - 1
    return round(max(-1.0, min(1.0, val)), 4)


def run_experiment(seed: int = 1234) -> dict[str, Any]:
    bar = breakeven_snr()
    fusion = run_fusion(seed=seed, bar=bar)
    comp = complementarity(seed=seed)
    law = [gaussian_fusion_law(0.96, 0.96, rho) for rho in (0.0, 0.3, 0.6, 0.9, 1.0)]
    return {
        "breakeven_snr": bar,
        "fusion": fusion.to_dict(),
        "complementarity": comp,
        "gaussian_fusion_law": law,
        "max_tolerable_rho(0.96,0.96)": max_tolerable_rho(0.96, 0.96, bar),
    }


def format_report(res: dict[str, Any]) -> str:
    f = res["fusion"]
    lines = [
        "Reflex fusion — self-consistency (A) + okf grounding-closure (B)",
        "=" * 72,
        f"N={f['n']}  base_error={f['base_error']}  break-even d′ bar={res['breakeven_snr']}",
        "",
        "DETECTABILITY (d′ / AUC)",
        f"  A self-consistency : d′ {f['d_prime_A']:>6}  AUC {f['auc_A']}  clears={f['A_clears']}",
        f"  B okf-grounding    : d′ {f['d_prime_B']:>6}  AUC {f['auc_B']}  clears={f['B_clears']}",
        f"  A+B fused (z-sum)  : d′ {f['d_prime_fused']:>6}  AUC {f['auc_fused']}  clears={f['fused_clears']}",
        f"  correlation ρ(A,B) : {f['correlation']}   (low ⇒ independent ⇒ additive)",
        "",
        f"COMPLEMENTARITY: among {res['complementarity']['A_missed_errors']} errors A misses, "
        f"B's AUC = {res['complementarity']['B_auc_on_A_missed']}",
        f"MAX TOLERABLE ρ for fusion to still clear (d′=0.96 each): "
        f"{res['max_tolerable_rho(0.96,0.96)']}",
        "",
        "DETECTION-THEORY LAW  d′_fused = (d_A+d_B)/√(2+2ρ)   [MC vs closed form]",
    ]
    for row in res["gaussian_fusion_law"]:
        lines.append(f"  ρ={row['rho']:.1f}: MC {row['mc_dprime']:.3f}  closed {row['closed_form']:.3f}")
    lines += [
        "",
        f"candidateOnly={f['candidateOnly']}  level3Evidence={f['level3Evidence']}",
        f"boundary: {f['boundary']}",
    ]
    return "\n".join(lines)


def _self_test() -> int:
    res = run_experiment(seed=1234)
    f = res["fusion"]
    # U1: neither detector clears the bar alone, yet their fusion does.
    assert not f["A_clears"] and not f["B_clears"], "U1 failed: a detector already cleared alone"
    assert f["fused_clears"], f"U1 failed: fusion did not clear the bar ({f['d_prime_fused']})"
    assert f["d_prime_fused"] > f["d_prime_A"] and f["d_prime_fused"] > f["d_prime_B"], "U4 failed"
    # Sanity vs the 'too clean' trap: fusion must be strong but not a degenerate perfect 1.0.
    assert f["auc_fused"] < 0.99, f"fused AUC suspiciously perfect ({f['auc_fused']})"
    # U2: the detection-theory law holds (MC ≈ closed form) and degrades with ρ.
    law = res["gaussian_fusion_law"]
    assert all(abs(r["mc_dprime"] - r["closed_form"]) < 0.05 for r in law), "U2 failed: MC≠closed"
    assert law[0]["closed_form"] > law[-1]["closed_form"] + 0.3, "U2 failed: no ρ degradation"
    # U3: B separates exactly where A is blind.
    assert res["complementarity"]["B_auc_on_A_missed"] > 0.6, "U3 failed: B not complementary"
    # Independence: measured correlation should be low (different failure modes).
    assert abs(f["correlation"]) < 0.5, f"correlation too high ({f['correlation']})"
    print(
        f"self-test OK: A d′={f['d_prime_A']} B d′={f['d_prime_B']} → fused d′={f['d_prime_fused']} "
        f"(clears {res['breakeven_snr']}); ρ={f['correlation']}; "
        f"B AUC on A-missed={res['complementarity']['B_auc_on_A_missed']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true", help="run the full experiment + report")
    p.add_argument("--self-test", action="store_true", help="assert the invariants and exit")
    p.add_argument("--json", action="store_true", help="emit raw results as JSON")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.run or args.json:
        res = run_experiment(seed=args.seed)
        print(json.dumps(res, indent=2) if args.json else format_report(res))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
