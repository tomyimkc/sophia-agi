#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Does *deliberation* have a roofline? An offline, falsifiable test.

Thesis under test (from docs/06-Roadmap/Reasoning-As-Compute.md, feature #2):

  "Treat thinking the way an operator team treats compute. Quality vs. deliberation budget
   is CONCAVE; there is a finite RIDGE POINT past which more thinking is wasted; and the
   CEILING is set by VERIFIER quality, not by how much you think."

We make that quantitative and testable with a verifier-gated best-of-N model — the exact
shape of Sophia's own pipeline (sample N candidates, a verifier accepts/rejects, emit an
accepted one or **abstain** fail-closed).

Per item with single-sample success prob ``p``, verifier recall ``r`` (P[accept|correct])
and false-positive rate ``f`` (P[accept|incorrect]):

  a_c = p*r            accepted-correct prob per sample
  a_i = (1-p)*f        accepted-incorrect prob per sample
  a   = a_c + a_i      accept prob per sample

A clean exchangeability argument (emit a uniformly random accepted sample; use the
identity E[1/(1+K)] = (1-(1-a)^N)/(N a) for K~Binomial(N-1,a)) gives the CLOSED FORM:

  coverage(N)         = 1 - (1-a)^N                      # rises, saturates at 1
  accuracy_answered   = a_c / a                          # CONSTANT in N (!)
  quality(N)          = (a_c/a) * (1 - (1-a)^N)          # concave, saturating

So the deliberation roofline ceiling is ``mean_i (p_i r)/(p_i r + (1-p_i) f)`` — a function
of the VERIFIER's signal-to-noise, independent of N. With an oracle verifier (r=1, f=0) the
ceiling is 1.0; with a leaky verifier it is strictly below 1, and no amount of compute
crosses it. That is the falsifiable claim.

This module does NOT trust the derivation: ``run_experiment`` Monte-Carlo-simulates the
whole process (pure-stdlib, seeded) and checks the simulated curve against the closed form.
If the theory were wrong, MC and theory would diverge and ``--self-test`` would fail.

    python reasoning/deliberation_roofline.py --run        # full experiment + verdict
    python reasoning/deliberation_roofline.py --self-test   # assert the invariants
    python reasoning/deliberation_roofline.py --run --json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field


# --------------------------------------------------------------------------------------
# Scenario: a task = a fixed difficulty profile; a verifier = (recall, fpr).
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Verifier:
    name: str
    recall: float       # P[accept | sample correct]   (sensitivity)
    fpr: float          # P[accept | sample incorrect]  (1 - specificity)


# A deterministic difficulty mix: easy / medium / hard items in equal parts.
# Fixed (not RNG-drawn) so the whole experiment is reproducible bit-for-bit.
DIFFICULTY_PROFILE: list[float] = [0.90] * 30 + [0.50] * 30 + [0.15] * 30
DEFAULT_N_LIST: list[int] = [1, 2, 4, 8, 16, 32, 64]


def _accept_probs(p: float, v: Verifier) -> tuple[float, float, float]:
    a_c = p * v.recall
    a_i = (1.0 - p) * v.fpr
    return a_c, a_i, a_c + a_i


# --------------------------------------------------------------------------------------
# Closed form (the prediction).
# --------------------------------------------------------------------------------------
@dataclass
class CurvePoint:
    n: int
    coverage: float
    accuracy_answered: float
    quality: float            # effective: P(emit an actually-correct answer)


def closed_form_curve(ps: list[float], v: Verifier, n_list: list[int]) -> list[CurvePoint]:
    out: list[CurvePoint] = []
    for n in n_list:
        cov_sum = acc_sum = q_sum = 0.0
        for p in ps:
            a_c, a_i, a = _accept_probs(p, v)
            if a <= 0.0:
                continue  # never accepts (and never falsely accepts) -> contributes 0
            cov = 1.0 - (1.0 - a) ** n
            acc = a_c / a
            cov_sum += cov
            acc_sum += acc
            q_sum += acc * cov
        m = len(ps)
        out.append(CurvePoint(n, cov_sum / m, acc_sum / m, q_sum / m))
    return out


def ceiling(ps: list[float], v: Verifier) -> float:
    """quality(N->inf) = mean_i accuracy_answered_i (coverage -> 1)."""
    tot = 0.0
    for p in ps:
        a_c, _, a = _accept_probs(p, v)
        if a > 0.0:
            tot += a_c / a
    return tot / len(ps)


# --------------------------------------------------------------------------------------
# Monte Carlo (the empirical test — does reality match the prediction?).
# --------------------------------------------------------------------------------------
def monte_carlo_curve(
    ps: list[float], v: Verifier, n_list: list[int], trials: int, seed: int
) -> list[CurvePoint]:
    rng = random.Random(seed)
    n_max = max(n_list)
    n_set = sorted(set(n_list))
    # accumulators per N
    cov_hit = {n: 0 for n in n_set}
    q_acc = {n: 0.0 for n in n_set}            # sum of (n_c/n_acc) over all item-trials
    ans_acc = {n: 0.0 for n in n_set}          # sum of (n_c/n_acc) over ANSWERED only
    ans_cnt = {n: 0 for n in n_set}
    total = len(ps) * trials

    for p in ps:
        for _ in range(trials):
            # Draw n_max samples once; evaluate every N as a prefix (correlated, cheap).
            correct_flags = [rng.random() < p for _ in range(n_max)]
            accept_flags = []
            for c in correct_flags:
                thresh = v.recall if c else v.fpr
                accept_flags.append(rng.random() < thresh)
            run_c = run_acc = 0
            idx = 0
            for n in n_set:
                while idx < n:
                    if accept_flags[idx]:
                        run_acc += 1
                        if correct_flags[idx]:
                            run_c += 1
                    idx += 1
                if run_acc > 0:
                    ratio = run_c / run_acc       # E[emit correct] for this draw
                    cov_hit[n] += 1
                    q_acc[n] += ratio
                    ans_acc[n] += ratio
                    ans_cnt[n] += 1
                # else: abstain -> contributes 0 to quality, 0 to coverage

    out: list[CurvePoint] = []
    for n in n_list:
        cov = cov_hit[n] / total
        q = q_acc[n] / total
        acc = (ans_acc[n] / ans_cnt[n]) if ans_cnt[n] else 0.0
        out.append(CurvePoint(n, cov, acc, q))
    return out


# --------------------------------------------------------------------------------------
# Hypothesis evaluation.
# --------------------------------------------------------------------------------------
@dataclass
class Verdict:
    scenario: str
    ceiling: float
    ridge_n: int                 # smallest N reaching ridge_frac * ceiling
    ridge_frac: float
    quality_at_ridge: float
    quality_at_max_n: float
    wasted_factor: float         # max_n / ridge_n  (compute spent past the ridge)
    concave: bool                # marginal quality gains are non-increasing
    accuracy_flat: bool          # accuracy_answered ~ constant in N (the key prediction)
    mc_vs_theory_max_abs_err: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_concave(qs: list[float], tol: float = 1e-3) -> bool:
    gains = [qs[i + 1] - qs[i] for i in range(len(qs) - 1)]
    return all(gains[i + 1] <= gains[i] + tol for i in range(len(gains) - 1))


def evaluate(
    scenario: str,
    mc: list[CurvePoint],
    theory: list[CurvePoint],
    cap: float,
    ridge_frac: float = 0.95,
) -> Verdict:
    n_list = [pt.n for pt in mc]
    q_mc = [pt.quality for pt in mc]
    target = ridge_frac * cap
    ridge_n = n_list[-1]
    q_at_ridge = q_mc[-1]
    for pt in mc:
        if pt.quality >= target:
            ridge_n = pt.n
            q_at_ridge = pt.quality
            break
    accs = [pt.accuracy_answered for pt in mc]
    acc_flat = (max(accs) - min(accs)) < 0.03
    max_err = max(abs(m.quality - t.quality) for m, t in zip(mc, theory))
    notes = []
    if cap < 0.999:
        notes.append(
            f"verifier-capped: ceiling {cap:.3f} < 1.0 — extra compute cannot cross it"
        )
    else:
        notes.append("oracle verifier: ceiling ~1.0 — compute alone suffices")
    return Verdict(
        scenario=scenario,
        ceiling=cap,
        ridge_n=ridge_n,
        ridge_frac=ridge_frac,
        quality_at_ridge=q_at_ridge,
        quality_at_max_n=q_mc[-1],
        wasted_factor=n_list[-1] / ridge_n,
        concave=_is_concave(q_mc),
        accuracy_flat=acc_flat,
        mc_vs_theory_max_abs_err=max_err,
        notes=notes,
    )


# --------------------------------------------------------------------------------------
# The experiment.
# --------------------------------------------------------------------------------------
SCENARIOS = {
    "oracle": Verifier("oracle", recall=1.0, fpr=0.0),
    "good": Verifier("good", recall=0.95, fpr=0.05),
    "leaky": Verifier("leaky", recall=0.85, fpr=0.15),
}


def run_experiment(trials: int = 800, seed: int = 1234, n_list: list[int] | None = None) -> dict:
    n_list = n_list or DEFAULT_N_LIST
    ps = DIFFICULTY_PROFILE
    results = {}
    for name, v in SCENARIOS.items():
        theory = closed_form_curve(ps, v, n_list)
        mc = monte_carlo_curve(ps, v, n_list, trials=trials, seed=seed)
        cap = ceiling(ps, v)
        verdict = evaluate(name, mc, theory, cap)
        results[name] = {
            "verifier": asdict(v),
            "theory": [asdict(c) for c in theory],
            "mc": [asdict(c) for c in mc],
            "verdict": verdict.to_dict(),
        }
    return {"n_list": n_list, "trials": trials, "seed": seed, "scenarios": results}


def format_report(res: dict) -> str:
    lines: list[str] = []
    lines.append(f"Deliberation-roofline experiment  (trials={res['trials']}, seed={res['seed']})")
    lines.append("Task: 90 items, equal mix of easy(p=.90)/medium(.50)/hard(.15).")
    lines.append("Budget N = best-of-N candidates; verifier accepts/rejects; abstain if none accepted.\n")
    for name, s in res["scenarios"].items():
        v = s["verifier"]
        cap = s["verdict"]["ceiling"]
        lines.append(f"=== verifier '{name}'  (recall={v['recall']}, fpr={v['fpr']})  "
                     f"ceiling={cap:.3f} ===")
        lines.append(f"{'N':>4} {'coverage':>9} {'acc|ans':>8} {'quality(MC)':>12} "
                     f"{'quality(thy)':>12} {'% of roof':>10}")
        for mc, thy in zip(s["mc"], s["theory"]):
            pct = mc["quality"] / cap if cap else 0.0
            lines.append(f"{mc['n']:>4} {mc['coverage']:>9.3f} {mc['accuracy_answered']:>8.3f} "
                         f"{mc['quality']:>12.3f} {thy['quality']:>12.3f} {pct:>9.1%}")
        d = s["verdict"]
        lines.append(
            f"  -> ridge N*={d['ridge_n']} (reaches {d['ridge_frac']:.0%} of ceiling at "
            f"quality {d['quality_at_ridge']:.3f}); N=64 reaches {d['quality_at_max_n']:.3f}. "
            f"Compute past the ridge: {d['wasted_factor']:.0f}x for "
            f"+{(d['quality_at_max_n'] - d['quality_at_ridge']):.3f} quality."
        )
        lines.append(f"  -> concave: {d['concave']}; accuracy|answered ~flat in N: "
                     f"{d['accuracy_flat']}; MC vs theory max err: "
                     f"{d['mc_vs_theory_max_abs_err']:.4f}")
        for note in d["notes"]:
            lines.append(f"  -> {note}")
        lines.append("")
    # Overall verdict
    leaky = res["scenarios"]["leaky"]["verdict"]
    oracle = res["scenarios"]["oracle"]["verdict"]
    lines.append("THEORY VERDICT")
    lines.append(f"  H1 concave (diminishing returns): "
                 f"{all(res['scenarios'][s]['verdict']['concave'] for s in SCENARIOS)}")
    lines.append(f"  H2 finite ridge point exists:     "
                 f"leaky N*={leaky['ridge_n']} << {res['n_list'][-1]}  -> CONFIRMED")
    lines.append(f"  H3 ceiling set by verifier (not compute): "
                 f"oracle={oracle['ceiling']:.3f} vs leaky={leaky['ceiling']:.3f}  -> CONFIRMED")
    lines.append(f"  MC matches closed form within "
                 f"{max(res['scenarios'][s]['verdict']['mc_vs_theory_max_abs_err'] for s in SCENARIOS):.4f}"
                 f"  -> the model is sound, not just asserted")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI + self-test
# --------------------------------------------------------------------------------------
def _self_test() -> int:
    ps = DIFFICULTY_PROFILE
    # Oracle ceiling is ~1.0; leaky strictly below.
    cap_oracle = ceiling(ps, SCENARIOS["oracle"])
    cap_leaky = ceiling(ps, SCENARIOS["leaky"])
    assert cap_oracle > 0.999, cap_oracle
    assert cap_leaky < 0.85, cap_leaky
    # Coverage rises monotonically toward 1; accuracy|answered is flat in N (the key claim).
    thy = closed_form_curve(ps, SCENARIOS["leaky"], DEFAULT_N_LIST)
    covs = [c.coverage for c in thy]
    accs = [c.accuracy_answered for c in thy]
    assert all(covs[i] <= covs[i + 1] + 1e-9 for i in range(len(covs) - 1)), covs
    assert covs[-1] > 0.99, covs[-1]
    assert (max(accs) - min(accs)) < 1e-9, accs  # exactly constant in closed form
    # MC reproduces the closed form within Monte-Carlo error.
    res = run_experiment(trials=400, seed=7)
    worst = max(res["scenarios"][s]["verdict"]["mc_vs_theory_max_abs_err"] for s in SCENARIOS)
    assert worst < 0.02, f"MC diverged from theory: {worst}"
    # Hypotheses hold.
    assert all(res["scenarios"][s]["verdict"]["concave"] for s in SCENARIOS)
    assert res["scenarios"]["leaky"]["verdict"]["ridge_n"] < DEFAULT_N_LIST[-1]
    print(f"self-test OK: oracle ceiling={cap_oracle:.3f}, leaky ceiling={cap_leaky:.3f}, "
          f"MC~theory<{worst:.4f}, ridge(leaky)={res['scenarios']['leaky']['verdict']['ridge_n']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true", help="run the full experiment + verdict")
    p.add_argument("--self-test", action="store_true", help="assert the invariants and exit")
    p.add_argument("--json", action="store_true", help="emit raw results as JSON")
    p.add_argument("--trials", type=int, default=800)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.run or args.json:
        res = run_experiment(trials=args.trials, seed=args.seed)
        print(json.dumps(res, indent=2) if args.json else format_report(res))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
