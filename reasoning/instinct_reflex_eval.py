# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Reflex ROC/SNR harness — *is a given reflex good enough to trust?*

``reasoning/instinct_gate.py`` proved (on a planted-truth policy model) that early
reflex re-route beats late self-correction **only above a break-even reflex SNR**: a
reflex below that bar *hurts*. That result is useless until you can answer, for a *real*
reflex, the one question it raises:

    Does this detector's separation between "chain is wrong" and "chain is fine"
    clear the break-even bar?

This module is that measurement. It is the **harness**, not a capability claim — exactly
the repo's "the harness is the deliverable" idiom. It wires the first concrete reflex,
**self-consistency disagreement** (``agent.calibration.self_consistency`` — label-free,
already shipped), to a planted ground-truth oracle, and reports:

  - **d′ (detectability)** — the standardized mean separation of the reflex score on
    errored vs clean items: ``(mean_err − mean_clean) / sqrt(½(var_err+var_clean))``.
    This is the *same quantity* ``instinct_gate``'s ``snr`` is (a unit-variance mean
    separation), so the two are directly comparable — modulo the honest caveat that the
    reflex score is bounded [0,1] and non-Gaussian, so d′ is an approximation.
  - **AUC** — rank-probability a random errored item scores above a random clean one.
  - **clears_breakeven** — ``d′ ≥`` the break-even SNR that ``instinct_gate`` reports for
    its policy model. This is the go/no-go on shipping the reflex.

Ground truth comes from ``eval/belief_revision/belief_revision_50_v1.jsonl`` (the
"change-its-mind" dataset: transitive retraction, planted abstain sets). Because a real
multi-sample model run is GPU/egress-gated here, the default *answer sampler* is a
seeded, deterministic synthetic reasoner whose per-item error propensity is driven by the
dataset's own case-type difficulty — so the harness runs offline and is testable, while
the sampler is a single pluggable function a real model drops into unchanged
(``run_reflex_eval(..., sampler=my_model_sampler)``).

Honest scope (``candidateOnly: true``, ``canClaimAGI: false``). The *measured d′ here is
the synthetic sampler's*, present to validate the harness end-to-end and demonstrate the
go/no-go. The real number — self-consistency d′ from a real model — is a gated next step
(≥2 judge families, ≥3 seeds, CI) and is NOT claimed. The harness is what ships.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.calibration import area_under_risk_coverage, self_consistency  # noqa: E402

DATASET = ROOT / "eval" / "belief_revision" / "belief_revision_50_v1.jsonl"

#: Per-case-type error propensity (harder retraction structure ⇒ more errors). Derived
#: from the dataset's own structure: deeper/multi cascades are harder than fail-closed.
DIFFICULTY: dict[str, float] = {
    "not_found_fail_closed": 0.15,             # remove target absent ⇒ trivial
    "transitive_retraction": 0.45,             # follow a transitive cascade
    "single_source_removed_multi_survives": 0.55,  # tempting to over-abstain
    "multi_retraction_orphans_multi": 0.80,    # largest cascade, multiple removes
}
DEFAULT_DIFFICULTY = 0.50
#: Number of sampled answers per item used to compute self-consistency.
N_SAMPLES = 7
#: Number of distinct wrong answers an errored reasoner can scatter across.
N_DISTRACTORS = 4


def load_cases(path: Path = DATASET) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Answer samplers (pluggable). A real model replaces ``synthetic_sampler``.
# ---------------------------------------------------------------------------

def synthetic_sampler(case: dict[str, Any], rng: random.Random, *, competence: float) -> list[str]:
    """Seeded synthetic reasoner. Each sample is the correct answer w.p. p_correct,
    else a uniformly-chosen distractor. Easier cases + higher competence ⇒ more
    agreement (high self-consistency); harder cases scatter (low self-consistency)."""
    d = DIFFICULTY.get(case.get("caseType", ""), DEFAULT_DIFFICULTY)
    p_correct = max(0.05, min(0.98, competence * (1.0 - 0.6 * d)))
    samples = []
    for _ in range(N_SAMPLES):
        if rng.random() < p_correct:
            samples.append("CORRECT")
        else:
            samples.append(f"WRONG_{rng.randrange(N_DISTRACTORS)}")
    return samples


def noise_sampler(case: dict[str, Any], rng: random.Random, **_: Any) -> list[str]:
    """Degenerate control: every token equally likely ⇒ no usable signal. The harness
    must report d′≈0 / AUC≈0.5 here (it must not manufacture separation)."""
    tokens = ["CORRECT"] + [f"WRONG_{i}" for i in range(N_DISTRACTORS)]
    return [rng.choice(tokens) for _ in range(N_SAMPLES)]


# ---------------------------------------------------------------------------
# Reflex signal (pluggable). Default: self-consistency disagreement.
# ---------------------------------------------------------------------------

def self_consistency_reflex(samples: Sequence[str]) -> float:
    """Reflex *wrongness* score = 1 − agreement fraction. High ⇒ fire (re-route)."""
    _answer, confidence = self_consistency(samples)
    return 1.0 - float(confidence)


# ---------------------------------------------------------------------------
# Detectability metrics
# ---------------------------------------------------------------------------

def _mean_var(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n
    return m, v


def d_prime(scores_err: Sequence[float], scores_clean: Sequence[float]) -> float:
    """Standardized mean separation — the same unit as ``instinct_gate``'s SNR."""
    me, ve = _mean_var(scores_err)
    mc, vc = _mean_var(scores_clean)
    pooled = math.sqrt(0.5 * (ve + vc))
    if pooled < 1e-9:
        return 0.0 if abs(me - mc) < 1e-9 else float("inf")
    return (me - mc) / pooled


def auc(scores_err: Sequence[float], scores_clean: Sequence[float]) -> float:
    """ROC AUC via the Mann-Whitney rank statistic (ties = 0.5)."""
    if not scores_err or not scores_clean:
        return 0.5
    wins = 0.0
    for se in scores_err:
        for sc in scores_clean:
            wins += 1.0 if se > sc else 0.5 if se == sc else 0.0
    return wins / (len(scores_err) * len(scores_clean))


def breakeven_snr() -> float:
    """The bar the reflex must clear — taken from the policy model, not invented here."""
    from reasoning.instinct_gate import run_experiment
    return run_experiment(trials=2000, seed=1234)["verdict"]["h3_breakeven_snr"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReflexReport:
    schema: str = "sophia.reasoning.reflex_eval.v1"
    n: int = 0
    base_error: float = 0.0
    d_prime: float = 0.0
    auc: float = 0.5
    mean_reflex_error: float = 0.0
    mean_reflex_clean: float = 0.0
    breakeven_snr: float = float("nan")
    clears_breakeven: bool = False
    candidateOnly: bool = True
    level3Evidence: bool = False
    boundary: str = "synthetic-sampler d′; real-model self-consistency d′ is a gated next step."

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def run_reflex_eval(
    *,
    competence: float = 0.62,
    seed: int = 1234,
    sampler: Callable[..., list[str]] = synthetic_sampler,
    reflex: Callable[[Sequence[str]], float] = self_consistency_reflex,
    cases: list[dict[str, Any]] | None = None,
    bar: float | None = None,
) -> ReflexReport:
    cases = cases if cases is not None else load_cases()
    rng = random.Random(seed)
    scores_err: list[float] = []
    scores_clean: list[float] = []
    for i, case in enumerate(cases):
        # Per-item deterministic stream so adding/removing a case can't reshuffle others.
        crng = random.Random(seed * 1009 + i)
        samples = sampler(case, crng, competence=competence)
        answer, _conf = self_consistency(samples)
        is_error = (answer != "CORRECT")  # oracle: majority answer wrong?
        score = reflex(samples)
        (scores_err if is_error else scores_clean).append(score)
    n = len(cases)
    base_err = len(scores_err) / n if n else 0.0
    dp = d_prime(scores_err, scores_clean)
    a = auc(scores_err, scores_clean)
    me, _ = _mean_var(scores_err)
    mc, _ = _mean_var(scores_clean)
    the_bar = bar if bar is not None else breakeven_snr()
    return ReflexReport(
        n=n,
        base_error=round(base_err, 4),
        d_prime=round(dp, 4) if math.isfinite(dp) else dp,
        auc=round(a, 4),
        mean_reflex_error=round(me, 4),
        mean_reflex_clean=round(mc, 4),
        breakeven_snr=the_bar,
        clears_breakeven=bool(math.isfinite(dp) and dp >= the_bar),
    )


def competence_sweep(seed: int = 1234, bar: float | None = None) -> list[dict[str, Any]]:
    the_bar = bar if bar is not None else breakeven_snr()
    rows = []
    for comp in [0.30, 0.45, 0.62, 0.80, 0.95]:
        r = run_reflex_eval(competence=comp, seed=seed, bar=the_bar)
        rows.append({
            "competence": comp, "base_error": r.base_error,
            "d_prime": r.d_prime, "auc": r.auc, "clears_breakeven": r.clears_breakeven,
        })
    return rows


def run_experiment(seed: int = 1234) -> dict[str, Any]:
    bar = breakeven_snr()
    main_report = run_reflex_eval(competence=0.62, seed=seed, bar=bar)
    noise_report = run_reflex_eval(competence=0.62, seed=seed, sampler=noise_sampler, bar=bar)
    sweep = competence_sweep(seed=seed, bar=bar)
    return {
        "dataset": str(DATASET.relative_to(ROOT)),
        "params": {"n_samples": N_SAMPLES, "n_distractors": N_DISTRACTORS, "seed": seed},
        "breakeven_snr": bar,
        "self_consistency_reflex": main_report.to_dict(),
        "noise_control": noise_report.to_dict(),
        "competence_sweep": sweep,
    }


def format_report(res: dict[str, Any]) -> str:
    r = res["self_consistency_reflex"]
    nz = res["noise_control"]
    lines = [
        "Reflex ROC/SNR harness — self-consistency disagreement vs belief-revision oracle",
        "=" * 80,
        f"dataset: {res['dataset']}  (N={r['n']})   break-even SNR bar = {res['breakeven_snr']}",
        "",
        "SELF-CONSISTENCY REFLEX (synthetic sampler, competence=0.62)",
        f"  base error rate      : {r['base_error']}",
        f"  mean reflex | error  : {r['mean_reflex_error']}   | clean: {r['mean_reflex_clean']}",
        f"  d′ (detectability)   : {r['d_prime']}   AUC: {r['auc']}",
        f"  clears break-even bar: {r['clears_breakeven']}  (d′ ≥ {res['breakeven_snr']})",
        "",
        "NOISE CONTROL (no-signal sampler — harness must NOT manufacture separation)",
        f"  d′ : {nz['d_prime']}   AUC: {nz['auc']}   clears: {nz['clears_breakeven']}",
        "",
        "COMPETENCE SWEEP",
    ]
    for row in res["competence_sweep"]:
        lines.append(
            f"  comp={row['competence']:.2f}: base_err {row['base_error']:.3f}  "
            f"d′ {row['d_prime']:.3f}  AUC {row['auc']:.3f}  clears={row['clears_breakeven']}"
        )
    lines += [
        "",
        f"candidateOnly={r['candidateOnly']}  level3Evidence={r['level3Evidence']}",
        f"boundary: {r['boundary']}",
    ]
    return "\n".join(lines)


def _self_test() -> int:
    res = run_experiment(seed=1234)
    r = res["self_consistency_reflex"]
    nz = res["noise_control"]
    # R1: the reflex separates errors from clean at a realistic competence.
    assert r["auc"] > 0.65, f"R1 failed: reflex AUC too low ({r['auc']})"
    assert r["mean_reflex_error"] > r["mean_reflex_clean"], "R1 failed: wrong sign"
    # R2: d′ is finite and positive (a usable detector direction exists).
    assert math.isfinite(r["d_prime"]) and r["d_prime"] > 0, f"R2 failed: d′={r['d_prime']}"
    # R3: the no-signal control must collapse to chance (harness honesty).
    assert abs(nz["auc"] - 0.5) < 0.12, f"R3 failed: noise AUC not ~0.5 ({nz['auc']})"
    assert not nz["clears_breakeven"], "R3 failed: noise control falsely cleared the bar"
    # R4: more competent reasoner ⇒ fewer errors (sampler/oracle wired correctly).
    errs = [row["base_error"] for row in res["competence_sweep"]]
    assert errs[0] > errs[-1] + 0.1, f"R4 failed: base error not decreasing ({errs})"
    print(
        f"self-test OK: reflex d′={r['d_prime']} AUC={r['auc']} "
        f"clears_breakeven({res['breakeven_snr']})={r['clears_breakeven']}; "
        f"noise AUC={nz['auc']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true", help="run the full evaluation + report")
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
