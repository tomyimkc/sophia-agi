# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""End-to-end instinct outcome — does *firing the reflex and re-routing* actually help?

Everything upstream measured *detection* (d′, AUC). This module measures the **outcome**: a
reflex with a *real, measured* operating point drives the ``instinct_gate`` re-route policy on
the belief-revision task, and we ask whether it improves what the operator actually cares
about — most of all the **confident-wrong rate** (a wrong answer asserted as if correct). In
Sophia's idiom, converting a confident error into a *re-route to a correct answer* or an honest
*escalate* is the whole point of an instinct.

It unifies the three prior pieces on ONE task:
  - the policy (commit / late-self-correct / instinct re-route) from ``instinct_gate``;
  - the *real* detector operating point (TPR/FPR) measured by the real-model fusion run
    (``reasoning/results/fusion_realmodel_*.json``), via the fire rule ``A≥0.6 OR B≥1``;
  - the same fail-closed ``escalate`` ceiling (a ko after the re-route budget).

Outcomes per policy (the load-bearing one is wrong_asserted):
  correct          — final answer right.
  wrong_asserted   — final answer wrong AND asserted (the dangerous outcome).
  escalate         — gave up to a human / new info (fail-closed; NOT a wrong assertion).
  mean_cost        — attempts spent.

Falsifiable results (``--self-test`` / the test module):
  E1  SAFETY WIN. With a usable detector (DeepSeek profile, TPR 0.74) the instinct's
      wrong_asserted rate is **below both commit and late** — it slashes confident errors.
  E2  RECOVERY. Its correct rate is ≥ commit's (re-route recovers errored attempts), not just
      a trade of correctness for abstention.
  E3  GATED BY DETECTION. With a blind detector (Claude-haiku profile, TPR≈0 because its errors
      are confident under-abstentions the bus can't see) the instinct ≈ commit on every metric:
      **an instinct cannot help against errors it cannot detect.**
  E4  MONOTONE. Confident-wrong reduction grows with detector TPR (a sweep), recovering E1/E3
      as two points on one curve.

Honest scope (``candidateOnly: true``, ``canClaimAGI: false``). The operating points are real
(measured from the committed real-model artifacts); the *outcome* simulation models re-attempts
as i.i.d. draws at the model's measured base error. That is the key simplification: a model with
*systematic* (correlated) errors would gain less from re-routing — though escalation still
bounds confident-wrong, and Claude-haiku's base_error=1.0 already exhibits the worst case
(re-route futile, only escalation could help, and its errors are undetectable so it can't even
do that). Not a model claim; the harness + the gating law are the deliverable.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_gate import MAX_REROUTE, P_BREAK, P_FIX  # noqa: E402

RESULTS = ROOT / "reasoning" / "results"
#: Fire rule operating thresholds (interpretable, model-independent): disagreement ≥ 0.6
#: (top answer below ~half the samples) OR any grounding over-inclusion (B) OR any grounding
#: under-inclusion / incompleteness (B2 — the detector added to cover confident under-abstention).
A_FIRE, B_FIRE, B2_FIRE = 0.6, 1.0, 1.0


@dataclass(frozen=True)
class OperatingPoint:
    """A reflex's measured behaviour on a model: base error + fire rates."""

    label: str
    base_error: float
    tpr: float  # P(fire | errored attempt)
    fpr: float  # P(fire | clean attempt)
    n_clean: int = 0


def profile_from_artifact(path: Path) -> OperatingPoint:
    """Derive (base_error, TPR, FPR) from a real-model fusion artifact via A≥0.6 OR B≥1."""
    data = json.loads(Path(path).read_text())
    pc = data["per_case"]
    model = data["report"]["model"]
    err = [c for c in pc if c["is_error"]]
    cln = [c for c in pc if not c["is_error"]]

    def fires(c: dict[str, Any]) -> bool:
        # B2 added to cover under-abstention; .get keeps old (v1, no-B2) artifacts loadable.
        return c["A"] >= A_FIRE or c["B"] >= B_FIRE or c.get("B2", 0.0) >= B2_FIRE

    tpr = sum(fires(c) for c in err) / len(err) if err else 0.0
    fpr = sum(fires(c) for c in cln) / len(cln) if cln else 0.0
    return OperatingPoint(
        label=model, base_error=round(len(err) / len(pc), 4) if pc else 0.0,
        tpr=round(tpr, 4), fpr=round(fpr, 4), n_clean=len(cln),
    )


# ---------------------------------------------------------------------------
# Policy outcome simulation
# ---------------------------------------------------------------------------

@dataclass
class Outcome:
    correct: float = 0.0
    wrong_asserted: float = 0.0
    escalate: float = 0.0
    mean_cost: float = 0.0


def _commit(rng: random.Random, op: OperatingPoint) -> tuple[str, float]:
    err = rng.random() < op.base_error
    return ("wrong" if err else "correct"), 1.0


def _late(rng: random.Random, op: OperatingPoint) -> tuple[str, float]:
    err = rng.random() < op.base_error
    if err:
        err = not (rng.random() < P_FIX)         # self-correct may fix
    elif rng.random() < P_BREAK:
        err = True                                # …or break a right answer
    return ("wrong" if err else "correct"), 2.0


def _instinct(rng: random.Random, op: OperatingPoint) -> tuple[str, float]:
    cost = 0.0
    for attempt in range(MAX_REROUTE + 1):
        err = rng.random() < op.base_error
        fire = rng.random() < (op.tpr if err else op.fpr)
        cost += 1.0
        if not fire:
            return ("wrong" if err else "correct"), cost   # commit this attempt
        if attempt == MAX_REROUTE:
            return "escalate", cost                          # ko: budget spent, still firing
        # else: re-route to a fresh attempt
    return "escalate", cost  # pragma: no cover


def simulate(op: OperatingPoint, *, trials: int = 20000, seed: int = 1234) -> dict[str, Outcome]:
    out: dict[str, Outcome] = {}
    # Fixed per-policy stream offsets (NOT hash(name) — Python salts str hashes per process).
    for name, offset, fn in (("commit", 1, _commit), ("late", 2, _late), ("instinct", 3, _instinct)):
        rng = random.Random(seed * 31 + offset)
        c = w = e = 0
        cost = 0.0
        for _ in range(trials):
            res, k = fn(rng, op)
            cost += k
            c += res == "correct"; w += res == "wrong"; e += res == "escalate"
        out[name] = Outcome(
            correct=round(c / trials, 4), wrong_asserted=round(w / trials, 4),
            escalate=round(e / trials, 4), mean_cost=round(cost / trials, 4),
        )
    return out


def tpr_sweep(base_error: float, fpr: float, *, trials: int = 20000, seed: int = 1234) -> list[dict[str, float]]:
    rows = []
    for tpr in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        op = OperatingPoint("sweep", base_error, tpr, fpr)
        ins = simulate(op, trials=trials, seed=seed)["instinct"]
        rows.append({
            "tpr": tpr, "wrong_asserted": ins.wrong_asserted,
            "correct": ins.correct, "escalate": ins.escalate, "cost": ins.mean_cost,
        })
    return rows


def run_experiment(seed: int = 1234) -> dict[str, Any]:
    profiles = {}
    for fname in ("fusion_realmodel_deepseek.json", "fusion_realmodel_llmhub-haiku.json"):
        p = RESULTS / fname
        if p.exists():
            op = profile_from_artifact(p)
            profiles[op.label] = {"op": op.__dict__, "policies": {k: v.__dict__ for k, v in simulate(op, seed=seed).items()}}
    ds = profile_from_artifact(RESULTS / "fusion_realmodel_deepseek.json")
    return {
        "fire_rule": f"A>={A_FIRE} OR B>={B_FIRE}",
        "profiles": profiles,
        "tpr_sweep_at_deepseek_base": tpr_sweep(ds.base_error, ds.fpr, seed=seed),
    }


def format_report(res: dict[str, Any]) -> str:
    lines = [
        "End-to-end instinct outcome — reflex-driven re-route on belief revision",
        "=" * 74,
        f"fire rule: {res['fire_rule']}",
        "",
    ]
    for model, blk in res["profiles"].items():
        op = blk["op"]
        lines.append(f"MODEL {model}  (base_err={op['base_error']} TPR={op['tpr']} FPR={op['fpr']})")
        lines.append(f"  {'policy':10}  {'correct':>8} {'wrong!':>8} {'escalate':>9} {'cost':>6}")
        for pol, o in blk["policies"].items():
            lines.append(f"  {pol:10}  {o['correct']:>8.3f} {o['wrong_asserted']:>8.3f} "
                         f"{o['escalate']:>9.3f} {o['mean_cost']:>6.2f}")
        lines.append("")
    lines.append("TPR SWEEP (confident-wrong vs detector recall, at DeepSeek base/FPR)")
    for r in res["tpr_sweep_at_deepseek_base"]:
        lines.append(f"  TPR={r['tpr']:.1f}: wrong_asserted {r['wrong_asserted']:.3f}  "
                     f"correct {r['correct']:.3f}  escalate {r['escalate']:.3f}  cost {r['cost']:.2f}")
    lines += ["", "candidateOnly=True  level3Evidence=False",
              "boundary: real operating points; outcomes model re-attempts as i.i.d. at base error."]
    return "\n".join(lines)


def _self_test() -> int:
    res = run_experiment(seed=1234)
    profs = res["profiles"]
    ds = next(b for m, b in profs.items() if "deepseek" in m.lower())
    hk = next((b for m, b in profs.items() if "haiku" in m.lower()), None)
    dpol = ds["policies"]
    # E1: safety win — instinct cuts confident-wrong below both commit and late.
    assert dpol["instinct"]["wrong_asserted"] < dpol["commit"]["wrong_asserted"], "E1 commit"
    assert dpol["instinct"]["wrong_asserted"] < dpol["late"]["wrong_asserted"], "E1 late"
    # E2: recovery — correctness does not fall below commit.
    assert dpol["instinct"]["correct"] >= dpol["commit"]["correct"] - 0.01, "E2"
    # E3: gated by detection — a blind detector yields ~no change vs commit.
    if hk is not None:
        hp = hk["policies"]
        assert abs(hp["instinct"]["wrong_asserted"] - hp["commit"]["wrong_asserted"]) < 0.05, "E3"
        assert hp["instinct"]["correct"] < 0.05 and hp["commit"]["correct"] < 0.05, "E3 base"
    # E4: monotone — confident-wrong falls as TPR rises.
    sweep = res["tpr_sweep_at_deepseek_base"]
    wa = [r["wrong_asserted"] for r in sweep]
    assert all(wa[i] >= wa[i + 1] - 0.01 for i in range(len(wa) - 1)), f"E4 not monotone: {wa}"
    assert wa[0] - wa[-1] > 0.2, "E4 no real reduction"
    print(
        f"self-test OK: DeepSeek wrong_asserted commit {dpol['commit']['wrong_asserted']} "
        f"-> instinct {dpol['instinct']['wrong_asserted']} (escalate {dpol['instinct']['escalate']}); "
        f"haiku instinct≈commit; TPR sweep wrong {wa[0]}->{wa[-1]}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--json", action="store_true")
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
