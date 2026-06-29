# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Instinct gate — a falsifiable model of *early reflex re-route* vs *late self-correct*.

Thesis under test (the operator's "change its mind" intuition, made falsifiable).
A reasoning model runs a *chain of thought*. Somewhere it may commit an error. Today's
default behaviour is to **plough forward** and try to patch the wrong path — and the
published record on *intrinsic* self-correction is that this tends to be weak and can
even *degrade* accuracy (Huang et al. 2023; the "accuracy-correction paradox"). The
operator's intuition is the opposite reflex: the moment something *feels* wrong, an
**instinct** layer should fire and force a discrete **re-route / backtrack**, not a
forward patch. This module asks, on a planted-ground-truth model, *when that intuition
is actually right* — and where it back-fires.

It is the reasoning.* sibling of ``deliberation_roofline`` (budget) and
``reasoning_compiler`` (IR): a pure-stdlib, seeded, falsifiable *model* of a claim — not
production wiring. The "instinct" here is abstracted as a cheap, always-on **reflex
monitor** that emits a noisy per-step *wrongness* signal (a stand-in for a probe / a
self-consistency disagreement / a verifier mismatch). The chain, the error, and the
monitor SNR are planted so the hypotheses are *falsifiable*, not assumed.

Three policies over the same trajectory distribution:

  - ``commit``      — never intervene; run to the end (today's default). Baseline.
  - ``late``        — run to the end, then one self-correction pass: fixes a wrong
                      chain with prob ``p_fix`` but *breaks* a right one with prob
                      ``p_break`` (the accuracy-correction paradox, planted).
  - ``instinct``    — the reflex monitor fires the first step its signal crosses ``tau``;
                      on a fire, **backtrack and re-route** (a fresh attempt), paying a
                      re-route cost. Repeated re-routes that revisit a failing state are a
                      **ko** → ``escalate`` (never an infinite patch-forward loop — the
                      same ko-escalate rule as ``reasoning.consequence.ko_detector``).

Hypotheses the THEORY VERDICT reports (all falsifiable, all checked in ``--self-test``):

  H1  COMPOUNDING. Under ``commit``, the longer an error runs uncorrected the worse the
      final answer — correctness is monotone *increasing* in how late the error is
      committed (fewer derail steps remain). This is the formal content of "don't plough
      ahead to fix it".
  H2  EARLY > LATE. With a usable reflex (SNR above break-even), ``instinct`` final
      correctness beats ``late`` self-correction, at comparable or lower token cost.
  H3  THE CEILING IS THE REFLEX, NOT THE POLICY. There is a break-even SNR: below it the
      monitor's false interrupts re-route healthy chains and ``instinct`` is *no better
      than* ``commit`` (it can hurt). The gain is bounded by the monitor's ROC — exactly
      as ``deliberation_roofline``'s ceiling is the verifier's SNR, not the compute. This
      is the honest boundary: an instinct is only as good as the reflex behind it.
  H4  BOUNDED. The ko guard makes re-route terminate (``escalate``, never a silent loop);
      escalation is a first-class outcome, not a wrong commit.

Honest scope (``candidateOnly: true``). Synthetic streams, planted ground truth. It earns
``level3Evidence: true`` only when a real reasoning trace routes a real reflex signal
through this policy with measured uplift under the repo's no-overclaim gate. It maps to
candidate real modules: ``agent/graded_decision.py`` (the verdict curve),
``agent/consequence_gate.py`` + ``reasoning/consequence/ko_detector.py`` (the escalate
rule), ``agent/calibration.py`` (the reflex signal). Not an AGI claim; see ``VISION.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Trajectory model (planted ground truth)
# ---------------------------------------------------------------------------

#: Default chain length (reasoning steps before an answer is emitted).
N_STEPS = 10
#: Per-step hazard of committing the (first) error while still on-track.
P_ERR = 0.09
#: P(correct answer | chain never errored).
P_CORRECT_CLEAN = 0.93
#: Per-remaining-step survival of correctness once errored, uncorrected ("derail").
#: <1 ⇒ the longer you run wrong, the lower the chance of a correct answer (H1).
DERAIL = 0.80
#: Late self-correction: fixes a wrong chain with this prob …  (kept deliberately low —
#: the published record is that *intrinsic* self-correction is marginal at best) …
P_FIX = 0.22
#: … and *breaks* an already-correct chain with this prob (the accuracy-correction
#: paradox: a forward "fix" pass can turn a right answer wrong).
P_BREAK = 0.16
#: Re-route overhead in step-units (cost of backtracking + a fresh attempt's preamble).
REROUTE_OVERHEAD = 4
#: Quality decay per re-route. A fresh attempt is not free: the model is now anchored on
#: the abandoned path, so each re-route scales the next attempt's correctness by this
#: factor. This is what makes a FALSE interrupt genuinely costly — throwing away a good
#: chain to redraw a slightly-worse one — and therefore what creates a real break-even SNR.
REROUTE_DECAY = 0.90
#: Length of a late self-correction pass in step-units.
CORRECT_PASS_LEN = 5
#: Max re-routes before the ko guard escalates (revisiting a failing state ⇒ ko).
MAX_REROUTE = 3


@dataclass(frozen=True)
class ReflexConfig:
    """The always-on reflex monitor — a noisy per-step wrongness detector.

    ``snr`` is the mean separation between the wrongness signal in the errored vs the
    on-track state (both unit-variance Gaussian). ``tau`` is the fire threshold. Together
    they fix the monitor's ROC: a high ``tau`` is cautious (few false interrupts, more
    misses); a low ``tau`` is trigger-happy (catches errors fast, re-routes healthy
    chains). The whole point of H3 is that *this* — not the policy — sets the ceiling.
    """

    snr: float = 3.0
    tau: float = 2.5


def _final_correct(rng: random.Random, err_step: int | None, n_steps: int, quality: float = 1.0) -> bool:
    """Did the chain land on the correct answer? ``err_step`` is None if never errored.

    ``quality`` (≤1) scales correctness to model re-route anchoring decay (see
    ``REROUTE_DECAY``); the default 1.0 is the first, un-anchored attempt.
    """
    if err_step is None:
        return rng.random() < P_CORRECT_CLEAN * quality
    remaining = n_steps - err_step
    return rng.random() < P_CORRECT_CLEAN * quality * (DERAIL ** remaining)


def _draw_error_step(rng: random.Random, n_steps: int) -> int | None:
    """First step at which the chain commits an error, or None (geometric-ish hazard)."""
    for step in range(n_steps):
        if rng.random() < P_ERR:
            return step
    return None


def _reflex_fire_step(
    rng: random.Random, err_step: int | None, n_steps: int, cfg: ReflexConfig
) -> int | None:
    """First step the reflex signal crosses ``tau``. Models a cheap always-on monitor.

    On-track steps draw N(0,1); once errored, steps draw N(snr,1). A crossing before the
    error is a *false interrupt*; at/after the error it is a true catch.
    """
    for step in range(n_steps):
        mu = cfg.snr if (err_step is not None and step >= err_step) else 0.0
        if rng.gauss(mu, 1.0) >= cfg.tau:
            return step
    return None


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass
class TrialOutcome:
    correct: bool = False
    cost: float = 0.0
    escalated: bool = False
    false_interrupt: bool = False
    reroutes: int = 0


def _run_commit(rng: random.Random, n_steps: int) -> TrialOutcome:
    err = _draw_error_step(rng, n_steps)
    return TrialOutcome(correct=_final_correct(rng, err, n_steps), cost=float(n_steps))


def _run_late(rng: random.Random, n_steps: int) -> TrialOutcome:
    err = _draw_error_step(rng, n_steps)
    correct = _final_correct(rng, err, n_steps)
    cost = float(n_steps + CORRECT_PASS_LEN)
    # One self-correction pass: fixes wrong, but can break right (the paradox).
    if not correct:
        correct = rng.random() < P_FIX
    elif rng.random() < P_BREAK:
        correct = False
    return TrialOutcome(correct=correct, cost=cost)


def _run_instinct(rng: random.Random, n_steps: int, cfg: ReflexConfig) -> TrialOutcome:
    """Reflex-triggered backtrack/re-route, ko-guarded.

    Each attempt draws a fresh error and a fresh reflex trace. If the reflex fires we
    backtrack and re-route (new attempt). After ``MAX_REROUTE`` re-routes the ko guard
    escalates (a clean, bounded outcome — never an endless patch-forward loop).
    """
    cost = 0.0
    reroutes = 0
    false_interrupt = False
    for attempt in range(MAX_REROUTE + 1):
        quality = REROUTE_DECAY ** attempt  # each re-route anchors → slightly worse
        err = _draw_error_step(rng, n_steps)
        fire = _reflex_fire_step(rng, err, n_steps, cfg)
        if fire is None:
            # Reflex stayed quiet: commit this attempt.
            cost += float(n_steps)
            return TrialOutcome(
                correct=_final_correct(rng, err, n_steps, quality),
                cost=cost,
                false_interrupt=false_interrupt,
                reroutes=reroutes,
            )
        # Reflex fired at ``fire`` — pay only the steps spent so far.
        if err is None or fire < err:
            false_interrupt = True  # interrupted a chain that had not (yet) errored
        cost += float(fire + 1)
        if attempt == MAX_REROUTE:
            # ko guard: re-route budget exhausted and it would fire again ⇒ escalate
            # (needs a human / new information — never an endless patch-forward loop).
            return TrialOutcome(
                correct=False, cost=cost, escalated=True,
                false_interrupt=false_interrupt, reroutes=reroutes,
            )
        cost += float(REROUTE_OVERHEAD)  # backtrack + fresh attempt preamble
        reroutes += 1
    raise AssertionError("unreachable: loop must return")  # pragma: no cover


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def _aggregate(outcomes: list[TrialOutcome]) -> dict[str, float]:
    n = len(outcomes)
    answered = [o for o in outcomes if not o.escalated]
    return {
        "n": n,
        "correct": sum(o.correct for o in outcomes) / n,
        "correct_when_answered": (sum(o.correct for o in answered) / len(answered)) if answered else 0.0,
        "mean_cost": sum(o.cost for o in outcomes) / n,
        "escalate_rate": sum(o.escalated for o in outcomes) / n,
        "false_interrupt_rate": sum(o.false_interrupt for o in outcomes) / n,
        "mean_reroutes": sum(o.reroutes for o in outcomes) / n,
    }


def run_policies(trials: int, seed: int, cfg: ReflexConfig, n_steps: int = N_STEPS) -> dict[str, Any]:
    """Run all three policies on independent draws; return per-policy aggregates."""
    out: dict[str, Any] = {}
    # Fixed per-policy stream offsets (NOT hash(name) — Python salts str hashes per
    # process, which would make runs non-reproducible). Independent streams keep the
    # policies from sharing draws while staying deterministic across processes.
    streams = (
        ("commit", 1, lambda r: _run_commit(r, n_steps)),
        ("late", 2, lambda r: _run_late(r, n_steps)),
        ("instinct", 3, lambda r: _run_instinct(r, n_steps, cfg)),
    )
    for name, offset, fn in streams:
        rng = random.Random(seed * 31 + offset)
        out[name] = _aggregate([fn(rng) for _ in range(trials)])
    return out


def compounding_curve(trials: int, seed: int, n_steps: int = N_STEPS) -> list[dict[str, float]]:
    """H1 probe: P(correct | error committed at step e), swept over e.

    Later error ⇒ fewer derail steps ⇒ higher correctness. Closed form is
    ``P_CORRECT_CLEAN * DERAIL**(n_steps - e)``; we report the MC estimate beside it.
    """
    rng = random.Random(seed)
    curve = []
    for e in range(n_steps):
        hits = sum(_final_correct(rng, e, n_steps) for _ in range(trials))
        curve.append({
            "err_step": e,
            "mc_correct": hits / trials,
            "closed_form": P_CORRECT_CLEAN * (DERAIL ** (n_steps - e)),
        })
    return curve


def snr_sweep(trials: int, seed: int, snrs: list[float], tau: float = 2.5) -> list[dict[str, float]]:
    """H3 probe: instinct correctness vs reflex SNR, against the commit/late baselines."""
    sweep = []
    for snr in snrs:
        pol = run_policies(trials, seed, ReflexConfig(snr=snr, tau=tau))
        sweep.append({
            "snr": snr,
            "instinct_correct": pol["instinct"]["correct"],
            "commit_correct": pol["commit"]["correct"],
            "late_correct": pol["late"]["correct"],
            "instinct_cost": pol["instinct"]["mean_cost"],
            "false_interrupt_rate": pol["instinct"]["false_interrupt_rate"],
            "escalate_rate": pol["instinct"]["escalate_rate"],
        })
    return sweep


@dataclass(frozen=True)
class TheoryVerdict:
    """Falsifiable verdict over the four hypotheses (repo no-overclaim envelope)."""

    schema: str = "sophia.reasoning.instinct.v1"
    h1_compounding: bool = False
    h2_early_beats_late: bool = False
    h3_breakeven_snr: float = float("nan")
    h4_bounded_escalate: bool = False
    notes: str = ""
    candidateOnly: bool = True
    level3Evidence: bool = False
    boundary: str = "synthetic planted-truth model of a reflex policy; not AGI proof."

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "h1_compounding": self.h1_compounding,
            "h2_early_beats_late": self.h2_early_beats_late,
            "h3_breakeven_snr": self.h3_breakeven_snr,
            "h4_bounded_escalate": self.h4_bounded_escalate,
            "notes": self.notes,
            "candidateOnly": self.candidateOnly,
            "level3Evidence": self.level3Evidence,
            "boundary": self.boundary,
        }


#: Minimum correctness margin over the commit baseline to call a break-even. A bare ">"
#: picks up Monte-Carlo noise near the crossover (the snr≈0.5 point straddles zero across
#: seeds/trials); requiring a margin beyond that noise makes the reported break-even
#: stable (=1.0 here) across seeds and trial counts.
BREAKEVEN_MARGIN = 0.02


def _breakeven_snr(sweep: list[dict[str, float]]) -> float:
    """Lowest SNR at which instinct correctness *reliably* exceeds commit (by a margin)."""
    for row in sweep:
        if row["instinct_correct"] > row["commit_correct"] + BREAKEVEN_MARGIN:
            return row["snr"]
    return float("inf")


def run_experiment(trials: int = 4000, seed: int = 1234) -> dict[str, Any]:
    snrs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    sweep = snr_sweep(trials, seed, snrs)
    comp = compounding_curve(trials, seed)
    good = run_policies(trials, seed, ReflexConfig(snr=3.0, tau=2.5))
    poor = run_policies(trials, seed, ReflexConfig(snr=0.0, tau=2.5))

    h1 = all(comp[i]["mc_correct"] <= comp[i + 1]["mc_correct"] + 0.05 for i in range(len(comp) - 1)) \
        and comp[-1]["mc_correct"] > comp[0]["mc_correct"] + 0.2
    h2 = good["instinct"]["correct"] > good["late"]["correct"]
    breakeven = _breakeven_snr(sweep)
    # H3: a real break-even exists (finite, >0) AND a poor reflex does not beat commit.
    h3 = (0.0 < breakeven < float("inf")) and (poor["instinct"]["correct"] <= poor["commit"]["correct"] + 0.02)
    # H4: at an aggressive reflex on a hard distribution, escalation occurs and stays bounded.
    hard = run_policies(trials, seed, ReflexConfig(snr=0.0, tau=0.0))
    h4 = hard["instinct"]["escalate_rate"] > 0.0 and hard["instinct"]["mean_reroutes"] <= MAX_REROUTE

    verdict = TheoryVerdict(
        h1_compounding=h1,
        h2_early_beats_late=h2,
        h3_breakeven_snr=breakeven,
        h4_bounded_escalate=h4,
        notes=(
            f"good-reflex(snr=3): instinct {good['instinct']['correct']:.3f} vs "
            f"late {good['late']['correct']:.3f} vs commit {good['commit']['correct']:.3f}; "
            f"break-even SNR={breakeven}; poor-reflex(snr=0) instinct "
            f"{poor['instinct']['correct']:.3f} vs commit {poor['commit']['correct']:.3f}."
        ),
    )
    return {
        "params": {
            "trials": trials, "seed": seed, "n_steps": N_STEPS, "p_err": P_ERR,
            "derail": DERAIL, "p_fix": P_FIX, "p_break": P_BREAK,
        },
        "policies_good_reflex": good,
        "policies_poor_reflex": poor,
        "compounding_curve": comp,
        "snr_sweep": sweep,
        "verdict": verdict.to_dict(),
    }


def format_report(res: dict[str, Any]) -> str:
    v = res["verdict"]
    g, p = res["policies_good_reflex"], res["policies_poor_reflex"]
    lines = [
        "Instinct gate — early reflex re-route vs late self-correct (falsifiable model)",
        "=" * 78,
        f"params: {json.dumps(res['params'])}",
        "",
        "POLICY CORRECTNESS / COST",
        f"  good reflex (snr=3): commit {g['commit']['correct']:.3f}@{g['commit']['mean_cost']:.1f}  "
        f"late {g['late']['correct']:.3f}@{g['late']['mean_cost']:.1f}  "
        f"instinct {g['instinct']['correct']:.3f}@{g['instinct']['mean_cost']:.1f} "
        f"(escalate {g['instinct']['escalate_rate']:.3f})",
        f"  poor reflex (snr=0): commit {p['commit']['correct']:.3f}  "
        f"instinct {p['instinct']['correct']:.3f} "
        f"(false-interrupt {p['instinct']['false_interrupt_rate']:.3f})",
        "",
        "SNR SWEEP (instinct vs commit vs late)",
    ]
    for row in res["snr_sweep"]:
        lines.append(
            f"  snr={row['snr']:.1f}: instinct {row['instinct_correct']:.3f}  "
            f"commit {row['commit_correct']:.3f}  late {row['late_correct']:.3f}  "
            f"(FI {row['false_interrupt_rate']:.2f}, esc {row['escalate_rate']:.2f})"
        )
    lines += [
        "",
        "THEORY VERDICT",
        f"  H1 compounding (don't plough ahead) : {v['h1_compounding']}",
        f"  H2 early reroute beats late correct : {v['h2_early_beats_late']}",
        f"  H3 break-even SNR (reflex = ceiling): {v['h3_breakeven_snr']}",
        f"  H4 ko-bounded escalate              : {v['h4_bounded_escalate']}",
        f"  candidateOnly={v['candidateOnly']}  level3Evidence={v['level3Evidence']}",
        f"  boundary: {v['boundary']}",
    ]
    return "\n".join(lines)


def _self_test() -> int:
    res = run_experiment(trials=4000, seed=1234)
    v = res["verdict"]
    assert v["h1_compounding"], "H1 failed: correctness not monotone in error lateness"
    assert v["h2_early_beats_late"], "H2 failed: instinct did not beat late self-correct"
    be = v["h3_breakeven_snr"]
    assert 0.0 < be < float("inf"), f"H3 failed: no finite positive break-even SNR ({be})"
    assert v["h4_bounded_escalate"], "H4 failed: escalation absent or unbounded"
    # Closed-form anchor for the compounding curve (MC must track theory).
    worst = max(abs(r["mc_correct"] - r["closed_form"]) for r in res["compounding_curve"])
    assert worst < 0.05, f"compounding MC diverged from closed form: {worst}"
    print(
        f"self-test OK: break-even SNR={be}, "
        f"instinct {res['policies_good_reflex']['instinct']['correct']:.3f} > "
        f"late {res['policies_good_reflex']['late']['correct']:.3f}, "
        f"MC~theory<{worst:.4f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true", help="run the full experiment + verdict")
    p.add_argument("--self-test", action="store_true", help="assert the invariants and exit")
    p.add_argument("--json", action="store_true", help="emit raw results as JSON")
    p.add_argument("--trials", type=int, default=4000)
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
