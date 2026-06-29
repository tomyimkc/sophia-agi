#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia-gated AutoResearch controller — the brakes and odometer for an autonomous loop.

Karpathy's ``autoresearch`` runs an agent overnight: edit ``train.py`` -> train 5 min ->
keep iff ``val_bpb`` dropped -> else ``git reset`` -> repeat. It is a powerful engine with
**no brakes and no odometer**: ``program.md`` itself adds *no* safeguard against overfitting,
validation-set leakage, or cheating, and treats the eval as ground truth the agent could edit.

This module is the missing trust layer. It keeps the autoresearch *architecture* (one editable
surface, fixed budget, one metric, git-as-trail, loop-until-interrupt) but replaces the greedy
point-estimate keep/discard with Sophia's discipline:

  1. **Reward-hacking firewall** — the agent may edit policy / data / hyperparameters, NEVER the
     verifier, gate, eval, reward, or constitution. A diff touching a protected path is rejected
     as tampering (the deontic "no reward/verifier tampering" rule, mechanised).
  2. **Evaluation isolation** — a result that failed decontamination (eval leakage) is discarded.
  3. **Power-gated improvement** — a change is kept only if the metric improves with a 95% CI
     that EXCLUDES zero on the improving side (not a single 5-minute number).
  4. **Protected-regression block** — religion / history (and any registered protected behaviour)
     must not regress, even for a metric win.
  5. **Honest trail** — every discard/reject yields a failure-ledger record (kept changes stay
     CANDIDATE until a real multi-seed run clears the project's κ≥0.40 / CI gate).

Pure-Python, deterministic, no torch, no GPU, no network: the controller is CI-testable; the GPU
training step plugs in behind it as the ``experiments`` iterator. canClaimAGI stays false.

    python tools/sophia_autoresearch.py --self-test
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Paths the optimiser must NEVER edit — editing what scores you is reward-hacking. Mirrors the
# constitution's reward/verifier-tampering prohibition and the repo's protected domains.
DEFAULT_PROTECTED_PATTERNS: "tuple[str, ...]" = (
    r"agent/.*verifier.*\.py",
    r"agent/gate\.py",
    r"agent/verifiers\.py",
    r"agent/benchmark_checks\.py",
    r"provenance_bench/",            # the reward definitions
    r"tools/claim_gate\.py",
    r"tools/eval_stats\.py",
    r"tools/assert_decontam\.py",
    r"tools/lint_claims\.py",
    r"constitution/",
    r".*/eval/.*\.jsonl",           # eval packs / validation sets
    r".*holdout.*\.jsonl",
    r".*heldout.*\.jsonl",
    r"data/religion_concepts\.json",  # PROTECTED domain
    r"data/history_events\.json",     # PROTECTED domain
)

Z95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True)
class Measurement:
    """A metric measured as PAIRED deltas (candidate − baseline) over seeds / held-out items —
    never a single number. ``lower_is_better`` sets the improving direction (val_bpb -> True)."""

    metric: str
    deltas: "tuple[float, ...]"
    lower_is_better: bool = True

    @property
    def n(self) -> int:
        return len(self.deltas)

    @property
    def mean(self) -> float:
        return sum(self.deltas) / self.n if self.n else 0.0

    def ci95(self) -> "tuple[float, float]":
        """Normal-approx 95% CI on the mean paired delta. Deterministic (no sampling)."""
        n = self.n
        if n < 2:
            return (self.mean, self.mean)
        m = self.mean
        var = sum((d - m) ** 2 for d in self.deltas) / (n - 1)
        se = math.sqrt(var / n)
        return (m - Z95 * se, m + Z95 * se)

    def improves(self) -> bool:
        """True iff the CI excludes zero ON THE IMPROVING SIDE (power-gated, not a point estimate)."""
        lo, hi = self.ci95()
        return hi < 0.0 if self.lower_is_better else lo > 0.0


@dataclass(frozen=True)
class Experiment:
    """One candidate change proposed by the autonomous agent."""

    experiment_id: str
    touched_files: "tuple[str, ...]"        # files the agent's diff changed
    measurement: Measurement
    decontaminated: bool = True             # result of tools/assert_decontam (supplied)
    protected_regressions: "tuple[str, ...]" = ()  # protected behaviours that regressed


@dataclass
class Decision:
    experiment_id: str
    verdict: str                            # "keep" | "discard" | "reject_tamper"
    reasons: "list[str]" = field(default_factory=list)
    improved: bool = False
    ci95: "tuple[float, float]" = (0.0, 0.0)
    ledger_entry: "dict | None" = None

    @property
    def kept(self) -> bool:
        return self.verdict == "keep"

    def to_dict(self) -> dict:
        return {
            "experimentId": self.experiment_id, "verdict": self.verdict,
            "reasons": list(self.reasons), "improved": self.improved,
            "ci95": [round(self.ci95[0], 5), round(self.ci95[1], 5)],
        }


def firewall_violations(touched_files: "Iterable[str]",
                        patterns: "Iterable[str]" = DEFAULT_PROTECTED_PATTERNS) -> "list[str]":
    """Files the optimiser is forbidden to edit (verifier / gate / eval / reward / protected)."""
    compiled = [re.compile(p) for p in patterns]
    return [f for f in touched_files if any(c.search(f) for c in compiled)]


def decide(exp: Experiment, *, patterns: "Iterable[str]" = DEFAULT_PROTECTED_PATTERNS) -> Decision:
    """Gated keep/discard. Firewall first (tampering is fatal), then leakage, improvement, regression."""
    reasons: list[str] = []
    ci = exp.measurement.ci95()
    improved = exp.measurement.improves()

    tamper = firewall_violations(exp.touched_files, patterns)
    if tamper:
        reasons.append(f"reward-hacking firewall: edited protected path(s) {tamper}")
        d = Decision(exp.experiment_id, "reject_tamper", reasons, improved, ci)
        d.ledger_entry = _ledger_entry(exp, d)
        return d

    if not exp.decontaminated:
        reasons.append("evaluation isolation: result failed decontamination (eval leakage)")
    if exp.protected_regressions:
        reasons.append(f"protected-regression: {list(exp.protected_regressions)} regressed")
    if not improved:
        reasons.append(f"no powered improvement: 95% CI {tuple(round(x, 4) for x in ci)} "
                       f"does not exclude zero on the improving side")

    verdict = "keep" if not reasons else "discard"
    if verdict == "keep":
        reasons.append(f"powered improvement: 95% CI {tuple(round(x, 4) for x in ci)} excludes zero; "
                       f"decontaminated; no protected regression (CANDIDATE until multi-seed gate)")
    d = Decision(exp.experiment_id, verdict, reasons, improved, ci)
    if verdict != "keep":
        d.ledger_entry = _ledger_entry(exp, d)
    return d


def _ledger_entry(exp: Experiment, d: Decision) -> dict:
    """A failure-ledger-shaped record for a discarded/rejected experiment (honest trail)."""
    return {
        "id": f"autoresearch-{exp.experiment_id}-{d.verdict}",
        "status": "OPEN",
        "verdict": d.verdict,
        "metric": exp.measurement.metric,
        "meanDelta": round(exp.measurement.mean, 5),
        "ci95": [round(d.ci95[0], 5), round(d.ci95[1], 5)],
        "reasons": list(d.reasons),
        "claimImpact": "none — change was not kept; recorded for the research trail",
    }


def run_loop(
    experiments: "Iterable[Experiment]",
    *,
    on_keep: "Callable[[Experiment, Decision], None] | None" = None,
    on_discard: "Callable[[Experiment, Decision], None] | None" = None,
    patterns: "Iterable[str]" = DEFAULT_PROTECTED_PATTERNS,
    max_iters: "int | None" = None,
) -> dict:
    """Drive the gated loop over an experiment stream (the GPU step is the iterator). ``on_keep``
    advances the git branch; ``on_discard`` resets it — both supplied by the caller. Loops until
    the stream ends, ``max_iters``, or KeyboardInterrupt (graceful stop, autoresearch-style)."""
    decisions: list[Decision] = []
    ledger: list[dict] = []
    kept = 0
    it: Iterator[Experiment] = iter(experiments)
    i = 0
    try:
        for exp in it:
            if max_iters is not None and i >= max_iters:
                break
            i += 1
            d = decide(exp, patterns=patterns)
            decisions.append(d)
            if d.ledger_entry:
                ledger.append(d.ledger_entry)
            if d.kept:
                kept += 1
                if on_keep:
                    on_keep(exp, d)
            elif on_discard:
                on_discard(exp, d)
    except KeyboardInterrupt:
        pass
    return {
        "evaluated": len(decisions),
        "kept": kept,
        "discarded": sum(1 for d in decisions if d.verdict == "discard"),
        "rejectedTamper": sum(1 for d in decisions if d.verdict == "reject_tamper"),
        "ledger": ledger,
        "decisions": [d.to_dict() for d in decisions],
    }


def offline_invariants() -> "tuple[bool, dict]":
    """Falsifiable, deterministic invariants for the gated controller (no GPU, no network)."""
    checks: dict[str, bool] = {}

    # 1. A genuine, powered improvement is kept.
    good = Experiment("genuine-win", ("train.py",),
                      Measurement("val_bpb", tuple([-0.05] * 12), lower_is_better=True))
    checks["genuine_win_kept"] = decide(good).verdict == "keep"

    # 2. An improvement that is NOT statistically separable from zero is discarded (anti-greedy).
    noisy = Experiment("noisy", ("train.py",),
                       Measurement("val_bpb", (-0.5, 0.4, -0.3, 0.45, -0.4, 0.5), lower_is_better=True))
    checks["noisy_point_estimate_discarded"] = decide(noisy).verdict == "discard"

    # 3. Editing a verifier/gate/eval path is rejected as tampering, even with a "great" metric.
    cheat = Experiment("cheater", ("agent/gate.py", "agent/math_verifier.py"),
                       Measurement("val_bpb", tuple([-0.9] * 12), lower_is_better=True))
    dc = decide(cheat)
    checks["tamper_rejected"] = dc.verdict == "reject_tamper" and not dc.kept

    # 4. Eval leakage (failed decontam) is discarded even with a powered win.
    leaky = Experiment("leaky", ("train.py",),
                       Measurement("val_bpb", tuple([-0.2] * 12), lower_is_better=True),
                       decontaminated=False)
    checks["leakage_discarded"] = decide(leaky).verdict == "discard"

    # 5. A protected-domain regression blocks the keep even on a metric win.
    regress = Experiment("regressor", ("train.py",),
                         Measurement("val_bpb", tuple([-0.2] * 12), lower_is_better=True),
                         protected_regressions=("religion-attribution",))
    checks["protected_regression_blocked"] = decide(regress).verdict == "discard"

    # 6. higher-is-better metric path works (e.g. verified hallucination-Δ).
    halluc = Experiment("halluc-up", ("agent/swarm_router.py",),
                        Measurement("verified_halluc_delta", tuple([0.08] * 12), lower_is_better=False))
    checks["higher_is_better_kept"] = decide(halluc).verdict == "keep"

    # 7. Every non-keep produces a ledger entry (honest trail); a keep does not.
    checks["discard_logs_ledger"] = decide(noisy).ledger_entry is not None
    checks["keep_no_ledger"] = decide(good).ledger_entry is None

    # 8. Loop accounting reconciles and the firewall count is surfaced.
    summary = run_loop([good, noisy, cheat, leaky, regress, halluc])
    checks["loop_reconciles"] = (
        summary["evaluated"] == 6
        and summary["kept"] == 2
        and summary["rejectedTamper"] == 1
        and summary["evaluated"] == summary["kept"] + summary["discarded"] + summary["rejectedTamper"]
    )

    ok = all(checks.values())
    return ok, {"checks": checks, "summary": run_loop([good, noisy, cheat, leaky, regress, halluc])}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the deterministic offline invariants")
    args = ap.parse_args(argv)
    if args.self_test:
        ok, detail = offline_invariants()
        print("Sophia-gated AutoResearch controller invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        s = detail["summary"]
        print(f"  loop: evaluated={s['evaluated']} kept={s['kept']} "
              f"discarded={s['discarded']} rejectedTamper={s['rejectedTamper']}")
        return 0 if ok else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
