# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Model↔harness closed-loop co-evolution — the engine Phase A closes.

Composes three existing, independently-tested modules into the ONE cycle the
DeepSeek "Model + Harness = Agent" thesis actually turns on:

    measure uplift  (harness vs bare, same external verifier, paired + bootstrapped)
      -> the harness run logs carry fail-then-fix traces
    distill those traces into preference pairs  (DPO signal for the next model)
      -> a train step consumes them and produces a candidate checkpoint
    gate the candidate through continual_plasticity  (hard reject on regression /
       contamination / catastrophic forgetting; promote only on a clean gain)
      -> re-measure uplift with the promoted model

Two invariants make this the honest signature of a *closing* loop rather than a
designed one:

  * NON-DEGENERACY — a promoted model's uplift never goes negative. The plasticity
    gate enforces this per-cycle (regression => reject); this module asserts it
    across the whole run and HALTS LOUD if it ever fires, because a negative
    uplift *after a promotion* means reward hacking / regression slipped past the
    gate — a loop failure, not a data point to publish.
  * SATURATION IS SUCCESS — if uplift converges to ~0 the harness has been
    distilled into the model (the model now does first-try what the harness used
    to rescue). That is a valid terminal state, not a failure; the next move is a
    HARDER harness (Phase C world model), not more of the same loop.

Honest scope: the train step is INJECTED. In CI it is a no-op (``ran=False``) so
the loop is exercised end-to-end without a GPU; the live GRPO/DPO step
(``tools/run_rlvr.py`` on a CUDA pod, consuming the distilled preference JSONL)
is the seam. This module is the orchestrator + the invariants; it trains nothing
itself and changes no weights.

Discipline (Sophia, preserved): every report carries ``candidateOnly=True`` and
``level3Evidence=False`` — closing the loop offline is rehearsal, not Level-3
evidence. Real Level-3 evidence needs a private hidden suite and a gated run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent import harness as h
from agent import trace_distill as td
from agent import uplift as up
from agent.continual_plasticity import (
    EvalMetric,
    UpdateCandidate,
    evaluate_update,
)
from agent.model import ModelClient

# A train step turns (cycle_index, distilled_pairs, current_model_spec) into a
# candidate outcome. Injected so CI runs the loop without a GPU and the live
# recipe shells out to tools/run_rlvr.py / a DPO trainer on a CUDA pod.
TrainStep = Callable[[int, "list[td.PreferencePair]", str], "TrainOutcome"]

# A client factory builds the ModelClient for a given model spec string, so the
# loop can point at the freshly-trained checkpoint after a promotion (and at the
# base model before any training).
ClientFactory = Callable[[str], ModelClient]


@dataclass(frozen=True)
class TrainOutcome:
    """Result of one train step. ``ran=False`` means no training happened (mock /
    CI no-op, or no distillable pairs) — the loop then has nothing to gate."""

    new_spec: str
    ran: bool
    artifact: str = ""  # checkpoint path / id, for the audit trail
    notes: str = ""


@dataclass(frozen=True)
class CycleReport:
    cycle: int
    model_spec: str
    baseline_uplift: float
    baseline_harness_rate: float
    baseline_demonstrated: bool
    pairs_distilled: int
    train_ran: bool
    candidate_spec: str
    promotion_verdict: str  # promote | quarantine | reject | no-candidate
    promotion_reasons: tuple[str, ...] = ()
    post_uplift: float = 0.0
    post_harness_rate: float = 0.0
    model_advanced: bool = False  # True iff a candidate was promoted this cycle

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "modelSpec": self.model_spec,
            "baselineUplift": round(self.baseline_uplift, 4),
            "baselineHarnessRate": round(self.baseline_harness_rate, 4),
            "baselineDemonstrated": self.baseline_demonstrated,
            "pairsDistilled": self.pairs_distilled,
            "trainRan": self.train_ran,
            "candidateSpec": self.candidate_spec,
            "promotionVerdict": self.promotion_verdict,
            "promotionReasons": list(self.promotion_reasons),
            "postUplift": round(self.post_uplift, 4),
            "postHarnessRate": round(self.post_harness_rate, 4),
            "modelAdvanced": self.model_advanced,
        }


@dataclass(frozen=True)
class LoopReport:
    schema: str
    candidateOnly: bool
    level3Evidence: bool
    initial_uplift: float
    final_uplift: float
    final_model_spec: str
    cycles: tuple[CycleReport, ...] = ()
    non_degenerate: bool = True  # uplift never went negative after a promotion
    saturated: bool = False  # uplift converged to ~0 AFTER a promotion (valid)
    promoted_any: bool = False  # at least one cycle promoted a candidate
    halted_early: bool = False
    halt_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidateOnly": self.candidateOnly,
            "level3Evidence": self.level3Evidence,
            "initialUplift": round(self.initial_uplift, 4),
            "finalUplift": round(self.final_uplift, 4),
            "finalModelSpec": self.final_model_spec,
            "cycles": [c.to_dict() for c in self.cycles],
            "invariants": {
                "nonDegenerate": self.non_degenerate,
                "saturated": self.saturated,
                "promotedAny": self.promoted_any,
            },
            "haltedEarly": self.halted_early,
            "haltReason": self.halt_reason,
            "interpretation": _interpretation(self),
        }


def _interpretation(report: "LoopReport") -> str:
    if not report.non_degenerate:
        return (
            "HALTED: a promoted model's uplift went negative — reward hacking or "
            "regression slipped past the plasticity gate. The loop is degenerate; "
            "do not publish. Diagnose the gate before resuming."
        )
    if report.halted_early:
        return f"halted early: {report.halt_reason}"
    if report.saturated:
        return (
            "SATURATED (success): uplift converged to ~0 after a promotion — the "
            "harness's rescue behaviour was distilled into the model. The loop did "
            "its job; the next competence gain needs a HARDER harness (a learned "
            "world model / planner), not more of the same distillation."
        )
    if not report.promoted_any:
        return (
            "IDLE: no candidate was promoted (no training ran, or every candidate "
            "was gated). The loop ran end-to-end without error but produced no "
            "model change. For a real run, point train_step at a live GPU trainer."
        )
    return (
        "The loop closed: uplift was measured, traces distilled, a candidate "
        "trained, gated, and promoted on a clean gain, and re-measured. Repeat for "
        "more cycles until saturation."
    )


def noop_train_step(_cycle: int, pairs: "list[td.PreferencePair]", current_spec: str) -> TrainOutcome:
    """Default train step for CI / dry runs: trains nothing. Keeps the loop
    exercisable end-to-end without a GPU; every cycle reports ``no-candidate``."""
    return TrainOutcome(new_spec=current_spec, ran=False, notes="noop (CI)")


def _build_verifier_for(task_spec_for: "Callable[[dict], Any] | None"):
    """Construct the ``verifier_for`` closure for ``uplift.measure_uplift``.

    When ``task_spec_for`` is given (the Phase B wedge), each case is routed to
    its execution verifier via ``execution_verifiers.verifier_for_task``; cases
    with no executable ground truth fall through to uplift's default verifier
    (the epistemic gate / mustInclude). Returns ``None`` (uplift's default) when
    no routing is configured, preserving prior behavior exactly.

    NB: the returned closure MUST always yield a callable verifier —
    ``measure_uplift`` calls ``verifier_for(case)`` then ``verifier(...)`` — so we
    fall back to ``uplift._default_verifier_for`` (never return None) when a case
    has no execution ground truth.
    """
    if task_spec_for is None:
        return None
    from agent.execution_verifiers import verifier_for_task
    from agent.uplift import _default_verifier_for

    def verifier_for(case: dict):
        spec = task_spec_for(case)
        if spec is not None:
            v = verifier_for_task(spec)
            if v is not None:
                return v  # route to execution truth
        return _default_verifier_for(case)  # fall back to gate / mustInclude

    return verifier_for


def run_closed_loop(
    suite: list[dict[str, Any]],
    *,
    suite_name: str,
    make_client: ClientFactory,
    initial_spec: str,
    train_step: TrainStep,
    runs_root: Path,
    max_cycles: int = 2,
    max_retries: int = 2,
    min_target_delta: float = 0.0,
    saturation_eps: float = 0.02,
    bootstrap_seed: int = 0,
    task_spec_for: "Callable[[dict], Any] | None" = None,
) -> LoopReport:
    """Run up to ``max_cycles`` distill -> train -> gate -> re-measure cycles.

    ``suite`` items follow ``agent.uplift`` shape: ``{"id", "goal", "mode"?,
    "mustInclude"?}``. Each cycle (a) measures uplift with the current model,
    which also writes the harness fail-then-fix traces to a per-cycle run dir;
    (b) distills those traces into preference pairs; (c) asks the injected
    ``train_step`` for a candidate; (d) gates it through ``continual_plasticity``
    on the harness pass-rate delta; (e) promotes on a clean gain and re-measures,
    or holds the baseline.

    ``task_spec_for`` (the Phase B wedge): maps a suite case to a
    :class:`~agent.execution_verifiers.TaskSpec`; when set, each case's verifier is
    routed through ``execution_verifiers.verifier_for_task`` — so a coding case is
    graded by ``code_tests_pass`` (execution truth) rather than the epistemic gate
    (citation truth). ``None`` keeps the gate default. This is how execution
    verification becomes FIRST-CLASS in the loop instead of every caller
    hand-wiring its verifier.

    Returns a ``LoopReport`` whose ``non_degenerate`` flag is the load-bearing
    invariant: False means a promoted model regressed and the loop halted.
    """
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    current_spec = initial_spec
    cycles: list[CycleReport] = []
    uplift_history: list[float] = []
    non_degenerate = True
    halted = False
    halt_reason = ""
    saved_runs_dir = h.RUNS_DIR  # restore on exit so we never leak the override

    # Phase B: build a verifier_for that routes cases to execution verifiers when
    # task_spec_for is provided, else fall back to uplift's default (epistemic gate).
    verifier_for = _build_verifier_for(task_spec_for)

    try:
        for cycle in range(max_cycles):
            cycle_runs = runs_root / f"cycle-{cycle}"
            cycle_runs.mkdir(parents=True, exist_ok=True)
            h.RUNS_DIR = cycle_runs  # route this cycle's harness traces here

            client = make_client(current_spec)
            baseline = up.measure_uplift(
                suite, client=client, max_retries=max_retries,
                bootstrap_seed=bootstrap_seed, verifier_for=verifier_for,
            )
            pairs = td.distill_dir(cycle_runs)
            outcome = train_step(cycle, pairs, current_spec)

            verdict = "no-candidate"
            reasons: tuple[str, ...] = ()
            post_uplift = baseline.uplift
            post_rate = baseline.harness_pass_rate
            advanced = False

            candidate_has_new_spec = outcome.ran and bool(outcome.new_spec) and outcome.new_spec != current_spec
            if candidate_has_new_spec:
                new_client = make_client(outcome.new_spec)
                cand_runs = runs_root / f"cycle-{cycle}-candidate"
                cand_runs.mkdir(parents=True, exist_ok=True)
                h.RUNS_DIR = cand_runs  # keep candidate traces separate from baseline
                post = up.measure_uplift(
                    suite, client=new_client, max_retries=max_retries,
                    bootstrap_seed=bootstrap_seed, verifier_for=verifier_for,
                )
                candidate = UpdateCandidate(
                    id=outcome.new_spec,
                    kind="lora_adapter",
                    verifier_artifacts=(
                        "uplift-paired-bootstrap-ci",
                        "trace-distill-gated-pairs",
                    ),
                    metrics=(
                        EvalMetric(
                            suite=suite_name,
                            before=baseline.harness_pass_rate,
                            after=post.harness_pass_rate,
                        ),
                    ),
                    notes=outcome.notes,
                )
                decision = evaluate_update(
                    candidate,
                    target_suite=suite_name,
                    min_target_delta=min_target_delta,
                )
                verdict = decision.verdict
                reasons = decision.reasons
                post_uplift = post.uplift
                post_rate = post.harness_pass_rate

                if verdict == "promote":
                    # NON-DEGENERACY, defense-in-depth: even with the gate cleared,
                    # a promoted model whose *uplift* (vs bare) went negative is a
                    # loop failure. The gate checks harness pass-rate delta; this
                    # checks the harness-vs-bare relationship the thesis depends on.
                    if post.uplift < 0:
                        non_degenerate = False
                        halted = True
                        halt_reason = (
                            f"promoted model at cycle {cycle} has negative uplift "
                            f"({post.uplift:.4f}) — reward hacking / regression past the gate; rolled back"
                        )
                        # do NOT advance current_spec (rollback)
                    else:
                        current_spec = outcome.new_spec
                        advanced = True
            elif outcome.ran:
                reasons = ("train step ran but produced no new spec",)

            cycles.append(
                CycleReport(
                    cycle=cycle,
                    model_spec=current_spec,
                    baseline_uplift=baseline.uplift,
                    baseline_harness_rate=baseline.harness_pass_rate,
                    baseline_demonstrated=baseline.demonstrated,
                    pairs_distilled=len(pairs),
                    train_ran=outcome.ran,
                    candidate_spec=outcome.new_spec,
                    promotion_verdict=verdict,
                    promotion_reasons=reasons,
                    post_uplift=post_uplift,
                    post_harness_rate=post_rate,
                    model_advanced=advanced,
                )
            )
            uplift_history.append(post_uplift)
            if halted:
                break
    finally:
        h.RUNS_DIR = saved_runs_dir  # never leak the per-cycle override

    promoted_any = any(c.model_advanced for c in cycles)
    initial_uplift = cycles[0].baseline_uplift if cycles else 0.0
    final_uplift = uplift_history[-1] if uplift_history else 0.0
    # SATURATION: a promotion happened AND uplift collapsed to ~0 — the harness
    # was distilled into the model. Valid terminal; distinct from idle (no
    # promotion) and from degenerate (negative uplift).
    saturated = promoted_any and abs(final_uplift) < saturation_eps

    return LoopReport(
        schema="sophia.closed_loop.v1",
        candidateOnly=True,
        level3Evidence=False,
        initial_uplift=initial_uplift,
        final_uplift=final_uplift,
        final_model_spec=current_spec,
        cycles=tuple(cycles),
        non_degenerate=non_degenerate,
        saturated=saturated,
        promoted_any=promoted_any,
        halted_early=halted,
        halt_reason=halt_reason,
    )


def write_report(report: LoopReport, out: str | Path) -> dict[str, Any]:
    """Persist a loop report as JSON (candidate artifact, never Level-3 evidence)."""
    payload = report.to_dict()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "TrainStep",
    "ClientFactory",
    "TrainOutcome",
    "CycleReport",
    "LoopReport",
    "noop_train_step",
    "run_closed_loop",
    "write_report",
]
