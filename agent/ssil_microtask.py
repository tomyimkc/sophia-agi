"""Provenance-routing micro-eval — a real, measurable held-out task for SSIL G4.

The point: give the plasticity gate (G4) something *measured* to promote on, so a
live proposal can earn a genuine improvement instead of always quarantining. The
task is deliberately small and Goodhart-resistant:

  - A candidate proposes an EXECUTABLE policy (thresholds), not prose. We run it
    against gold labels it never sees. No trusting the model's self-report.
  - Headline metric = accuracy on the held-out TEST split.
  - PROTECTED metric = recall on truly-answerable cases. A degenerate "always
    abstain" policy scores ~0.5 headline but ~0.0 protected -> G4 rejects it. You
    cannot win by refusing to answer.

The proposer sees only the TRAIN split's feature ranges (never TEST labels), so a
measured TEST gain is not leakage.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, "") and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow direct `python3 agent/ssil_microtask.py`

from agent.continual_plasticity import EvalMetric

_DATA = Path(__file__).resolve().parents[1] / "eval" / "ssil_microtask" / "provenance_routing.v1.jsonl"


@dataclass(frozen=True)
class PolicySpec:
    """An executable routing policy: answer iff sources & quality clear thresholds."""

    min_sources: int
    min_quality: float
    default_action: str = "abstain"  # action when thresholds are NOT met

    def action(self, sources: int, quality: float) -> str:
        if sources >= self.min_sources and quality >= self.min_quality:
            return "answer"
        return self.default_action

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicySpec":
        return cls(
            min_sources=int(d.get("min_sources", 0)),
            min_quality=float(d.get("min_quality", 0.0)),
            default_action=str(d.get("default_action", "abstain")),
        )


def load_cases(split: str | None = None, *, path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path is not None else _DATA
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if split is None or r["split"] == split]


def baseline_spec() -> PolicySpec:
    """Weak baseline: answer everything (min thresholds). Scores ~class balance."""
    return PolicySpec(min_sources=0, min_quality=0.0, default_action="answer")


def _accuracy(spec: PolicySpec, cases: list[dict]) -> float:
    if not cases:
        return 0.0
    hit = sum(spec.action(c["independent_sources"], c["source_quality"]) == c["gold"] for c in cases)
    return round(hit / len(cases), 4)


def _answer_recall(spec: PolicySpec, cases: list[dict]) -> float:
    """Protected metric: of cases whose gold is 'answer', how many does the policy
    actually answer? An always-abstain policy scores 0 here."""
    answerable = [c for c in cases if c["gold"] == "answer"]
    if not answerable:
        return 1.0
    hit = sum(spec.action(c["independent_sources"], c["source_quality"]) == "answer" for c in answerable)
    return round(hit / len(answerable), 4)


# Absolute floor for the protected answer-recall metric. A policy must answer at
# least this fraction of genuinely-answerable cases. Encoded as the protected
# metric's "before" so the plasticity gate's delta check becomes a floor check:
# candidate_recall - FLOOR < 0  => protected regression => reject. This punishes
# degenerate "always abstain" without penalizing healthy selective abstention
# (the always-answer baseline's recall is trivially 1.0 and is NOT the reference).
ANSWER_RECALL_FLOOR = 0.5


def measure_policy(candidate: PolicySpec, *, path: str | Path | None = None) -> tuple[tuple[EvalMetric, ...], dict[str, Any]]:
    """Measure candidate vs baseline on the held-out TEST split.

    Returns (metrics, detail). Metrics feed G4: a headline 'routing_accuracy' suite
    (improvement over the weak baseline) plus a PROTECTED 'answer_recall' suite held
    to an absolute floor (degenerate always-abstain scores 0 and is rejected).
    """
    test = load_cases("test", path=path)
    base = baseline_spec()
    cand_recall = _answer_recall(candidate, test)
    metrics = (
        EvalMetric("routing_accuracy", _accuracy(base, test), _accuracy(candidate, test), protected=False),
        EvalMetric("answer_recall", ANSWER_RECALL_FLOOR, cand_recall, protected=True),
    )
    detail = {
        "testCases": len(test),
        "answerRecallFloor": ANSWER_RECALL_FLOOR,
        "baseline": {"accuracy": _accuracy(base, test), "answer_recall": _answer_recall(base, test)},
        "candidate": {"accuracy": _accuracy(candidate, test), "answer_recall": cand_recall,
                      "spec": candidate.__dict__},
    }
    return metrics, detail


def train_feature_summary(*, path: str | Path | None = None) -> dict[str, Any]:
    """What the proposer is allowed to see: TRAIN feature ranges, NOT test labels."""
    train = load_cases("train", path=path)
    src = sorted({c["independent_sources"] for c in train})
    q = [c["source_quality"] for c in train]
    return {
        "task": "Decide answer vs abstain from (independent_sources, source_quality).",
        "trainSize": len(train),
        "sourcesObserved": src,
        "qualityRange": [min(q), max(q)] if q else [0, 0],
        "hint": "Well-supported claims should be answered; weakly-sourced ones abstained.",
    }


def demo_microtask_report() -> dict[str, Any]:
    """Show baseline, a good discovered policy, and a degenerate always-abstain."""
    good = PolicySpec(min_sources=2, min_quality=0.6)
    degenerate = PolicySpec(min_sources=99, min_quality=1.1)  # never answers
    test = load_cases("test")
    return {
        "schema": "sophia.ssil_microtask_demo.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "baselineAccuracy": _accuracy(baseline_spec(), test),
        "goodPolicy": {"accuracy": _accuracy(good, test), "answer_recall": _answer_recall(good, test)},
        "degeneratePolicy": {"accuracy": _accuracy(degenerate, test), "answer_recall": _answer_recall(degenerate, test)},
        "invariants": {
            "good_beats_baseline": _accuracy(good, test) > _accuracy(baseline_spec(), test),
            "degenerate_tanks_protected_metric": _answer_recall(degenerate, test) == 0.0,
        },
    }


if __name__ == "__main__":
    print(json.dumps(demo_microtask_report(), ensure_ascii=False, indent=2))
