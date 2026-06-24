#!/usr/bin/env python3
"""C1 promotion-gate tests: the real v2 adapter must auto-reject (reproducing the
failure-ledger decision), a clean improving adapter must promote, and the independent
formal protected-floor proof must agree."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_plasticity import evaluate_update  # noqa: E402
from tools.promote_adapter import (  # noqa: E402
    _formal_protected_floor_proof,
    _retention_from_shift_report,
    _rung,
    build_candidate,
)


def _ladder(base: dict[str, float], adapter: dict[str, float]) -> dict:
    def rung(name: str, doms: dict[str, float]) -> dict:
        passed = sum(round(v * 9) for v in doms.values())
        total = 9 * len(doms)
        return {
            "rung": name,
            "summary": {
                "domains": {k: {"score_pct": round(v * 100, 1)} for k, v in doms.items()},
                "score_pct": round(100 * passed / total, 1),
                "passed": passed,
                "total": total,
            },
        }
    return {"rungs": [rung("base", base), rung("adapter", adapter)]}


def test_v2_adapter_rejects_on_religion_regression() -> None:
    """Reproduce the hand-written v2 ledger reject (religion 1/6 -> 0/6).

    Built inline rather than read from `eval_ladder_adapter.json`: that artifact path
    is reused by each training run and was overwritten by the v3 convergence pass, so
    binding this regression test to it makes the test track whatever adapter ran last.
    """
    ladder = _ladder(
        {"philosophy": 0.66, "psychology": 0.44, "history": 0.62, "religion": 0.167},
        {"philosophy": 0.77, "psychology": 0.66, "history": 0.66, "religion": 0.0},
    )
    after = _rung(ladder, "adapter")
    assert after is not None and after["domains"]["religion"] == 0.0  # the known v2 regression

    cand = build_candidate(
        candidate_id="local-sophia-v2-mlx", kind="lora_adapter", baseline_ladder=None,
        adapter_ladder=ladder, contaminated=False, protected=("religion", "history"),
        extra_artifacts=("a", "b"), proof_tag="formal_verifier:protected_floor[fallback:rejected]",
    )
    d = evaluate_update(cand, target_suite="total", min_target_delta=0.03, max_protected_regression=0.01)
    assert d.verdict == "reject"
    assert any("religion" in r for r in d.reasons)


def test_clean_improving_adapter_promotes() -> None:
    base = {"philosophy": 0.66, "psychology": 0.44, "history": 0.62, "religion": 0.33}
    adapter = {"philosophy": 0.77, "psychology": 0.66, "history": 0.66, "religion": 0.44}
    ladder = _ladder(base, adapter)
    cand = build_candidate(
        candidate_id="clean-v1", kind="lora_adapter", baseline_ladder=None, adapter_ladder=ladder,
        contaminated=False, protected=("religion", "history"),
        extra_artifacts=("ladder", "manifest"), proof_tag="formal_verifier:protected_floor[fallback:accepted]",
    )
    d = evaluate_update(cand, target_suite="total", min_target_delta=0.03, max_protected_regression=0.01)
    assert d.verdict == "promote", d.reasons


def test_v3_adapter_rejects_under_retention_gate() -> None:
    """v3 cleared the old (ladder-only) gate but its real learning-shift report shows a
    -50pp old-task regression; the retention-aware gate must now reject it."""
    import json
    adapter_ladder = json.loads((ROOT / "training/local_sophia_v2/eval_ladder_adapter.json").read_text())
    shift = json.loads(
        (ROOT / "agi-proof/learning-under-shift/"
                "shift-result-local-sophia-v3-mlx-2026-06-24.public-report.json").read_text()
    )
    cand = build_candidate(
        candidate_id="local-sophia-v3-mlx", kind="lora_adapter", baseline_ladder=None,
        adapter_ladder=adapter_ladder, contaminated=False, protected=("religion", "history"),
        extra_artifacts=("a", "b"), proof_tag="formal_verifier:protected_floor[fallback:accepted]",
    )
    retention = _retention_from_shift_report(shift, source="shift")
    # Without retention evidence the ladder-only gate promotes (the historical verdict).
    assert evaluate_update(cand, target_suite="total", min_target_delta=0.03).verdict == "promote"
    # With the real shift report attached, the same adapter is rejected for forgetting.
    d = evaluate_update(cand, target_suite="total", min_target_delta=0.03, retention=retention)
    assert d.verdict == "reject"
    assert any("retention regression" in r for r in d.reasons)


def test_formal_proof_agrees_with_scorecard() -> None:
    # protected regression -> formal proof rejects; clean -> accepts
    bad = build_candidate(
        candidate_id="bad", kind="lora_adapter", baseline_ladder=None,
        adapter_ladder=_ladder({"history": 0.6, "religion": 0.5}, {"history": 0.6, "religion": 0.0}),
        contaminated=False, protected=("religion", "history"), extra_artifacts=(), proof_tag="",
    )
    assert _formal_protected_floor_proof(list(bad.metrics), tolerance=0.01)["verdict"] == "rejected"

    good = build_candidate(
        candidate_id="good", kind="lora_adapter", baseline_ladder=None,
        adapter_ladder=_ladder({"history": 0.6, "religion": 0.5}, {"history": 0.6, "religion": 0.6}),
        contaminated=False, protected=("religion", "history"), extra_artifacts=(), proof_tag="",
    )
    assert _formal_protected_floor_proof(list(good.metrics), tolerance=0.01)["verdict"] == "accepted"


def main() -> int:
    test_v2_adapter_rejects_on_religion_regression()
    test_clean_improving_adapter_promotes()
    test_v3_adapter_rejects_under_retention_gate()
    test_formal_proof_agrees_with_scorecard()
    print("test_promote_adapter: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
