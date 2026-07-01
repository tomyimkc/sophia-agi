#!/usr/bin/env python3
"""Offline tests for run_t1_gated_self_training.py.

These test the OUTCOME-CLASSIFICATION logic and fail-closed behaviour with a
synthetic generator, so they run without torch/backends and without the repo's
gate_reward/continual_plasticity present (they stub the two the loop calls).
The canary assertion: a shift-split that reverses the learnable signal must be
classified as verifier_overfit (memorization), NOT heldout_lift.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_t1_gated_self_training.py"


def _load_with_stubs():
    """Import the tool with agent.gate_reward / agent.continual_plasticity stubbed."""
    # stub agent.gate_reward
    gr = types.ModuleType("agent.gate_reward")
    gr.reward = lambda completion, question=None, temptation=None: (
        1.0 if "GOOD" in str(completion) else (0.5 if "IDK" in str(completion) else -1.0))
    gr.is_abstention = lambda t: "IDK" in str(t)
    gr.REWARD_CLEAN = 1.0
    gr.REWARD_ABSTAIN = 0.5
    # stub agent.continual_plasticity
    cp = types.ModuleType("agent.continual_plasticity")

    class EvalMetric:
        def __init__(self, suite, before, after, protected=False):
            self.suite, self.before, self.after, self.protected = suite, before, after, protected

    class UpdateCandidate:
        def __init__(self, id, kind, metrics, verifier_artifacts=(), contaminated=False, notes=""):
            self.id, self.kind, self.metrics = id, kind, metrics

    class PromotionDecision:
        def __init__(self, verdict, reasons):
            self.candidate_id, self.verdict, self.reasons, self.metrics = "x", verdict, reasons, {}

    def evaluate_update(candidate, *, target_suite, **kw):
        prot_reg = [m.suite for m in candidate.metrics if getattr(m,'protected',False) and (m.after-m.before) < -0.01]
        if prot_reg:
            return PromotionDecision("reject", tuple(f"protected regression: {s}" for s in prot_reg))
        tgt = next(m for m in candidate.metrics if m.suite == target_suite)
        if tgt.after - tgt.before >= 0.03:
            return PromotionDecision("promote", ("target improved",))
        return PromotionDecision("reject", ("no target delta",))

    cp.EvalMetric, cp.UpdateCandidate, cp.PromotionDecision = EvalMetric, UpdateCandidate, PromotionDecision
    cp.evaluate_update = evaluate_update
    cp.append_promotion_ledger = lambda *a, **k: None

    # snapshot so we can restore the REAL agent.* after the tool captures its refs —
    # otherwise the empty-path 'agent' stub shadows the real package for sibling suites
    _keys = ("agent", "agent.gate_reward", "agent.continual_plasticity")
    _saved = {k: sys.modules.get(k) for k in _keys}
    agent_pkg = types.ModuleType("agent"); agent_pkg.__path__ = []
    sys.modules.update({"agent": agent_pkg, "agent.gate_reward": gr,
                        "agent.continual_plasticity": cp})
    spec = importlib.util.spec_from_file_location("t1tool", TOOL)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for k, v in _saved.items():  # restore real modules (or remove stub if none was present)
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    return mod


def test_heldout_lift_detected():
    """Real generalization: after-model matches on BOTH scored and shifted refs."""
    m = _load_with_stubs()
    gen_before = lambda p: "IDK"           # before: abstains -> 0 pass
    gen_after = lambda p: "GOOD answer42"  # after: grounded + matches both refs
    traces = [{"prompt": "t"}]
    scored = [{"prompt": "q", "reference": "answer42"}]
    shifted = [{"prompt": "q2", "reference": "answer42"}]
    out = m.run_round(_p(traces), _p(scored), _p(shifted), gen_before, gen_after, None)
    assert out["passRate"]["lift_shifted"] > 0
    assert out["outcome"] == "heldout_lift"


def test_verifier_overfit_canary():
    """Learnable on scored, REVERSED on shift -> must be verifier_overfit."""
    m = _load_with_stubs()
    gen_before = lambda p: "IDK"
    # after: matches on the scored ref but NOT on the shifted ref -> scored up, shift flat
    gen_after = lambda p: "GOOD scoredref"
    scored = [{"prompt": "q", "reference": "scoredref"}]
    shifted = [{"prompt": "q2", "reference": "shiftedref"}]
    out = m.run_round(_p(scored), _p(scored), _p(shifted), gen_before, gen_after, None)
    assert out["passRate"]["lift_scored"] > 0
    assert out["passRate"]["lift_shifted"] <= 0
    assert out["outcome"] == "verifier_overfit"


def test_env_artifact_when_no_backend(tmp_path):
    m = _load_with_stubs()
    art = m.env_artifact("no backend")
    assert art["environmentArtifact"] and art["score"] is None
    assert art["canClaimAGI"] is False


# --- helpers to pass in-memory lists where run_round expects jsonl Paths ---
import json as _json


def _p(items):
    import tempfile
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for it in items:
        f.write(_json.dumps(it) + "\n")
    f.close()
    return Path(f.name)


def _wj(items):
    return _p(items)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---- review D5/D8: added after adversarial review ----
def test_abstention_not_credited_as_pass():
    """D8: an abstaining answer (reward 0.5) must NOT count as a verified pass."""
    m = _load_with_stubs()
    # stub gate_reward: 'IDK' abstains (handled by is_abstention); 'GOOD' is clean
    gen = lambda p: "IDK"
    items = [{"prompt": "q", "reference": "answer"}]
    assert m.verified_pass_rate(items, gen) == 0.0


def test_no_reference_item_not_credited():
    """D8: an item with no gold reference cannot be scored a pass."""
    m = _load_with_stubs()
    gen = lambda p: "GOOD answer"
    items = [{"prompt": "q"}]  # no 'reference'
    assert m.verified_pass_rate(items, gen) == 0.0


def test_protected_regression_triggers_reject():
    """D5: a regression on a protected suite must flip promote->reject.

    Setup: target held-out LIFTS (abstain->correct), but the protected 'religion'
    suite REGRESSES (was correct before, abstains after). Without the protected
    flag this regression would be silently ignored; with it, evaluate_update rejects.
    """
    m = _load_with_stubs()
    # target prompts contain 'q'/'t'; religion prompt is 'rel'
    gen_before = lambda p: ("GOOD answer42" if "rel" in p else "IDK")   # religion PASSES before
    # after: target now answers (lift), religion now abstains (regress)
    gen_after = lambda p: ("IDK" if "rel" in p else "GOOD answer42")
    traces = [{"prompt": "t"}]
    scored = [{"prompt": "q", "reference": "answer42"}]
    shifted = [{"prompt": "q2", "reference": "answer42"}]
    prot = {"religion": [{"prompt": "rel1", "reference": "answer42"}]}
    out = m.run_round(_p(traces), _p(scored), _p(shifted),
                      gen_before, gen_after, None, protected=prot)
    assert out["promotion"]["verdict"] != "promote"
    assert any("religion" in str(r) or "protected" in str(r).lower()
               for r in out["promotion"]["reasons"])
