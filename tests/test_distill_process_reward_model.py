# SPDX-License-Identifier: Apache-2.0
"""Offline tests for W1 verifier-distilled PRM. Stubs agent.* so no sympy/backend needed."""
import importlib.util
import sys
import types
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "distill_process_reward_model.py"


def _install_stubs():
    # stub agent.step_verifier.verify_derivation with a controllable verdict sequence
    sv = types.ModuleType("agent.step_verifier")

    class _V:
        def __init__(self, index, fe, te, verdict, checker="stub"):
            self.index, self.from_expr, self.to_expr = index, fe, te
            self.verdict, self.checker = verdict, checker

    class _Res:
        def __init__(self, steps):
            self.steps = steps

    def verify_derivation(steps, *, gold=None, default_domain="math"):
        # exprs like {"expr": "..#accepted"} / "..#rejected" / "..#abstain" encode the verdict
        out = []
        for i in range(len(steps) - 1):
            cur = steps[i + 1]
            expr = cur["expr"] if isinstance(cur, dict) else str(cur)
            verdict = "accepted"
            for v in ("accepted", "rejected", "abstain"):
                if v in expr:
                    verdict = v
            fe = steps[i]["expr"] if isinstance(steps[i], dict) else str(steps[i])
            out.append(_V(i, fe, expr, verdict))
        return _Res(out)

    sv.verify_derivation = verify_derivation
    sys.modules["agent.step_verifier"] = sv

    # stub activation_probes: a trivial probe that predicts by presence of "GOOD"
    ap = types.ModuleType("agent.activation_probes")

    class _Probe:
        def __init__(self, rows):
            self.rows = rows

        def to_dict(self):
            return {"name": "stub"}

    def train_centroid_probe(rows, *, name="p", threshold=0.5):
        return _Probe(rows)

    def evaluate_probe(probe, rows):
        # predict label True iff "accepted" in text; report accuracy vs true label
        tp = tn = fp = fn = 0
        for r in rows:
            pred = "accepted" in r["text"]
            lab = bool(r["label"])
            tp += pred and lab; tn += (not pred) and (not lab)
            fp += pred and (not lab); fn += (not pred) and lab
        n = len(rows)
        return {"n": n, "metrics": {"accuracy": (tp + tn) / n if n else 0,
                                    "precision": tp / (tp + fp) if tp + fp else 0,
                                    "recall": tp / (tp + fn) if tp + fn else 0,
                                    "falsePositiveRate": fp / (fp + tn) if fp + tn else 0}}

    ap.train_centroid_probe = train_centroid_probe
    ap.evaluate_probe = evaluate_probe
    sys.modules["agent.activation_probes"] = ap


def _load():
    _install_stubs()
    sys.modules.pop("w1tool", None)
    spec = importlib.util.spec_from_file_location("w1tool", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # remove stub agent.* from sys.modules so sibling suites (W2..W5) import the REAL
    # agent.* modules, not our stubs. The tool module already captured its references.
    for name in ("agent.step_verifier", "agent.activation_probes"):
        sys.modules.pop(name, None)
    return m


def _derivs():
    # each derivation: start -> accepted -> rejected -> accepted ...
    return [
        {"id": "d1", "domain": "math", "steps": ["s", "a accepted", "b rejected", "c accepted"]},
        {"id": "d2", "domain": "math", "steps": ["s", "d accepted", "e rejected", "f accepted"]},
        {"id": "d3", "domain": "physics", "steps": ["s", "g accepted", "h rejected"]},
    ]


def test_labels_map_verdict_correctly():
    m = _load()
    rows = m.label_steps(_derivs())
    # label must equal (target-side verdict == accepted); the to_expr carries the verdict token
    for r in rows:
        target = r["text"].split("->")[-1]
        assert r["label"] == ("accepted" in target)
    assert len(rows) == 8  # 3 + 3 + 2 transitions, none abstain


def test_abstain_steps_dropped():
    m = _load()
    # s->p abstain (dropped), p->q accepted, q->r rejected  => 1 dropped, 2 kept
    d = [{"id": "x", "domain": "math", "steps": ["s", "p abstain", "q accepted", "r rejected"]}]
    rows = m.label_steps(d)
    assert getattr(m.label_steps, "last_dropped_abstain") == 1
    assert all("abstain" not in r["text"].split("->")[-1] for r in rows)


def test_bare_string_steps_coerced():
    m = _load()
    # bare strings (not dicts) must be coerced, not crash
    d = [{"id": "s1", "domain": "math", "steps": ["start", "a accepted", "b rejected"]}]
    rows = m.label_steps(d)
    assert len(rows) == 2


def test_degenerate_single_class_fails_closed():
    m = _load()
    # >=4 labeled steps but all one class -> must reach the degenerate-label guard
    d = [{"id": "d1", "domain": "math", "steps": ["s", "a accepted", "b accepted", "c accepted"]},
         {"id": "d2", "domain": "math", "steps": ["s", "d accepted", "e accepted", "f accepted"]}]
    r = m.run(d)
    assert r["ok"] is False and "degenerate" in r["reason"]


def test_full_run_reports_heldout_and_domain():
    m = _load()
    r = m.run(_derivs(), holdout_domain="physics", holdout_frac=0.3, seed=0)
    assert r["ok"] is True
    assert "heldOutRandom" in r and "heldOutDomain" in r
    assert r["heldOutDomain"]["domain"] == "physics"
    assert r["labelBalance"]["accepted"] > 0 and r["labelBalance"]["rejected"] > 0


def test_empty_input_fails_closed():
    m = _load()
    assert m.run([])["ok"] is False