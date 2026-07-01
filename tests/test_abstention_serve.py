# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""CI regression for the abstention hedge: offline invariants + the real v5 operating point.

Makes the serve-time abstention gate (``serving/abstention_serve.py``) and the frontier measurement
(``serving/quant_abstention.py``) standing checks. The v5 NVFP4 cert genuinely FAILS the never-flip
bar (top1 0.9219 < 0.97), but the frontier found a shippable operating point (answered_top1 ~0.982 @
coverage ~0.86) and the serve gate enforces it from the quant top1-top2 margin alone. No capability
claim; canClaimAGI stays false. Runs under pytest OR as a plain script.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_quant_abstention_offline_invariants():
    from serving import quant_abstention
    ok, d = quant_abstention.offline_invariants()
    assert ok, d["checks"]


def test_abstention_serve_offline_invariants():
    from serving import abstention_serve
    ok, d = abstention_serve.offline_invariants()
    assert ok, d["checks"]


def test_abstaining_decoder_offline_invariants():
    from serving import abstaining_decoder
    ok, d = abstaining_decoder.offline_invariants()
    assert ok, d["checks"]


def test_decoder_circuit_breaker_never_ships_mostly_abstained():
    import numpy as np

    from serving.abstaining_decoder import ABSTAIN_TOKEN, AbstainingDecoder
    from serving.abstention_serve import AbstentionPolicy
    # A policy that abstains on everything must trip the breaker and NOT return a full-length answer.
    pol = AbstentionPolicy(threshold=-1.0, target_answered=0.97, measured_coverage=0.0,
                           measured_answered_top1=0.0, raw_top1=0.5, n_test=512, source="t")
    V = 16

    def tie(step, emitted):
        r = np.full(V, 0.005 / (V - 2)); r[0] = 0.5; r[1] = 0.495; return r
    tr = AbstainingDecoder(pol).decode(tie, max_tokens=64, stop_on_coverage_below=0.5)
    assert tr.n_steps < 64, "breaker must stop a mostly-abstained decode early"
    assert all(t == ABSTAIN_TOKEN for t in tr.tokens)


def test_policy_from_cert_loads_shippable_point():
    from serving.abstention_serve import policy_from_cert
    # A cert artifact whose frontier cleared the bar -> a usable policy with the right threshold.
    cert = {"abstention_frontier": {
        "raw_top1": 0.9219, "target_answered": 0.97, "shippable": True,
        "shippable_operating_point": {"calib_quantile": 0.7, "threshold": 0.8072,
                                      "coverage": 0.8594, "answered_top1": 0.9818},
        "n_test": 128}}
    fd, p = tempfile.mkstemp(suffix=".json")
    Path(p).write_text(json.dumps(cert))
    try:
        pol = policy_from_cert(p)
    finally:
        Path(p).unlink()
    assert abs(pol.threshold - 0.8072) < 1e-9
    assert pol.measured_answered_top1 >= pol.target_answered   # the point actually clears the bar
    assert pol.n_test == 128                                    # thin-sample provenance is preserved


def test_lcb_selection_is_more_conservative_than_point_estimate():
    from serving.abstention_serve import policy_from_cert
    # v5 @ n=1024 shape: the max-coverage point (0.9737) clears 0.97 as a POINT estimate but its 95%
    # lower bound is below it; a stricter, lower-coverage point clears the bound. The LCB selection must
    # pick a point with coverage <= the point-estimate point and record the floor.
    fr = {"abstention_frontier": {
        "raw_top1": 0.8975, "target_answered": 0.97, "shippable": True, "n_test": 512,
        "shippable_operating_point": {"threshold": 0.8072, "coverage": 0.7422, "answered_top1": 0.9737},
        "frontier": [
            {"calib_quantile": 1.0, "threshold": 1.0, "coverage": 1.0, "answered_top1": 0.8750},
            {"calib_quantile": 0.85, "threshold": 0.9252, "coverage": 0.8027, "answered_top1": 0.9611},
            {"calib_quantile": 0.80, "threshold": 0.8072, "coverage": 0.7422, "answered_top1": 0.9737},
            {"calib_quantile": 0.75, "threshold": 0.8614, "coverage": 0.7012, "answered_top1": 0.9833},
            {"calib_quantile": 0.70, "threshold": 0.8072, "coverage": 0.6426, "answered_top1": 0.9970},
        ]}}
    fd, p = tempfile.mkstemp(suffix=".json")
    Path(p).write_text(json.dumps(fr))
    try:
        pt_pol = policy_from_cert(p)                     # point estimate -> max coverage
        lcb_pol = policy_from_cert(p, confidence=0.95)   # confidence floor -> stricter
    finally:
        Path(p).unlink()
    assert pt_pol.selection == "point_estimate"
    assert lcb_pol.selection.startswith("wilson_lcb")
    assert lcb_pol.measured_coverage <= pt_pol.measured_coverage        # more conservative
    assert lcb_pol.answered_lcb is not None and lcb_pol.answered_lcb >= 0.97  # floor clears the bar


def test_lcb_raises_when_no_point_is_robust():
    from serving.abstention_serve import policy_from_cert
    # A frontier that clears the target only as point estimates on tiny samples -> no robust point.
    fr = {"abstention_frontier": {
        "raw_top1": 0.9, "target_answered": 0.97, "n_test": 20,
        "shippable_operating_point": {"threshold": 0.5, "coverage": 0.9, "answered_top1": 0.972},
        "frontier": [{"calib_quantile": 0.9, "threshold": 0.5, "coverage": 0.9, "answered_top1": 0.972}]}}
    fd, p = tempfile.mkstemp(suffix=".json")
    Path(p).write_text(json.dumps(fr))
    try:
        raised = False
        try:
            policy_from_cert(p, confidence=0.95)
        except ValueError:
            raised = True
        assert raised, "LCB selection must raise when no point is robust, not answer everything"
    finally:
        Path(p).unlink()


def test_adopted_operating_points_are_self_consistent():
    from serving.abstention_serve import ADOPTED_OPERATING_POINTS, adopted_policy
    data = json.loads(Path(ADOPTED_OPERATING_POINTS).read_text())
    assert data["adopted"] is True and data["ingredient"] == "conformal-abstention-serve"
    min_cov = float(data["min_coverage"]); target = float(data["target_answered"])
    # every pinned adopted point must clear its own pre-registered bar (cov>=min AND LCB floor>=target)
    for key, op in data["operating_points"].items():
        assert op["coverage"] >= min_cov, (key, op["coverage"])
        assert op["answered_top1_lcb95"] >= target, (key, op["answered_top1_lcb95"])
        assert op["raw_top1"] < target, (key, "raw should FAIL the bar — this is a hedge, not a raw pass")
    # the default loads into a usable AbstentionPolicy carrying the confidence-floor provenance
    pol = adopted_policy()
    assert pol.selection.startswith("wilson_lcb") and pol.answered_lcb >= target
    assert pol.measured_coverage >= min_cov


def test_none_operating_point_refuses_to_answer_everything():
    from serving.abstention_serve import policy_from_cert
    # If abstention cannot rescue the model, loading MUST raise, never silently answer-everything.
    cert = {"abstention_frontier": {"shippable_operating_point": None, "target_answered": 0.97,
                                    "raw_top1": 0.5, "n_test": 10}}
    fd, p = tempfile.mkstemp(suffix=".json")
    Path(p).write_text(json.dumps(cert))
    try:
        raised = False
        try:
            policy_from_cert(p)
        except ValueError:
            raised = True
        assert raised, "None operating point must raise, not fall back to answering everything"
    finally:
        Path(p).unlink()


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [ok] {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  [XX] {fn.__name__}: {exc}")
    print(f"abstention-serve regression: {'PASS' if not failed else 'FAIL'} ({len(fns) - failed}/{len(fns)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
