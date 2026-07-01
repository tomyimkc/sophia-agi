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
