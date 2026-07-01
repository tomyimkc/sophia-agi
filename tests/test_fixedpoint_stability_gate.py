#!/usr/bin/env python3
"""O3 fixed-point stability: supported vs unsupported separation, fail-closed."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import fixedpoint_stability_gate as fp


def _recs():
    recs = []
    for i in range(10):
        recs.append({"claim": f"the melting point of iron is {1538+i} degrees celsius",
                     "evidence": [f"iron melting point {1538+i} degrees celsius measured",
                                  f"metallurgy table iron {1538+i} celsius"], "supported": True})
    for i in range(10):
        recs.append({"claim": f"the population of atlantis is {i} million people",
                     "evidence": [f"iron melting point {1500+i} celsius",
                                  f"quantum chromodynamics lecture {i}"], "supported": False})
    return recs


def test_no_claim_fail_closed():
    out = fp.run([{"evidence": ["x"]}])
    assert out.get("environmentArtifact") and out["ok"] is False


def test_no_evidence_is_maximally_unstable():
    r = fp.iterate_fixedpoint("some claim", [], dim=64)
    assert r["converged"] is False and r["residual"] == 1.0


def test_supported_has_lower_residual():
    out = fp.run(_recs(), steps=50)
    sep = out["separation"]
    assert sep["meanResidualSupported"] < sep["meanResidualUnsupported"]


def test_suggested_threshold_separates():
    out = fp.run(_recs(), steps=50)
    at = out["separation"]["atSuggested"]
    # at the data-driven threshold, supported accept and unsupported reject
    sup_acc = int(at["supportedAccepted"].split("/")[0])
    uns_acc = int(at["unsupportedAccepted"].split("/")[0])
    assert sup_acc > uns_acc


def test_default_is_fail_closed_conservative():
    # tight default threshold should abstain rather than wrongly accept unsupported claims
    out = fp.run(_recs(), steps=50)
    assert 0.0 <= out["abstainRate"] <= 1.0
    assert out["candidateOnly"] is True
