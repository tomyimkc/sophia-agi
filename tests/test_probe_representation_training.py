# SPDX-License-Identifier: Apache-2.0
"""Tests for W5 probe-as-loss + Goodhart audit. Binds to the REAL agent.activation_probes."""
import importlib.util
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "probe_representation_training.py"


def _load():
    sys.modules.pop("w5tool", None)
    spec = importlib.util.spec_from_file_location("w5tool", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _skip_if_no_repo(m):
    if not m._REPO_OK:
        import pytest
        pytest.skip("agent.activation_probes unavailable")


def _separable_rows(n=20):
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append({"id": f"h{i}", "text": "per the cited source approximately x see reference", "label": False})
        else:
            rows.append({"id": f"d{i}", "text": "definitely certainly exactly proven absolute certainty", "label": True})
    return rows


def test_hidden_state_seam_reports_not_ready():
    """The honest seam: build_hidden_state_featurizer is still a stub, so this must be False."""
    m = _load(); _skip_if_no_repo(m)
    r = m.run(_separable_rows())
    assert r["hiddenStateFeaturizerReady"] is False


def test_audit_probe_is_disjoint_and_reported():
    m = _load(); _skip_if_no_repo(m)
    r = m.run(_separable_rows())
    assert r["ok"] is True
    assert "lossProbeAccuracy" in r and "auditProbeAccuracy" in r
    # splits are disjoint: sizes add to <= n
    s = r["splits"]
    assert s["lossTrain"] + s["auditTrain"] + s["test"] <= r["nRows"]


def test_goodhart_gap_gates_claim():
    m = _load(); _skip_if_no_repo(m)
    r = m.run(_separable_rows())
    # canClaimImprovement must equal (not gamingSuspected), and gap drives gaming flag
    assert r["canClaimImprovement"] == (not r["gamingSuspected"])
    assert r["gamingSuspected"] == (r["goodhartGap"] > 0.15)


def test_degenerate_labels_fail_closed():
    m = _load(); _skip_if_no_repo(m)
    rows = [{"id": f"x{i}", "text": "same", "label": True} for i in range(10)]
    assert m.run(rows)["ok"] is False


def test_too_few_rows_fail_closed():
    m = _load(); _skip_if_no_repo(m)
    assert m.run([{"id": "a", "text": "t", "label": True}])["ok"] is False