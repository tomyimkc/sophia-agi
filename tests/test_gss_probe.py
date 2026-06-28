# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the GSS probe harness — pure helpers, a real MoELM forward pass, and skips.

The ``moelm`` backend gives a genuine (toy) forward pass with no heavy deps, so the full
extraction→gate pipeline is exercised in CI. The ``hf`` backend is asserted to skip
cleanly when torch/transformers are absent (the repo's gated-GPU-path convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

np = pytest.importorskip("numpy")

import gss_probe  # noqa: E402
from serving.gss_feasibility import GSSFeasibilityReport  # noqa: E402


# ---- pure helpers ----------------------------------------------------------

def test_softmax_normalises() -> None:
    p = gss_probe.softmax_np(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()


def test_int4_roundtrip_is_lossy_but_bounded_and_shape_preserving() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 40))
    deq = gss_probe.int4_group_quant_roundtrip(W, group=8)
    assert deq.shape == W.shape
    rel = np.linalg.norm(deq - W) / np.linalg.norm(W)
    assert 0.0 < rel < 0.25                      # real 4-bit error, not zero, not huge


def test_int4_roundtrip_exact_on_representable_values() -> None:
    W = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])   # within one group's int grid
    deq = gss_probe.int4_group_quant_roundtrip(W, group=8)
    assert np.allclose(deq, W)


def test_gate_probs_topk_zeroes_unrouted_experts() -> None:
    logits = np.array([[3.0, 2.0, 1.0, 0.0]])
    c = gss_probe.gate_probs_to_contribs(logits, top_k=2)
    assert (c[0] > 0).sum() == 2                 # only the top-2 experts are "read"
    assert c[0, 2] == 0.0 and c[0, 3] == 0.0


# ---- moelm backend: a real forward pass end-to-end -------------------------

def test_moelm_probe_shapes_and_ranges() -> None:
    contribs, target, draft = gss_probe.probe_moelm(
        n_experts=8, tokens=32, train_steps=100, seed=1
    )
    assert contribs.shape == (32, 8)
    assert target.shape[0] == 32 and draft.shape[0] == 32
    assert target.shape[1] == draft.shape[1]
    assert np.allclose(target.sum(axis=1), 1.0)  # valid distributions
    # top-1 routing → exactly one expert read per token → ρ row-fraction = 1/8
    assert ((contribs > 0).sum(axis=1) == 1).all()


def test_moelm_probe_drives_a_complete_go_nogo() -> None:
    from serving.gss_feasibility import GSSFeasibilityGate
    contribs, target, draft = gss_probe.probe_moelm(n_experts=16, tokens=48, train_steps=200)
    rep = GSSFeasibilityGate(gamma=4).evaluate(contribs, target, draft)
    assert isinstance(rep, GSSFeasibilityReport)
    assert 0.0 < rep.rho <= 1.0
    assert 1.0 <= rep.k <= 5.0
    assert rep.cost_ratio > 0
    d = rep.as_dict()
    assert all(d[key] is not None for key in ("go", "rho", "k", "cost_ratio"))


# ---- CLI smoke / skip behaviour --------------------------------------------

def test_main_moelm_returns_zero(capsys) -> None:
    rc = gss_probe.main(["--backend", "moelm", "--experts", "8", "--tokens", "24",
                         "--train-steps", "50"])
    assert rc == 0
    assert "GO" in capsys.readouterr().out


def test_main_hf_requires_model() -> None:
    assert gss_probe.main(["--backend", "hf"]) == 2


def test_main_hf_skips_cleanly_without_torch(capsys) -> None:
    """Without torch the hf backend must skip and exit 0 (CI-green), not crash."""
    if _has_torch():
        pytest.skip("torch present; skip-path not exercised")
    rc = gss_probe.main(["--backend", "hf", "--model", "x/y", "--prompt", "hi"])
    assert rc == 0
    assert "skipped" in capsys.readouterr().out


def test_writes_json_report(tmp_path) -> None:
    out = tmp_path / "gss.json"
    rc = gss_probe.main(["--backend", "moelm", "--tokens", "16", "--train-steps", "30",
                         "--out", str(out)])
    assert rc == 0 and out.exists()
    import json
    d = json.loads(out.read_text())
    assert d["backend"] == "moelm" and "cost_ratio" in d


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False
