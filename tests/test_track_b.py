# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for Track B: QAT training, distillation-into-sparsity, calibration stage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pretraining.distill import study as distill_study  # noqa: E402
import run_calibration  # noqa: E402


# ---- B1: quantization-aware training ---------------------------------------

def test_qat_offline_invariants() -> None:
    pytest.importorskip("numpy")
    from training import qat
    ok, detail = qat.offline_invariants()
    assert ok, detail["checks"]


def test_qat_target_set_and_coverage() -> None:
    """The QAT served set covers attn + per-expert/fused experts and excludes head/norm/router/LoRA;
    coverage accounting SEES the experts (the blind spot that hid the MoE under-quantization)."""
    from training import qat
    assert qat.is_qat_target_name("model.layers.0.self_attn.q_proj")
    assert qat.is_qat_target_name("model.layers.0.mlp.experts.7.down_proj.weight")
    assert qat.is_qat_target_name("model.layers.0.mlp.experts.gate_up_proj")   # fused
    for n in ("lm_head", "model.embed_tokens", "model.layers.0.mlp.gate",      # router
              "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"):
        assert not qat.is_qat_target_name(n)
    cov = qat.summarize_qat_coverage([
        "model.layers.0.self_attn.q_proj", "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.1.down_proj"])
    assert cov["attn"] == 1 and cov["expert"] == 2
    assert qat.summarize_qat_coverage(["model.layers.0.self_attn.q_proj"])["expert"] == 0


def test_qat_fake_quant_roundtrip_tight() -> None:
    np = pytest.importorskip("numpy")
    from training import qat
    W = np.random.default_rng(0).standard_normal((48, 32))
    rel = np.linalg.norm(qat.fake_quant(W, "int8") - W) / np.linalg.norm(W)
    assert rel < 0.02


def test_qat_push_lowers_post_quant_error() -> None:
    np = pytest.importorskip("numpy")
    from training import qat
    rng = np.random.default_rng(1)
    W = rng.standard_normal((48, 48))
    x = rng.standard_normal((16, 48))
    before = qat.post_quant_matmul_error(W, x, "nvfp4")
    Wc = W.copy()
    for _ in range(40):
        Wc = Wc - 0.5 * (Wc - qat.fake_quant(Wc, "nvfp4"))
    assert qat.post_quant_matmul_error(Wc, x, "nvfp4") < before


def test_qat_unknown_scheme_rejected() -> None:
    pytest.importorskip("numpy")
    from training import qat
    with pytest.raises(ValueError):
        qat.fake_quant([[1.0, 2.0]], "ternary")


# ---- B2: distillation into sparsity ----------------------------------------

def test_distill_offline_invariants() -> None:
    ok, detail = distill_study.offline_invariants()
    assert ok, detail["checks"]


def test_distill_teacher_grounded_and_moe_sparse() -> None:
    rep = distill_study.run_study(vocab=6, context=1, teacher_hidden=48, student_hidden=8,
                                  n_experts=6, n_train=400, n_eval=200, epochs=40, lr=0.1, seed=0)
    # Teacher reaches near the known floor (so distillation is from a real teacher).
    assert rep["distill_grounded"]
    # The MoE student is genuinely sparse: more total params than active footprint.
    assert rep["moe_total_over_active"] > 1.0
    assert rep["moe_active_fraction"] < 1.0


def test_distill_relabel_preserves_contexts() -> None:
    from pretraining.nano.data import make_source, sample_stream, to_examples
    from pretraining.nano.model import NanoLM
    from pretraining.nano.train import train
    ex = to_examples(sample_stream(make_source(6, order=1, seed=0), 50, seed=0), 1)
    t = NanoLM(6, 1, 16, seed=0)
    train(t, ex, epochs=3, optimizer="adam", lr=0.1, seed=0)
    rel = distill_study.teacher_relabel(t, ex)
    assert len(rel) == len(ex)
    assert all(rel[i][0] == ex[i][0] for i in range(len(ex)))
    assert all(0 <= lbl < 6 for _, lbl in rel)


# ---- B3: calibration stage -------------------------------------------------

def test_calibration_stage_offline_invariants() -> None:
    ok, detail = run_calibration.offline_invariants()
    assert ok, detail["checks"]


def test_calibration_empty_sources_fail_closed(tmp_path) -> None:
    ok, ds = run_calibration.run_calibration(
        [tmp_path / "nope.jsonl"], tmp_path / "out.json",
        target_bits=4.5, max_rows=16, dry_run=True)
    assert ok is False and "error" in ds


def test_calibration_clean_source_passes(tmp_path) -> None:
    import json
    row = {"messages": [{"role": "user", "content": "deployment prompt " + "z" * 80},
                        {"role": "assistant", "content": "grounded answer " + "y" * 80}]}
    src = tmp_path / "deploy.jsonl"
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "ds.json"
    ok, ds = run_calibration.run_calibration(
        [src], out, target_bits=4.5, max_rows=16,
        eval_prompts={"unrelated eval prompt"}, dry_run=True)
    assert ok and out.exists()
    assert ds["decontamination"]["disjoint_from_eval"] is True
    assert "necessary, not sufficient" in ds["honest_scope"]


# ---- B3: calibrate stage wires into the experiment plan when enabled --------

def test_calibrate_stage_absent_by_default() -> None:
    from sophia.trainer.config import ExperimentConfig
    from sophia.trainer.plan import build_experiment_plan
    plan = build_experiment_plan(ExperimentConfig())
    assert "calibrate" not in [spec.stage for spec in plan]


def test_calibrate_stage_present_when_enabled() -> None:
    from sophia.trainer.config import CalibrateConfig, ExperimentConfig
    from sophia.trainer.plan import build_experiment_plan
    cfg = ExperimentConfig(calibrate=CalibrateConfig(enabled=True, target_bits=4.0))
    plan = build_experiment_plan(cfg, stages=["calibrate"])
    spec = next(s for s in plan if s.stage == "calibrate")
    assert "tools/run_calibration.py" in spec.shell()
    assert "--target-bits" in spec.command and "4.0" in spec.command
    assert spec.dry_run  # default dry-run safety


def test_config_roundtrips_calibrate() -> None:
    from sophia.trainer.config import ExperimentConfig
    cfg = ExperimentConfig.from_dict(
        {"schema": "sophia.experiment.v1",
         "calibrate": {"enabled": True, "targetBits": 3.5, "sources": ["a.jsonl"]}})
    assert cfg.calibrate.enabled and cfg.calibrate.target_bits == 3.5
    assert cfg.to_dict()["calibrate"]["enabled"] is True
