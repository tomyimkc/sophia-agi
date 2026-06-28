# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""End-to-end low-RAM loop: train a real MoE block -> account RAM -> compose runtime -> certify.

Proves the whole stack composes on a *trained* model (not just reference policy):
  1. train a real trainable MoE LM block (pretraining.architecture.moe.MoELM) until it learns;
  2. bridge its measured param counts into a frontier-scale ModelSpec (ModelSpec.from_block_counts);
  3. account resident RAM under the stack (plan_ram) — frontier total, fraction of the RAM;
  4. compose the integrated runtime (LowRamRuntime) and prove a decode step stays in budget;
  5. certify a *quantized* copy of the trained model against full precision via the no-overclaim
     gate (serving.lowram_eval.LowRamGate) — the train -> quantize -> measure loop, in miniature.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from moe.quant import nvfp4_roundtrip  # noqa: E402
from pretraining.architecture.moe import MoELM  # noqa: E402
from pretraining.nano.data import make_source, sample_stream, to_examples  # noqa: E402
from serving import lowram_eval  # noqa: E402
from serving.lowram_runtime import LowRamRuntime, ModelSpec, plan_ram  # noqa: E402


def _train_moe_block(epochs: int = 25, lr: float = 0.1, seed: int = 0):
    """Train a tiny real MoE LM block on the known-floor nano substrate; return (model, eval_ex)."""
    src = make_source(6, order=1, seed=seed)
    train_ex = to_examples(sample_stream(src, 400, seed=seed), 1)
    eval_ex = to_examples(sample_stream(src, 200, seed=seed + 999), 1)
    m = MoELM(6, 1, 8, 6, seed=seed)
    import random
    rng = random.Random(seed)
    order = list(range(len(train_ex)))
    for _ in range(epochs):
        rng.shuffle(order)
        for j in order:
            m.train_step(train_ex[j][0], train_ex[j][1], lr)
    return m, eval_ex


def _quantize_moe_block(m: MoELM) -> MoELM:
    """Return a copy of the trained block with every expert weight matrix NVFP4 round-tripped."""
    q = copy.deepcopy(m)
    for exp in q.experts:
        for key in ("W1", "W2"):
            W = np.asarray(exp[key], dtype=np.float64)
            exp[key] = nvfp4_roundtrip(W).tolist()
    return q


def _probs(m: MoELM, eval_ex) -> "np.ndarray":
    """Next-token distribution per eval context (N, V)."""
    return np.asarray([m.forward(ctx) for ctx, _ in eval_ex], dtype=np.float64)


def test_end_to_end_train_account_compose_certify() -> None:
    # 1. Train a real MoE block; confirm it actually learned (loss below the uniform baseline).
    m, eval_ex = _train_moe_block()
    eval_nll = sum(m.nll(ctx, t) for ctx, t in eval_ex) / len(eval_ex)
    import math
    assert eval_nll < math.log(6)                       # better than uniform over vocab=6

    # 2. Bridge measured block counts -> a frontier-scale ModelSpec (92-layer MoE).
    spec = ModelSpec.from_block_counts(
        "sophia-v1-e2e", n_layers=92, hidden=12288, vocab=151552,
        block_total_params=m.num_params(), block_active_params=m.active_params(),
        n_routed_experts=m.n_experts, active_experts=2)
    assert spec.is_moe and spec.sparsity_ratio > 1.0

    # 3. Account RAM: frontier total params, a fraction of the resident footprint.
    rep = plan_ram(spec)
    assert rep["reduction_vs_dense"]["expert_offload"] > 1.0
    assert rep["operating_points"]["full_stream"]["resident_gb"] < \
        rep["operating_points"]["dense_fp16"]["resident_gb"]

    # 4. Compose the integrated runtime; a full decode step stays within the GPU budget.
    rt = LowRamRuntime(spec, gpu_budget_bytes=50_000_000, weight_bits=4.5, prefetch_depth=1)
    rt.decode_step()
    assert rt.peak_resident_bytes() <= rt.gpu_budget_bytes() + 1
    assert rt.trunk.stats.disk_loads >= spec.n_layers
    assert rt.experts.stats.promotes > 0

    # 5. Certify: quantize the trained block, measure it vs full precision through the gate.
    full = _probs(m, eval_ex)
    quant = _probs(_quantize_moe_block(m), eval_ex)
    gate = lowram_eval.LowRamGate()
    # Identical model passes with ~zero divergence (sanity of the loop).
    assert gate.evaluate(full, full.copy(), mem_ratio=3.56).passed
    # The real quantized model yields a structured, finite verdict carrying the memory saving.
    rep_q = gate.evaluate(full, quant, mem_ratio=3.56)
    assert rep_q.n_eval == len(eval_ex)
    assert rep_q.passed in (True, False)
    assert rep_q.mem_ratio == round(3.56, 4)
    assert np.isfinite(rep_q.mean_kl)


def test_runpod_qat_lowram_plan_is_cost_gated() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import runpod_qat_lowram as rq
    ok, detail = rq.offline_invariants()
    assert ok, detail["checks"]
    # Without inputs: not launchable. With inputs: QAT flags passed through, never self-launches.
    assert rq.build_run_plan(base_model=None, gpu="A100-80G", scheme="nvfp4", budget_usd=None,
                             branch="b", epochs=1, calib="c.json", target_bits=4.5)["ready_to_launch"] is False
    plan = rq.build_run_plan(base_model="org/base", gpu="NVIDIA A100-SXM4-80GB", scheme="nvfp4",
                             budget_usd=10.0, branch="b", epochs=1, calib="c.json", target_bits=4.5)
    assert plan["ready_to_launch"] is True
    allcmd = " ".join(" ".join(s.get("command", [])) for s in plan["steps"])
    assert "--qat --qat-scheme nvfp4" in allcmd and "--yes" not in allcmd


def test_emitted_runpod_command_uses_real_runpod_train_flags() -> None:
    """CLI-drift guard: every flag the launcher emits must exist in runpod_train.py's argparse."""
    import re
    sys.path.insert(0, str(ROOT / "tools"))
    import runpod_qat_lowram as rq
    defined = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', (ROOT / "tools" / "runpod_train.py").read_text()))
    assert "--gpu-type" in defined and "--extra-train-args" in defined   # sanity: the passthrough exists
    plan = rq.build_run_plan(base_model="org/MoE", gpu="NVIDIA H200", scheme="nvfp4", budget_usd=200.0,
                             branch="b", epochs=1, calib="c.json", target_bits=4.5, gpu_count=8)
    cmd = next(s for s in plan["steps"] if s["stage"] == "qat_train")["command"]
    # Flag tokens targeting runpod_train.py (skip the --extra-train-args VALUE, which has spaces).
    flags = [t for t in cmd if t.startswith("--") and " " not in t]
    for f in flags:
        assert f in defined, f"launcher emits {f}, not a real runpod_train.py flag"
    # The sharding passthrough rides inside --extra-train-args, not as bare runpod_train flags.
    assert "--gpu" not in cmd and "--extra" not in cmd


def test_runpod_qat_lowram_spark_target_is_free_and_aarch64_safe() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import runpod_qat_lowram as rq
    # local-spark: ready with just a base model (no budget), bf16 + --qat, no aarch64-blocked deps.
    sp = rq.build_run_plan(base_model=rq.TIER_BASES["low"], gpu="-", scheme="nvfp4", budget_usd=None,
                           branch="b", epochs=1, calib="c.json", target_bits=4.5, target="local-spark")
    assert sp["ready_to_launch"] is True and sp["missing"] == []
    train = " ".join(next(s for s in sp["steps"] if s["stage"] == "qat_train")["command"])
    assert "tools/train_lora.py" in train and "--qat" in train and "--dtype bf16" in train
    assert not any(x in train for x in ("--4bit", "unsloth", "flash_attention_2", "--yes"))


def test_spec_from_block_counts_scales_with_layers() -> None:
    one = ModelSpec.from_block_counts("a", n_layers=1, hidden=128, vocab=1000,
                                      block_total_params=1_000_000, block_active_params=200_000,
                                      n_routed_experts=8, active_experts=2)
    ten = ModelSpec.from_block_counts("b", n_layers=10, hidden=128, vocab=1000,
                                      block_total_params=1_000_000, block_active_params=200_000,
                                      n_routed_experts=8, active_experts=2)
    # More layers -> more total params; sparsity ratio stays > 1 (MoE).
    assert ten.total_params > one.total_params
    assert ten.sparsity_ratio > 1.0
