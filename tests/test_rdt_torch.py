# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Smoke + invariant tests for the PyTorch Recurrent-Depth Transformer (Phase 1).

Skipped automatically where torch is absent (e.g. the pure-Python nano CI lane); runs in
the GPU image / any torch env. These are the "cheap validation FIRST" gate the cost-guard
runbook requires before any pod is launched: shapes, finite gradients, the structural
spectral-radius guarantee, the recurrence actually being used, a tiny-batch overfit, and a
checkpoint round-trip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from pretraining.architecture.rdt_torch import RDT, RDTConfig, LTIInjection  # noqa: E402


def test_self_test_all_passed() -> None:
    """The module's own CPU self-test (dense+MoE forward/backward, spectral radius,
    depth-uses-loop, tiny-batch overfit, generate) must fully pass."""
    from pretraining.architecture.rdt_torch import self_test
    rep = self_test(verbose=False)
    assert rep["all_passed"], rep


def test_spectral_radius_below_one_for_random_params() -> None:
    """a = exp(-softplus(A_log)·softplus(log_dt)) ∈ (0,1) for ANY parameters — the
    structural stability guarantee, checked against hostile random initializations."""
    torch.manual_seed(0)
    for _ in range(50):
        lti = LTIInjection(32)
        with torch.no_grad():
            lti.A_log.normal_(0, 5)
            lti.log_dt.normal_(0, 5)
        assert lti.spectral_radius() < 1.0


def test_forward_shapes_and_loss() -> None:
    cfg = RDTConfig(vocab_size=48, d_model=32, n_heads=4, n_kv_heads=2, d_ff=64,
                    n_prelude=1, n_coda=1, n_loop=3, max_seq_len=16)
    model = RDT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    tgt = torch.randint(0, cfg.vocab_size, (2, 10))
    logits, loss, _ = model(idx, tgt)
    assert logits.shape == (2, 10, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_loop_count_changes_output() -> None:
    """Running more loops must change the logits — proof the recurrence is load-bearing,
    not a no-op (the depth knob is real, as in the nano study's extrapolation result)."""
    cfg = RDTConfig(vocab_size=48, d_model=32, n_heads=4, n_kv_heads=2, d_ff=64,
                    n_prelude=1, n_coda=1, n_loop=4, max_seq_len=16)
    model = RDT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        a, _, _ = model(idx, n_loop=2)
        b, _, _ = model(idx, n_loop=8)
    assert (a - b).abs().max() > 1e-4


def test_gqa_and_moe_paths_run() -> None:
    """Both the dense and fine-grained-MoE recurrent-block FFNs train one step."""
    for moe in (False, True):
        cfg = RDTConfig(vocab_size=40, d_model=32, n_heads=4, n_kv_heads=1, d_ff=64,
                        n_prelude=1, n_coda=1, n_loop=2, max_seq_len=16,
                        use_moe=moe, moe_n_experts=4, moe_top_k=2)
        model = RDT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        tgt = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(idx, tgt)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_halt_head_shape() -> None:
    """The halting head emits one score per loop per position — the confidence signal
    Phase 2 couples to the provenance gate."""
    cfg = RDTConfig(vocab_size=40, d_model=32, n_heads=4, n_kv_heads=2, d_ff=64,
                    n_prelude=1, n_coda=1, n_loop=5, max_seq_len=16, use_halt_head=True)
    model = RDT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 7))
    _, _, halt = model(idx, return_halt=True)
    assert halt.shape == (2, 5, 7)


def test_loop_trajectory_feeds_vgrd() -> None:
    """The bridge from the trained RDT to the Phase-2 gate: loop_trajectory() emits a plain
    per-loop logit list that vgrd_decide consumes to produce a fail-closed verdict."""
    from pretraining.architecture.vgrd import vgrd_decide
    cfg = RDTConfig(vocab_size=32, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128,
                    n_prelude=1, n_coda=1, n_loop=8, max_seq_len=16)
    model = RDT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 6))
    traj = model.loop_trajectory(idx, pos=-1, n_loop=8)
    assert len(traj) == 8 and len(traj[0]) == cfg.vocab_size
    d = vgrd_decide(traj, verify_fn=lambda a: True)
    assert d["verdict"] in {"accept", "abstain", "block"}


def test_pretrain_smoke_reduces_loss_and_roundtrips(tmp_path) -> None:
    """The end-to-end pretrain pipeline (data→loss→optimizer→checkpoint) drives loss
    below uniform on the hermetic synthetic stream and writes a loadable checkpoint."""
    from pretraining.architecture import rdt_pretrain as P
    args = type("A", (), {})()
    P._apply_smoke(args)
    args.lr = 6e-4
    args.min_lr = 6e-5
    args.grad_clip = 1.0
    args.dataset = "synthetic"
    args.seed = 0
    args.moe = False
    args.out = str(tmp_path / "rdt-smoke")
    rep = P.train(args)
    assert rep["loss_decreased"]
    assert rep["below_uniform"]
    assert rep["checkpoint_roundtrip_ok"]
    assert rep["final_spectral_radius"] < 1.0
