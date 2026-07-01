# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the MegaTrain memory-centric *training* planner.

Mirrors ``tests/test_layer_stream.py``: pure / offline / deterministic, NO GPU, NO torch. Proves the
byte-accounting (ceilings, double-buffer peak, optimizer-scheme shrink, activation recompute, overlap
schedule, 512k-context activation dominance) BEFORE any GPU run. PLANNING ONLY; canClaimAGI false.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import layer_stream_train as lst  # noqa: E402

GIB = lst.GIB


def _assert_raises(exc, fn) -> None:
    """pytest-free ValueError assertion so the file runs both under pytest and as a plain script."""
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_offline_invariants() -> None:
    ok, detail = lst.offline_invariants()
    assert ok, detail["checks"]


def test_ceiling_128gb_is_about_8B() -> None:
    """128 GiB adam-fp32 (16 B/param) → ~8B trainable params, within ~10%."""
    c = lst.ceiling_params(128 * GIB, "adam-fp32")
    assert abs(c - 8e9) / 8e9 <= 0.10


def test_ceiling_512gb_is_about_32B() -> None:
    """512 GiB adam-fp32 → ~32B trainable params, within ~10%."""
    c = lst.ceiling_params(512 * GIB, "adam-fp32")
    assert abs(c - 32e9) / 32e9 <= 0.10


def test_double_buffer_peak_uses_depth_not_nlayers() -> None:
    """The KEY MegaTrain property: peak DEVICE bytes = depth layers + activations, NOT the model."""
    params, nlayers = 8_000_000_000, 32
    layer_params = params // nlayers
    act = lst.activation_bytes(1, 4096, 4096, nlayers, 2)
    pdb = lst.peak_device_bytes(layer_params, "adam-fp32", act, double_buffer_depth=2)
    # exactly two layers' (weight+grad) plus activations — never nlayers' worth
    assert pdb == 2 * layer_params * (4 + 4) + act
    # peak is far below the whole-model host residence
    assert pdb < lst.host_bytes(params, "adam-fp32") // 4


def test_more_buffer_depth_raises_peak_linearly() -> None:
    lp = 100_000_000
    act = 1_000_000
    p2 = lst.peak_device_bytes(lp, "adam-fp32", act, 2)
    p4 = lst.peak_device_bytes(lp, "adam-fp32", act, 4)
    assert (p4 - act) == 2 * (p2 - act)  # param window doubles with depth, activation term fixed


def test_galore_shrinks_bytes_per_param_and_raises_ceiling() -> None:
    base = lst.ceiling_params(128 * GIB, "adam-fp32")
    galore = lst.ceiling_params(128 * GIB, "galore", rank_ratio=0.25)
    assert lst.optimizer_bytes_per_param("galore", rank_ratio=0.25) < 16.0
    assert galore > base
    # rank_ratio 1.0 degenerates back to full Adam
    assert lst.optimizer_bytes_per_param("galore", rank_ratio=1.0) == 16.0


def test_lora_shrinks_bytes_per_param_and_raises_ceiling() -> None:
    base = lst.ceiling_params(128 * GIB, "adam-fp32")
    lora = lst.ceiling_params(128 * GIB, "lora", lora_param_ratio=0.005)
    assert lst.optimizer_bytes_per_param("lora", lora_param_ratio=0.005) < 16.0
    assert lora > base


def test_sgd_momentum_cheaper_than_adam() -> None:
    assert lst.optimizer_bytes_per_param("sgd-momentum") == 12.0
    assert lst.optimizer_bytes_per_param("sgd-momentum") < lst.optimizer_bytes_per_param("adam-fp32")


def test_activation_recompute_reduces_activation_bytes() -> None:
    full = lst.activation_bytes(1, 8192, 4096, 64, 2, recompute=False)
    recomp = lst.activation_bytes(1, 8192, 4096, 64, 2, recompute=True)
    assert recomp < full


def test_overlap_efficiency_buffered_vs_serial() -> None:
    """> 1 when overlapped (depth >= 2); ~1 when serial (depth 1); capped at 2."""
    assert lst.overlap_efficiency(100.0, 100.0, 2) > 1.0
    assert abs(lst.overlap_efficiency(100.0, 100.0, 1) - 1.0) < 1e-9
    assert lst.overlap_efficiency(50.0, 200.0, 2) <= 2.0
    # balanced compute/transfer overlapped → near-2x (best case)
    assert abs(lst.overlap_efficiency(100.0, 100.0, 2) - 2.0) < 1e-9


def test_determinism() -> None:
    a = lst.fits(8_000_000_000, 32, 128 * GIB, "adam-fp32", lst.ActivationCfg(), 2)
    b = lst.fits(8_000_000_000, 32, 128 * GIB, "adam-fp32", lst.ActivationCfg(), 2)
    assert a == b
    assert lst.ceiling_params(128 * GIB, "adam-fp32") == lst.ceiling_params(128 * GIB, "adam-fp32")


def test_8B_fits_128gb_with_recompute() -> None:
    """The pre-registered first deliverable: 8B full-precision adam-fp32 fits 128 GiB under
    double-buffered streaming WITH activation recomputation (the composing MegaTrain lever)."""
    r = lst.fits(8_000_000_000, 32, 128 * GIB, "adam-fp32",
                 lst.ActivationCfg(recompute=True), 2)
    assert r["fits"]
    assert r["headroomBytes"] > 0


def test_512k_context_activations_dominate() -> None:
    """At 512k context the activation working set dwarfs the streaming param/grad window."""
    nlayers = 32
    lp = 8_000_000_000 // nlayers
    act = lst.activation_bytes(1, 524288, 4096, nlayers, 2, recompute=True)
    param_window = 2 * lp * (4 + 4)
    assert act > param_window
    assert act > 100 * GIB  # hundreds of GiB — this is the long-context wall


def test_host_bytes_consistent_with_bytes_per_param() -> None:
    for scheme in ("adam-fp32", "adam-bf16-master", "sgd-momentum"):
        bpp = lst.optimizer_bytes_per_param(scheme)
        assert lst.host_bytes(1_000_000_000, scheme) == int(round(1_000_000_000 * bpp))


def test_fail_closed_on_bad_inputs() -> None:
    _assert_raises(ValueError, lambda: lst.optimizer_bytes_per_param("nope"))
    _assert_raises(ValueError, lambda: lst.host_bytes(0, "adam-fp32"))
    _assert_raises(ValueError, lambda: lst.optimizer_bytes_per_param("galore", rank_ratio=1.5))
    _assert_raises(ValueError, lambda: lst.peak_device_bytes(100, "adam-fp32", 0, double_buffer_depth=0))
    _assert_raises(ValueError, lambda: lst.ceiling_params(0, "adam-fp32"))


def test_cli_self_test_and_report() -> None:
    assert lst.main(["--self-test"]) == 0
    assert lst.main(["--report"]) == 0
    assert lst.main(["--params", "8e9", "--layers", "32", "--budget-gb", "128",
                     "--scheme", "adam-fp32", "--recompute"]) == 0


# Repo convention: a runnable main() that prints PASS/ok and exits non-zero on failure.
def _run() -> int:
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for f in funcs:
        try:
            f()
            print(f"  [ok] {f.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [XX] {f.__name__}: {e}")
    print("test_layer_stream_train:", "PASS" if failed == 0 else f"FAIL ({failed})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run())
