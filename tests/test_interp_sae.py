# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Interpretability M0 — TopK SAE tests (plain-script style, pure stdlib)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.sae import metrics  # noqa: E402
from interp.sae.model import TopKSAE  # noqa: E402
from interp.sae.synthetic import planted_activations  # noqa: E402


def test_topk_exact_l0_and_selection() -> None:
    # All-positive pre-activations → exactly k features fire, the k largest.
    sae = TopKSAE(4, 6, 3, seed=1)
    sae.W_enc = [[0.0] * 4 for _ in range(6)]
    sae.b_enc = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a, keep = sae.encode([0.0, 0.0, 0.0, 0.0])
    assert sum(1 for v in a if v != 0.0) == 3
    assert keep == [3, 4, 5]  # indices of the 3 largest pre-acts


def test_topk_never_exceeds_k() -> None:
    sae = TopKSAE(8, 20, 4, seed=2)
    X, _ = planted_activations(30, 8, 6, 3, seed=2)
    for x in X:
        a, keep = sae.encode(x)
        assert len(keep) <= 4
        assert all(v > 0.0 for v in (a[h] for h in keep))


def test_decoder_columns_unit_norm() -> None:
    sae = TopKSAE(12, 16, 2, seed=0)
    assert max(abs(n - 1.0) for n in sae.decoder_norms()) < 1e-9
    X, _ = planted_activations(64, 12, 6, 2, seed=0)
    sae.fit(X, steps=20, lr=0.5)  # stays unit-norm after updates
    assert max(abs(n - 1.0) for n in sae.decoder_norms()) < 1e-6


def test_reconstructs_planted_signal() -> None:
    X, _ = planted_activations(160, 12, 6, 2, seed=0, noise=0.02)
    sae = TopKSAE(12, 16, 2, seed=0)
    fvu_init = metrics.fvu(X, sae.reconstruct_batch(X))
    sae.fit(X, steps=600, lr=0.5)
    fvu_final = metrics.fvu(X, sae.reconstruct_batch(X))
    assert fvu_final < 0.40, f"fvu_final={fvu_final}"
    assert fvu_final < 0.5 * fvu_init, f"init={fvu_init} final={fvu_final}"


def test_training_is_deterministic() -> None:
    X, _ = planted_activations(80, 12, 6, 2, seed=3)
    a = TopKSAE(12, 16, 2, seed=3)
    b = TopKSAE(12, 16, 2, seed=3)
    assert a.fit(X, steps=50, lr=0.5) == b.fit(X, steps=50, lr=0.5)


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"PASSED {len(tests)} interp-sae tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
