# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Mechanistic-interpretability package (Sophia-AGI).

Sparse dictionary features over the local model's residual stream, in service of
honesty. The OFFLINE core (this package's `sae.model`, `sae.metrics`,
`hookpoints`) is pure-stdlib, deterministic, and CI-green — mirroring
`agent/steering`'s pure-stdlib core. The GPU path (real Qwen2.5-7B harvesting +
SAELens/torch training) is gated behind `requirements-interp.txt` and runs on
RunPod / the DGX Spark; it never imports at module load here.

Roadmap: docs/06-Roadmap/frontier-readiness/03-interpretability.md (M0 = this).
Claim discipline: every result is falsifiable, fail-closed, candidate-only;
a null result is an accepted, reported outcome. Not AGI.
"""

__all__ = ["sae", "hookpoints"]
