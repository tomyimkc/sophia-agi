# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Inference-serving scheduling simulator (offline, deterministic, pure stdlib).

A discrete, iteration-stepped SIMULATION of two scheduling policies — static
(request-level) batching vs continuous / iteration-level batching (Orca; Yu et al.
2022) — over a seeded workload, reporting TTFT / TPOT / throughput / goodput@SLO.
This is the CI-green M0 core: it demonstrates the scheduling + the serving metrics
without a GPU. The REAL engine (driving a model through the repo's Rust paged
prefix KV-cache, with PagedAttention block management and speculative decoding) is
the GPU milestone — see docs/06-Roadmap/frontier-readiness/04-inference-serving.md.

Honest scope: this is a simulator (like `clustersim/`), not a served model. It
predicts the *shape* of the static→continuous uplift; the measured number comes
from the real engine on the DGX Spark / RunPod.
"""
from serving.sim import engine, workload

__all__ = ["engine", "workload"]
