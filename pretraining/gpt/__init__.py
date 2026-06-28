# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tier-0 "rasbt-faithful" from-scratch GPT — the bridge between the zero-dep
``pretraining/nano`` research model and the real-GPU fine-tuning tooling.

Two layers, on purpose:

- ``tokenizer`` — a **dependency-free** byte-level tokenizer that reserves the
  provenance special tokens (``<src>``, ``<abstain>``, …) *now*, so the vocab is
  stable when born-gated training (idea #1 in
  ``docs/06-Roadmap/From-Scratch-LLM-Brainstorm.md``) lands. Runs anywhere,
  including CI and the pure-Python charter path.
- ``model`` / ``train`` — a small, readable GPT in **PyTorch (optional dep)**,
  device-agnostic across the dev cluster (DGX Spark CUDA/bf16, Mac Studio M3
  MPS/MLX, CPU fallback). Imported lazily so this package stays importable —
  and CI-testable for the tokenizer — without torch installed.

Honesty posture (inherits the charter): every training run stamps
``canClaimAGI: false`` and headline numbers stay on x86 RunPod — a Spark/M3 is
the iteration tier, not the evidence tier (see ``docs/11-Platform/DGX-Spark.md``).
"""
from __future__ import annotations

from pretraining.gpt.tokenizer import ByteProvenanceTokenizer, PROVENANCE_SPECIALS

__all__ = ["ByteProvenanceTokenizer", "PROVENANCE_SPECIALS"]
