# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Quantization-aware-training study on the known-floor nano substrate.

Answers a falsifiable question: does ternary-pushing QAT lower the *quantization gap*
(not the floor) against a computable ground-truth irreducible loss? See ``study.py``.
"""
from __future__ import annotations

from pretraining.qat.study import (
    offline_invariants,
    run_study,
    ternary_quantize_model,
    ternary_quantize_value,
    ternary_regularizer,
    train_qat,
)

__all__ = [
    "offline_invariants",
    "run_study",
    "ternary_quantize_model",
    "ternary_quantize_value",
    "ternary_regularizer",
    "train_qat",
]
