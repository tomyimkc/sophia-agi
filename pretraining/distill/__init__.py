# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Distillation-into-sparsity study on the known-floor nano substrate.

Answers a falsifiable question: at matched *active* compute, does a sparse MoE student
distilled from a dense teacher beat an equally-active dense student — i.e. does
sparsity+distillation recover teacher quality at a fraction of the active (RAM-at-release)
cost? See ``study.py``.
"""
from __future__ import annotations

from pretraining.distill.study import (
    offline_invariants,
    run_study,
    teacher_relabel,
)

__all__ = ["offline_invariants", "run_study", "teacher_relabel"]
