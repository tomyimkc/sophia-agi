# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Versioned scoring rubrics + inter-rater agreement (pure stdlib).

A capability rubric is a versioned, hashed JSON artifact (so a score is tied to a
specific rubric revision), and any LLM-judge grading reports inter-rater agreement
(Cohen's κ) against a gold/second-rater set — reusing the repo's `cohen_kappa` so
the κ floor (0.40) matches the rest of the codebase.
"""
from __future__ import annotations

import json
from pathlib import Path

from provenance_bench.consensus import cohen_kappa

REQUIRED_FIELDS = ("name", "version", "criteria")


def validate_rubric(rubric: dict) -> dict:
    """Assert a rubric has the required shape; return it. Fail-closed on malformed."""
    if not isinstance(rubric, dict):
        raise ValueError("rubric must be a dict")
    missing = [f for f in REQUIRED_FIELDS if f not in rubric]
    if missing:
        raise ValueError(f"rubric missing required fields: {missing}")
    if not isinstance(rubric["criteria"], list) or not rubric["criteria"]:
        raise ValueError("rubric.criteria must be a non-empty list")
    return rubric


def load_rubric(path: "str | Path") -> dict:
    return validate_rubric(json.loads(Path(path).read_text(encoding="utf-8")))


def inter_rater_kappa(a_labels: "list[int]", b_labels: "list[int]") -> "float | None":
    """Cohen's κ between two raters' binary labels (None when undefined)."""
    return cohen_kappa(a_labels, b_labels)
