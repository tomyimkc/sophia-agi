#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Dry tests for model lab helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.model_lab_lib import (  # noqa: E402
    adapter_config,
    build_modelfile,
    distill_specs_from_attributions,
    sample_teacher_examples,
)
from sophia_mcp.tools_impl import corpus_stats  # noqa: E402


def test_sample_teacher_examples() -> None:
    paths = sample_teacher_examples(5)
    assert len(paths) == 5


def test_distill_specs() -> None:
    specs = distill_specs_from_attributions(10)
    assert len(specs) == 10
    assert "user" in specs[0]


def test_modelfile_build() -> None:
    cfg = adapter_config(ROOT / "training" / "lora" / "checkpoints" / "sophia-v1")
    text = build_modelfile(cfg)
    assert "FROM" in text
    assert "SYSTEM" in text


def test_corpus_stats() -> None:
    stats = corpus_stats()
    assert stats["trainingExamples"] >= 500


def main() -> int:
    test_sample_teacher_examples()
    test_distill_specs()
    test_modelfile_build()
    test_corpus_stats()
    print("test_model_lab: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
