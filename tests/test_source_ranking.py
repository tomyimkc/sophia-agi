#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.source_ranking import rank_source, rank_sources  # noqa: E402


def test_local_okf_ranks_high() -> None:
    assert rank_source("okf://belief-graph").rank >= 0.9


def test_model_only_rejected_by_grounding_threshold() -> None:
    out = rank_sources(["model:raw-llm"], min_rank=0.5)
    assert out["accepted"] == [] and out["rejected"]


def test_generic_source_backcompat_passes_low_risk_threshold() -> None:
    out = rank_sources(["tmpl"], min_rank=0.5)
    assert out["accepted"] == ["tmpl"]


def main() -> int:
    test_local_okf_ranks_high()
    test_model_only_rejected_by_grounding_threshold()
    test_generic_source_backcompat_passes_low_risk_threshold()
    print("test_source_ranking: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
