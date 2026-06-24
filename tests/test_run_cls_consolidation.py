#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools.run_cls_consolidation — offline selection for the GPU-gated CLS run.

Verifies the selector picks grounded, answer-bearing facts (a thin stub is excluded) and
that nothing is promoted offline (the script only emits a manifest). Synthetic pages via a
temp wiki dir; offline, deterministic, dependency-free.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import frontmatter  # noqa: E402
from tools.run_cls_consolidation import build_selection  # noqa: E402

_RICH_BODY = ("# analects (論語)\n\nThe Analects is a compiled record of conversations attributed "
              "to Confucius and his disciples.\n")
_THIN_BODY = "# loner\n\n- **Domain:** philosophy\n"


def _write(dirpath, pid, body):
    meta = {"id": pid, "pageType": "concept", "authorConfidence": "consensus"}
    (Path(dirpath) / f"{pid}.md").write_text(frontmatter.serialize(meta, body), encoding="utf-8")


def test_selects_answer_bearing_excludes_thin() -> None:
    with tempfile.TemporaryDirectory() as d:
        _write(d, "analects", _RICH_BODY)
        _write(d, "loner", _THIN_BODY)
        sel = build_selection(d, min_stable_snapshots=1)
        assert sel["grounded"] == 2
        assert "analects" in sel["selected"]        # answer-bearing -> selected
        assert "loner" not in sel["selected"]       # thin stub -> not worth distilling
        assert sel["gateCleared"] == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
