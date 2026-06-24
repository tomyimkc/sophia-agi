#!/usr/bin/env python3
"""C3 token-fitting tests: short rows pass through, multi-turn rows split at turn
boundaries, single overlong turns drop, and no emitted row ever exceeds the budget."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.split_long_training_rows import fit_rows, heuristic_counter  # noqa: E402

COUNT = heuristic_counter()


def _row(*turns: tuple[str, str]) -> dict:
    return {"messages": [{"role": r, "content": c} for r, c in turns]}


def test_short_row_kept_unchanged() -> None:
    rows = [_row(("system", "be honest"), ("user", "who wrote X?"), ("assistant", "uncertain"))]
    out, rep = fit_rows(rows, max_tokens=1024, count=COUNT)
    assert rep["rowsKeptUnchanged"] == 1 and rep["rowsOut"] == 1
    assert out[0] == rows[0]


def test_multiturn_overlong_splits_and_fits() -> None:
    big = "x" * 1000  # ~290 tokens each; one (user,assistant) pair (~585) fits in 700, two don't
    row = _row(
        ("system", "sys"),
        ("user", big), ("assistant", big),
        ("user", big), ("assistant", big),
    )
    out, rep = fit_rows([row], max_tokens=700, count=COUNT)
    assert rep["rowsSplit"] == 1 and rep["subRowsFromSplits"] >= 2
    # every emitted row fits, and each carries the system preamble
    for r in out:
        assert COUNT(r["messages"]) <= 700
        assert r["messages"][0]["role"] == "system"


def test_single_overlong_turn_dropped() -> None:
    huge = "y" * 100_000
    row = _row(("system", "sys"), ("user", huge), ("assistant", huge))
    out, rep = fit_rows([row], max_tokens=512, count=COUNT)
    assert out == []
    assert rep["rowsDroppedUnsplittable"] == 1


def test_invariant_no_overlong_output() -> None:
    rows = [
        _row(("user", "a" * n), ("assistant", "b" * n))
        for n in (10, 500, 5000, 50_000)
    ]
    out, rep = fit_rows(rows, max_tokens=600, count=COUNT)
    assert all(COUNT(r["messages"]) <= 600 for r in out)
    assert rep["rowsIn"] == 4 and rep["rowsOut"] == len(out)


def main() -> int:
    test_short_row_kept_unchanged()
    test_multiturn_overlong_splits_and_fits()
    test_single_overlong_turn_dropped()
    test_invariant_no_overlong_output()
    print("test_split_long_training_rows: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
