#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Issue->skill bridge: grouping, class guard, dry-run forging via the verifier gate."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import skill_from_failure as sff  # noqa: E402

ROWS = [
    {"task": "destructive-command", "text": "delete the production database now", "label": True},
    {"task": "destructive-command", "text": "delete every row in users", "label": True},
    {"task": "destructive-command", "text": "delete the backups folder", "label": True},
    {"task": "destructive-command", "text": "delete all snapshots permanently", "label": True},
    {"task": "destructive-command", "text": "read the production database now", "label": False},
    {"task": "destructive-command", "text": "list every row in users", "label": False},
    {"task": "destructive-command", "text": "back up the backups folder", "label": False},
    {"task": "destructive-command", "text": "show all snapshots", "label": False},
]


def test_class_guard_rejects_single_class():
    assert sff._classes_ok(ROWS) is True
    assert sff._classes_ok(ROWS[:4]) is False           # all-positive (one class)
    assert sff._classes_ok(ROWS[:2] + ROWS[4:5]) is False  # only 3 rows (too few)
    assert sff._classes_ok(ROWS[:3] + ROWS[4:5]) is True   # 4 rows, both classes -> ok


def test_grouping_by_task():
    rows = ROWS + [{"task": "other", "text": "x", "label": True}]
    g = sff._group(rows)
    assert set(g) == {"destructive-command", "other"}
    assert len(g["destructive-command"]) == 8


def test_dry_run_forges_but_writes_nothing():
    before = set(p.name for p in (ROOT / "skills" / "registry").glob("*.json"))
    results = sff.run(ROWS, apply=False)
    assert len(results) == 1
    r = results[0]
    assert r["status"] == "forged" and r["promoted"] is True, r
    # dry-run must not touch the committed registry
    after = set(p.name for p in (ROOT / "skills" / "registry").glob("*.json"))
    assert before == after, (before ^ after)


def test_underspecified_group_is_skipped_not_forged():
    results = sff.run(ROWS[:4], apply=False)  # all-positive -> cannot separate
    assert results[0]["status"] == "skipped", results


def main() -> int:
    test_class_guard_rejects_single_class()
    test_grouping_by_task()
    test_dry_run_forges_but_writes_nothing()
    test_underspecified_group_is_skipped_not_forged()
    print("test_skill_from_failure: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
