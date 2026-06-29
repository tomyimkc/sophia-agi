# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Powered Focus-Frontier battery — size/power, decontamination, single-axis, determinism."""
from __future__ import annotations

import json
from pathlib import Path

from tools.build_focus_battery import (
    BATTERY_PATH,
    DECONTAM_JACCARD,
    MDE_TARGET,
    build_battery,
    build_tasks,
    decontam_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def test_battery_is_powered():
    b = build_battery()
    assert b["publicN"] >= 100
    assert b["mdeAtPublicN"] <= MDE_TARGET
    assert b["powered"] is True


def test_private_split_is_powered_and_sealed():
    # The held-out private split must itself be powered so a sealed-split run can be valid.
    b = build_battery()
    assert b["privateN"] >= 100
    assert b["mdeAtPrivateN"] <= MDE_TARGET
    assert b["privatePowered"] is True


def test_tasks_are_unique_and_have_a_key():
    tasks = build_tasks()
    assert len({t["goal"] for t in tasks}) == len(tasks)  # all distinct
    for t in tasks:
        assert any(s.get("key") for s in t["segments"])  # every task has a solution key


def test_decontaminated_against_training_corpus():
    rec = decontam_receipt(build_tasks())
    assert rec["clean"] is True
    assert rec["maxJaccardVsTrain"] < DECONTAM_JACCARD


def test_has_goalshift_and_safety_subsets():
    b = build_battery()
    assert b["goalShiftCount"] > 0
    assert b["safetyCount"] > 0


def test_public_private_split_disjoint():
    tasks = build_tasks()
    pub = {t["id"] for t in tasks if t["split"] == "public"}
    priv = {t["id"] for t in tasks if t["split"] == "private"}
    assert pub and priv and not (pub & priv)


def test_deterministic_rebuild():
    assert build_battery() == build_battery()  # no RNG / timestamps


def test_committed_battery_matches_build():
    assert BATTERY_PATH.exists(), "run `python tools/build_focus_battery.py`"
    on_disk = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    assert on_disk == build_battery()
