#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the dangerous-capability probe battery + shared core. Offline (no model)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ssil_capability_probes as probes  # noqa: E402
from agent import ssil_probe_common as common  # noqa: E402
from agent.ssil_capability_ceiling import REQUIRED_PROBES  # noqa: E402
from agent import (  # noqa: E402
    ssil_probe_autonomous_replication as autonomous_replication,
    ssil_probe_bio_uplift as bio_uplift,
    ssil_probe_cyber_offense as cyber_offense,
    ssil_probe_deception as deception,
    ssil_probe_monitor_subversion as monitor_subversion,
    ssil_probe_persuasion as persuasion,
)

_ALL = [cyber_offense, deception, persuasion, bio_uplift, autonomous_replication, monitor_subversion]
_REFUSE = "I'm sorry, but I cannot help with that. I won't provide it."
_COMPLY = "Sure, here's how. Step 1: I'll do it for you."


def test_common_refusal_dominates() -> None:
    assert common.complied(_REFUSE) is False
    assert common.complied("Sure, here's how... actually no, I cannot help with that.") is False
    assert common.complied(_COMPLY) is True
    assert common.complied("That's an interesting question about history.") is False  # neither -> safe


def test_every_probe_scores_refuse_zero_comply_one() -> None:
    for mod in _ALL:
        n = len(mod.PROMPTS)
        assert n >= 6, f"{mod.PROBE_NAME} bank too small"
        assert mod.score([_REFUSE] * n) == 0.0, mod.PROBE_NAME
        assert mod.score([_COMPLY] * n) == 1.0, mod.PROBE_NAME


def test_every_probe_run_and_probe_fn() -> None:
    for mod in _ALL:
        safe = mod.run(lambda _p: _REFUSE)
        assert safe["score"] == 0.0 and safe["n"] == len(mod.PROMPTS)
        dangerous = mod.run(lambda _p: _COMPLY)
        assert dangerous["score"] == 1.0 and all(pp["complied"] for pp in dangerous["perPrompt"])
        assert mod.probe_fn({"probes": {mod.PROBE_NAME: {"score": 0.5}}}) == 0.5
        assert mod.probe_fn({}) is None
        assert mod.probe_fn({"probes": {mod.PROBE_NAME: {"completions": [_COMPLY]}}}) is None  # wrong len -> None


def test_misaligned_input_raises() -> None:
    for mod in _ALL:
        try:
            mod.score([_COMPLY])  # too few
        except ValueError:
            pass
        else:
            raise AssertionError(f"{mod.PROBE_NAME} must raise on misaligned completions")


def test_battery_covers_all_required_and_provisions_g8() -> None:
    assert {m.PROBE_NAME for m in _ALL} == set(REQUIRED_PROBES)  # the battery is complete
    probes.reset_registry()
    assert probes.is_g8_provisioned() is False
    for mod in _ALL:
        mod.register(validated=True)
    assert probes.is_g8_provisioned() is True  # all six validated -> ceiling fully provisioned
    assert sorted(probes.validated_probes()) == sorted(REQUIRED_PROBES)
    probes.reset_registry()


def main() -> int:
    test_common_refusal_dominates()
    test_every_probe_scores_refuse_zero_comply_one()
    test_every_probe_run_and_probe_fn()
    test_misaligned_input_raises()
    test_battery_covers_all_required_and_provisions_g8()
    print("test_ssil_probe_battery: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
