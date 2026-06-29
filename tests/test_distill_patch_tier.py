#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""T1 (two-teacher failure-patching) + T7 (verified self-consistency) + T9 (teacher split).

Uses tiny fake clients so the failure->recovery path is deterministic and offline (the
real mock teacher is deterministic per prompt and can't model 'weak fails, strong fixes')."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import distill_export as d  # noqa: E402


@dataclasses.dataclass
class _Cfg:
    model: str
    temperature: float = 0.2


@dataclasses.dataclass
class _Res:
    text: str
    model: str
    ok: bool = True
    cost_usd: float = 0.0


class _ScriptedClient:
    """Returns a fixed answer (or cycles through a list) regardless of prompt. The answer
    must clear the advisor gate, so we include the source-discipline markers the gate wants."""

    def __init__(self, model: str, answers, cost: float = 0.0):
        self.primary = _Cfg(model)
        self._answers = answers if isinstance(answers, list) else [answers]
        self._i = 0
        self._cost = cost

    def generate(self, system, user, **kw):
        a = self._answers[min(self._i, len(self._answers) - 1)]
        self._i += 1
        return _Res(text=a, model=self.primary.model, ok=True, cost_usd=self._cost)


# A gate-clean answer that routes/qualifies (so check_response passes) AND contains "Laozi".
_GOOD = ("Routing to source discipline: this attribution is contested/legendary. "
         "The Dao De Jing is traditionally attributed to Laozi, not Confucius. 据传 存疑。")
_BAD = "Confucius wrote it."  # missing 'Laozi' -> oracle fails


def _item():
    return {"id": "ddj", "prompt": "Who wrote the Dao De Jing?", "mustInclude": ["Laozi"]}


def test_patch_tier_recovers_and_mines_dpo() -> None:
    main = _ScriptedClient("weak/main", _BAD, cost=0.001)
    patch = _ScriptedClient("strong/patch", _GOOD, cost=0.01)
    data = d.distill([_item()], main, decontam=False, patch_client=patch)
    assert data["passedFirstTry"] == 0
    assert data["patched"] == 1
    assert data["accepted"] == 1
    assert data["rejected"] == 0  # the failure was recovered, dropped from rejected
    # the kept row is tagged patched_after_failure, teacher = the strong model
    meta = data["sft"][0]["metadata"]
    assert meta["verification_provenance"] == d.PROV_PATCHED
    assert meta["teacher"] == "strong/patch"
    # a DPO pair was mined: strong answer chosen, weak failure rejected
    assert len(data["dpoPairs"]) == 1
    pair = data["dpoPairs"][0]
    assert pair["chosen"] == _GOOD and pair["rejected"] == _BAD
    # T9 teacher split attributes the kept row to the patch teacher
    assert data["teacherSplit"] == {"strong/patch": 1}


def test_self_consistency_recovers_on_verified_majority() -> None:
    # main fails first try; the higher-temp vote yields 2 good of 3 -> verified majority keeps one.
    main = _ScriptedClient("weak/main", [_BAD, _GOOD, _BAD, _GOOD], cost=0.0)
    data = d.distill([_item()], main, decontam=False, self_consistency_n=3)
    assert data["selfConsistent"] == 1
    assert data["accepted"] == 1
    assert data["sft"][0]["metadata"]["verification_provenance"] == d.PROV_SELF_CONSISTENT


def test_self_consistency_minority_stays_rejected() -> None:
    # only 1 of 3 pass -> below majority -> not kept.
    main = _ScriptedClient("weak/main", [_BAD, _GOOD, _BAD, _BAD], cost=0.0)
    data = d.distill([_item()], main, decontam=False, self_consistency_n=3)
    assert data["selfConsistent"] == 0
    assert data["rejected"] == 1


def test_no_recovery_is_backward_compatible() -> None:
    main = _ScriptedClient("weak/main", _GOOD, cost=0.0)
    data = d.distill([_item()], main, decontam=False)
    assert data["passedFirstTry"] == 1 and data["patched"] == 0 and data["selfConsistent"] == 0
    assert data["dpoPairs"] == []


def main() -> int:
    test_patch_tier_recovers_and_mines_dpo()
    test_self_consistency_recovers_on_verified_majority()
    test_self_consistency_minority_stays_rejected()
    test_no_recovery_is_backward_compatible()
    print("test_distill_patch_tier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
