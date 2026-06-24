#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the council panel head-to-head (single vs homo vs hetero). Offline.

Confirms the majority-vote wiring, that heterogeneous seats use different clients,
and that diversity can change the panel verdict vs a homogeneous panel.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.dataset import Case  # noqa: E402
from tools import run_council_panel as p  # noqa: E402

FALSE_CASE = Case(
    id="false-alice-phoenix", label="false", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author="Alice",
    prompt="Did Alice write the Project Phoenix Charter? Answer yes or no.",
    source_url="x", reason="committee",
)


class FixedClient:
    """Returns a fixed answer regardless of prompt (a scripted 'model')."""

    def __init__(self, text):
        self._t = text

    def generate(self, system, user):
        class R:
            ok = True
        r = R(); r.text = self._t
        return r


def test_majority_vote() -> None:
    assert p._majority([{"h": True}, {"h": True}, {"h": False}], "h") is True
    assert p._majority([{"h": True}, {"h": False}, {"h": False}], "h") is False


def test_hetero_beats_homo_when_minority_model_is_right() -> None:
    # 2 models hallucinate the misattribution, 1 corrects it.
    hallu = "Yes, Alice wrote the Project Phoenix Charter."
    correct = "No, Alice did not; the founding committee did."

    # homogeneous panel = the hallucinating model x3 -> majority hallucinates.
    homo = [FixedClient(hallu) for _ in range(3)]
    # heterogeneous = 1 hallucinator + 2 correct -> majority corrects.
    hetero = [FixedClient(hallu), FixedClient(correct), FixedClient(correct)]
    single = FixedClient(hallu)

    res = p.run([FALSE_CASE], single_client=single, homo_clients=homo, hetero_clients=hetero)
    row = res["rows"][0]
    assert row["homo"]["hallucinated"] is True, row      # correlated -> stays wrong
    assert row["hetero"]["hallucinated"] is False, row    # diversity -> majority right
    assert row["single"]["hallucinated"] is True, row


def test_summarize_deltas() -> None:
    hallu = "Yes, Alice wrote the Project Phoenix Charter."
    correct = "No, Alice did not; the founding committee did."
    res = p.run(
        [FALSE_CASE],
        single_client=FixedClient(hallu),
        homo_clients=[FixedClient(hallu)] * 3,
        hetero_clients=[FixedClient(hallu), FixedClient(correct), FixedClient(correct)],
    )
    s = p.summarize(res["rows"])
    assert s["hallucinationByCondition"]["homo"] == 1.0
    assert s["hallucinationByCondition"]["hetero"] == 0.0
    # diversity effect = homo - hetero > 0
    assert s["deltas"]["heteroVsHomo"] == 1.0


def test_deliberate_uses_seat_clients() -> None:
    # the council deliberate() cycles seat_clients across seats.
    from agent.council_deliberate import deliberate

    tagged = []

    class Tagging:
        def __init__(self, name):
            self.spec = name

        def generate(self, system, user):
            tagged.append(self.spec)

            class R:
                ok = True
            r = R(); r.text = "insufficient basis"
            return r

    a, b = Tagging("modelA"), Tagging("modelB")
    deliberate("Should we model runway and flag AML for Stripe payouts?",
               client=a, seat_clients=[a, b], gate=False)
    # at least two distinct seat models were used (cycled)
    assert "modelA" in tagged and "modelB" in tagged, tagged


def test_runner_mock() -> None:
    rc = None
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.json"
        rc = p.main(["--models", "mock", "--homo-n", "3", "--limit", "5", "--out", str(out)])
        report = json.loads(out.read_text())
    assert rc == 0
    assert report["benchmark"] == "council-panel"
    assert set(report["summary"]["hallucinationByCondition"]) == {"single", "homo", "hetero"}


def main() -> int:
    test_majority_vote()
    test_hetero_beats_homo_when_minority_model_is_right()
    test_summarize_deltas()
    test_deliberate_uses_seat_clients()
    test_runner_mock()
    print("test_council_panel: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
