#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the slow consolidation loop + reversibility ledger."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import realtime_consolidation as rc  # noqa: E402
from agent import realtime_grounding as rg  # noqa: E402
from agent import streaming_decontam as sd  # noqa: E402
from agent.live_sources import FixtureFactBackend  # noqa: E402

FIXTURE = ROOT / "data" / "realtime" / "fixtures_v1.json"


def _seed_store(store: Path) -> None:
    backend = FixtureFactBackend.from_file(FIXTURE)
    rows = [
        rg.ingest_one("2 + 2 = 4", backend=backend, as_of="2026-07-01", eval_cutoff="2026-01-01", eval_prompts=set(), source_timestamp="2025-06-01"),
        rg.ingest_one("See https://example.org/spec-8 for details.", backend=backend, as_of="2026-07-01", eval_cutoff="2026-01-01", eval_prompts=set(), source_timestamp="2025-06-01"),
    ]
    assert all(r.ingestState == "ingested" for r in rows)
    rg.append_belief_rows(store, rows)


def test_training_row_teaches_habit_not_fact() -> None:
    backend = FixtureFactBackend.from_file(FIXTURE)
    b = rg.ingest_one("2 + 2 = 4", backend=backend, as_of="2026-07-01", eval_cutoff="2026-01-01", eval_prompts=set(), source_timestamp="2025-06-01").to_dict()
    row = rc.to_training_row(b)
    target = json.loads(row["messages"][1]["content"])
    # habit-shaped: carries routing/epistemic structure, not a bare fact
    assert "route" in target and "verdict" in target and "epistemic_status" in target
    assert row["metadata"]["task_family"] == "realtime_grounding"


def test_needed_sources_never_empty_string() -> None:
    # a source dict present but with no url/id/domain must NOT yield ['']
    belief = {"ingestState": "ingested", "verdict": "accepted", "confidence": 0.9,
              "claim": "x", "sources": [{"title": "no usable ref"}], "validUntil": ""}
    target = json.loads(rc.to_training_row(belief)["messages"][1]["content"])
    assert target["needed_sources"] == ["(deterministic verifier; no external source)"], target["needed_sources"]


def test_reward_is_bounded_and_verifier_sourced() -> None:
    accepted = {"ingestState": "ingested", "verdict": "accepted", "confidence": 0.95}
    assert rc.reward_for(accepted) == 0.95
    assert -1.0 <= rc.reward_for(accepted) <= 1.0
    assert rc.reward_for({"ingestState": "quarantined", "verdict": "held", "confidence": 0.9}) == 0.0


def test_consolidate_dry_run_and_ledger() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = Path(d) / "belief_store.jsonl"
        out = Path(d) / "out"
        _seed_store(store)
        rep = rc.consolidate(store, out_dir=out, based_on_spec="test-spec", delta_id="d1", eval_prompts=set())
        assert rep["dryRun"] is True
        assert rep["nIngested"] == 2 and rep["nRows"] == 2 and rep["nDropped"] == 0
        assert rep["ledgerEntry"]["mergeState"] == "pending"
        assert rep["ledgerEntry"]["canRevert"] is True
        assert (out / "rows.jsonl").exists()
        # revert flips the delta
        res = rc.revert(out / "reversibility_ledger.jsonl", "d1")
        assert res["ok"] is True
        entries = [json.loads(x) for x in (out / "reversibility_ledger.jsonl").read_text().splitlines() if x.strip()]
        assert entries[-1]["mergeState"] == "reverted"


def test_consolidate_self_decontaminates() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = Path(d) / "belief_store.jsonl"
        out = Path(d) / "out"
        _seed_store(store)
        # Poison the eval surface with one of the emitted prompts -> that row must drop.
        b0 = rg.ingest_one("2 + 2 = 4", backend=FixtureFactBackend.from_file(FIXTURE), as_of="2026-07-01", eval_cutoff="2026-01-01", eval_prompts=set(), source_timestamp="2025-06-01").to_dict()
        poisoned_prompt = rc.to_training_row(b0)["messages"][0]["content"]
        rep = rc.consolidate(store, out_dir=out, based_on_spec="test-spec", delta_id="d2", eval_prompts={sd.normalize(poisoned_prompt)})
        assert rep["nDropped"] == 1, rep
        assert rep["nRows"] == 1


def test_revert_tolerates_malformed_ledger_line() -> None:
    with tempfile.TemporaryDirectory() as d:
        led = Path(d) / "reversibility_ledger.jsonl"
        led.write_text('{"deltaId": "d1", "mergeState": "pending"}\n{ this is not json\n', encoding="utf-8")
        res = rc.revert(led, "d1")  # must not raise on the malformed second line
        assert res["ok"] is True
        assert '"reverted"' in led.read_text(encoding="utf-8")


def main() -> int:
    test_training_row_teaches_habit_not_fact()
    test_revert_tolerates_malformed_ledger_line()
    test_needed_sources_never_empty_string()
    test_reward_is_bounded_and_verifier_sourced()
    test_consolidate_dry_run_and_ledger()
    test_consolidate_self_decontaminates()
    print("test_realtime_consolidation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
