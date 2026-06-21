#!/usr/bin/env python3
"""Tests for the active-learning promotion loop (tools/promote_pending.py). Offline.

Exercises the full second half: a gate-miss candidate in the pending queue is
re-verified against ground truth, deduped, and (on --apply) written to the live
learned sink. Confirms a correct pen-name is NOT promoted (re-verify guard).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate_feedback import append_pending, candidate_record  # noqa: E402
from tools import promote_pending as pp  # noqa: E402


def _pending_with(*candidates):
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "pending.jsonl"
    for c in candidates:
        append_pending(c, path)
    return tmp, path


def test_promotes_confirmed_wrong_attribution() -> None:
    # "Napoleon wrote The Republic" — The Republic is in the snapshot (gold: Plato),
    # Napoleon is a recognized distinct author? grounded gate requires the claimed
    # author be a known distinct entity, so use a known wrong author: Hume.
    tmp, pending = _pending_with(candidate_record("The Social Contract", "David Hume"))
    sink = tmp / "learned.json"
    out = pp.promote(pending, sink, apply=True)
    assert len(out["promoted"]) == 1, out
    assert out["applied"] is True
    # the live sink now holds the record
    written = json.loads(sink.read_text())
    assert any("social_contract" in rid for rid in written), written


def test_does_not_promote_correct_penname() -> None:
    # "Mary Ann Evans wrote Middlemarch" is CORRECT (pen name George Eliot) — the
    # re-verify guard must refuse to promote it (cardinal: never forbid a correct one).
    tmp, pending = _pending_with(candidate_record("Middlemarch", "Mary Ann Evans"))
    sink = tmp / "learned.json"
    out = pp.promote(pending, sink, apply=True)
    assert len(out["promoted"]) == 0, out
    assert len(out["skippedUnconfirmed"]) == 1, out


def test_dry_run_does_not_write() -> None:
    tmp, pending = _pending_with(candidate_record("The Social Contract", "David Hume"))
    sink = tmp / "learned.json"
    out = pp.promote(pending, sink, apply=False)
    assert out["applied"] is False
    assert not sink.exists()  # dry run writes nothing


def test_dedupe_against_repeats() -> None:
    # same miss twice -> append_pending dedupes to 1, promote yields 1
    tmp, pending = _pending_with(
        candidate_record("The Social Contract", "David Hume"),
        candidate_record("The Social Contract", "David Hume"),
    )
    sink = tmp / "learned.json"
    out = pp.promote(pending, sink, apply=True)
    assert len(out["promoted"]) == 1, out


def test_closes_the_loop_live_gate_catches_after_promotion() -> None:
    # After promotion, a fresh provenance gate built from the live records (which now
    # include the learned sink) must fire on the promoted misattribution.
    tmp, pending = _pending_with(candidate_record("The Social Contract", "David Hume"))
    # use a sink path the live loader will actually read is not safe in tests; instead
    # verify the synthesized record makes provenance_faithful fire directly.
    out = pp.promote(pending, tmp / "learned.json", apply=True)
    rec = out["promoted"][0]["record"]
    from agent.verifiers import provenance_faithful

    records = {out["promoted"][0]["rid"]: rec}
    gate = provenance_faithful(records)
    assert gate("David Hume wrote The Social Contract.", None, {})["passed"] is False


def main() -> int:
    test_promotes_confirmed_wrong_attribution()
    test_does_not_promote_correct_penname()
    test_dry_run_does_not_write()
    test_dedupe_against_repeats()
    test_closes_the_loop_live_gate_catches_after_promotion()
    print("test_promote_pending: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
