#!/usr/bin/env python3
"""C4 tests: the continual loop is non-circular — only genuine gate misses are mined,
nothing promotes without an explicit human step, and promoted candidates convert to
well-formed source-discipline SFT rows."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.feedback_to_training as f2t  # noqa: E402
from agent.gate_feedback import detect_miss  # noqa: E402


def test_detect_miss_only_on_clean_and_hallucinated() -> None:
    miss = {"work": "The Wealth of Nations", "claimed_author": "David Hume",
            "gated_action": "clean", "gated": {"hallucinated": True}}
    assert detect_miss(miss) is not None
    # not a miss: gate already revised it
    assert detect_miss({**miss, "gated_action": "revised"}) is None
    # not a miss: judge did not flag a hallucination
    assert detect_miss({**miss, "gated": {"hallucinated": False}}) is None
    # not actionable: no claimed author to forbid
    assert detect_miss({**miss, "claimed_author": ""}) is None


def test_default_deny_then_promote(tmp_path: Path | None = None) -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f2t.PENDING = Path(d) / "pending.jsonl"
        f2t.SFT_OUT = Path(d) / "sft.jsonl"
        f2t.PROMOTED_RECORDS = Path(d) / "promoted.jsonl"

        cand = f2t.make_candidate("The Wealth of Nations", "David Hume", mined_from="t")
        f2t._write_jsonl(f2t.PENDING, [cand])

        # build-sft before approval -> nothing (default deny)
        f2t.cmd_build_sft(type("A", (), {"dry_run": False})())
        assert f2t._read_jsonl(f2t.SFT_OUT) == []

        # human review step promotes it
        f2t.cmd_approve(type("A", (), {"rid": [cand["rid"]], "reviewer": "me", "note": "verified"})())
        promoted = [c for c in f2t._read_jsonl(f2t.PENDING) if c.get("promoted")]
        assert len(promoted) == 1 and promoted[0]["reviewer"] == "me"

        # now build-sft emits exactly one well-formed row
        f2t.cmd_build_sft(type("A", (), {"dry_run": False})())
        rows = f2t._read_jsonl(f2t.SFT_OUT)
        assert len(rows) == 1
        msgs = rows[0]["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert "David Hume" in msgs[1]["content"] and "The Wealth of Nations" in msgs[1]["content"]
        assert msgs[2]["content"].lower().startswith("no.")
        assert rows[0]["metadata"]["promoted"] is True


def test_mine_dedupes_and_skips_non_misses() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f2t.PENDING = Path(d) / "pending.jsonl"
        results = Path(d) / "results.jsonl"
        f2t._write_jsonl(results, [
            {"work": "Meditations", "claimed_author": "Seneca", "gated_action": "clean", "gated": {"hallucinated": True}},
            {"work": "Meditations", "claimed_author": "Seneca", "gated_action": "clean", "gated": {"hallucinated": True}},  # dup
            {"work": "Critique", "claimed_author": "Kant", "gated_action": "clean", "gated": {"hallucinated": False}},  # not a miss
        ])
        f2t.cmd_mine(type("A", (), {"case_results": str(results)})())
        pending = f2t._read_jsonl(f2t.PENDING)
        assert len(pending) == 1  # deduped, non-miss skipped
        assert pending[0]["promoted"] is False


def test_candidate_to_record_shape() -> None:
    cand = f2t.make_candidate("On Liberty", "Adam Smith", mined_from="t")
    rec = f2t.candidate_to_record(cand)
    (rid, body), = rec.items()
    assert rid == cand["rid"]
    assert "doNotAttributeTo" in body and body["canonicalTitleEn"] == "On Liberty"


def main() -> int:
    test_detect_miss_only_on_clean_and_hallucinated()
    test_default_deny_then_promote()
    test_mine_dedupes_and_skips_non_misses()
    test_candidate_to_record_shape()
    print("test_feedback_to_training: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
