#!/usr/bin/env python3
"""Tests for the learning-under-distribution-shift experiment logic."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_learning_shift as shift  # noqa: E402


def test_promotion_gate_keeps_only_promoted() -> None:
    records = [
        {"recordId": "r1", "promoted": True},
        {"recordId": "r2", "promoted": False},
        {"recordId": "r3"},  # missing -> not promoted
    ]
    promoted, rejected = shift.apply_promotion_gate(records)
    assert [r["recordId"] for r in promoted] == ["r1"]
    assert {r["recordId"] for r in rejected} == {"r2", "r3"}


def test_append_only_changes_hash_and_keeps_protected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mem = Path(tmp) / "learning_shift.jsonl"
        diff = shift.append_learning_records([{"recordId": "r1", "text": "new fact", "promoted": True}], mem)
        assert diff["appended"] is True
        assert diff["oldHash"] is None
        assert diff["newHash"] is not None
        assert diff["appendedRecordIds"] == ["r1"]
        # protected knowledge files are untouched by appending to the learning log
        assert diff["protectedKnowledgeUnchanged"] is True
        # second append grows the file (append-only, never rewrite)
        first_lines = mem.read_text(encoding="utf-8").splitlines()
        shift.append_learning_records([{"recordId": "r2", "text": "another", "promoted": True}], mem)
        second_lines = mem.read_text(encoding="utf-8").splitlines()
        assert len(second_lines) == len(first_lines) + 1


def test_append_no_promoted_records_is_noop_hash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mem = Path(tmp) / "learning_shift.jsonl"
        diff = shift.append_learning_records([], mem)
        assert diff["appended"] is False
        assert diff["appendedRecordIds"] == []


def test_contamination_audit_flags_id_overlap() -> None:
    pre = {"cases": [{"id": "c1", "prompt": "old q"}]}
    post = {"cases": [{"id": "c1", "prompt": "new q"}]}  # shares id c1
    audit = shift.contamination_audit(pre, post, [], training_text="")
    assert audit["clean"] is False
    assert any("share case ids" in i for i in audit["issues"])


def test_contamination_audit_flags_verbatim_prompt_in_records() -> None:
    # an 8+ word post prompt copied verbatim into a learning record is regurgitation
    prompt = "name the first two signatories of the kessari accord today"
    pre = {"cases": [{"id": "p1", "prompt": "unrelated"}]}
    post = {"cases": [{"id": "q1", "prompt": prompt}]}
    records = [{"text": f"reference: {prompt}", "source": "x"}]
    audit = shift.contamination_audit(pre, post, records, training_text="")
    assert audit["clean"] is False
    assert any("verbatim in learning records" in i for i in audit["issues"])


def test_contamination_audit_flags_answer_token_in_training() -> None:
    # gain is not attributable to learning if the answer is already in training
    post = {"cases": [{"id": "q1", "prompt": "who signed?", "scoring": {"mustInclude": [{"match": "Velm"}]}}]}
    pre = {"cases": [{"id": "p1", "prompt": "pre"}]}
    audit = shift.contamination_audit(pre, post, [], training_text="the city of velm is famous")
    assert audit["clean"] is False
    assert any("already in training corpus" in i for i in audit["issues"])


def test_contamination_audit_does_not_flag_answer_in_learning_records() -> None:
    # answer tokens SHOULD appear in the learning records (that is the teaching)
    post = {"cases": [{"id": "q1", "prompt": "who signed?", "scoring": {"mustInclude": [{"match": "Velm"}]}}]}
    pre = {"cases": [{"id": "p1", "prompt": "pre"}]}
    records = [{"text": "Velm and Toran signed.", "source": "x"}]
    audit = shift.contamination_audit(pre, post, records, training_text="nothing relevant")
    assert audit["clean"] is True


def test_contamination_audit_clean_when_disjoint() -> None:
    pre = {"cases": [{"id": "p1", "prompt": "pre question"}]}
    post = {"cases": [{"id": "q1", "prompt": "fresh post question"}]}
    records = [{"text": "unrelated learned fact", "source": "y"}]
    audit = shift.contamination_audit(pre, post, records, training_text="nothing relevant here")
    assert audit["clean"] is True
    assert audit["issues"] == []


def test_contamination_audit_no_false_positive_on_short_generic_prompt() -> None:
    # a short generic prompt that coincidentally substrings the corpus must NOT flag
    post = {"cases": [{"id": "q1", "prompt": "what is 2 + 2?"}]}
    pre = {"cases": [{"id": "p1", "prompt": "pre"}]}
    audit = shift.contamination_audit(pre, post, [], training_text="... what is 2 + 2? ...")
    assert audit["clean"] is True


def main() -> int:
    test_promotion_gate_keeps_only_promoted()
    test_append_only_changes_hash_and_keeps_protected()
    test_append_no_promoted_records_is_noop_hash()
    test_contamination_audit_flags_id_overlap()
    test_contamination_audit_flags_verbatim_prompt_in_records()
    test_contamination_audit_flags_answer_token_in_training()
    test_contamination_audit_does_not_flag_answer_in_learning_records()
    test_contamination_audit_clean_when_disjoint()
    test_contamination_audit_no_false_positive_on_short_generic_prompt()
    print("test_learning_shift: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
