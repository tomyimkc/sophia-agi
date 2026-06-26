#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the M2 teacher->gate->admission pipeline (build_sophia_wisdom_dataset).

Covers: synthesized candidates pass the live gate (admission is real, not bypassed),
gate-rejected candidates are actually rejected, route-first JSON is well-formed, and
preference pairs pair a gated chosen against a fabricated rejected.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(modname: str, relpath: str):
    spec = importlib.util.spec_from_file_location(modname, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M2 = _load("build_sws", "tools/build_sophia_wisdom_dataset.py")


def test_synthesized_candidates_mostly_pass_the_gate() -> None:
    """The templates are calibrated to the gate; the vast majority must be admitted.
    This proves admission is real (the gate runs) AND the teacher is well-formed."""
    cands = M2.synthesize()
    assert len(cands) > 200
    verdicts = [M2.admit(c) for c in cands]
    passed = sum(1 for v in verdicts if v["ok"])
    rate = passed / len(verdicts)
    assert rate > 0.9, f"admission rate too low ({rate:.2f}) — templates drifted from the gate"


def test_route_first_json_is_wellformed() -> None:
    for c in M2.synthesize()[:50]:
        a = next(m["content"] for m in c["messages"] if m["role"] == "assistant")
        head = a.split("\n", 1)[0]
        obj = json.loads(head)
        assert obj["route"] in ("allow", "revise", "retrieve", "clarify", "escalate", "abstain", "block")
        assert "中文摘要" in a, "every synthesized answer must carry a 中文 summary"


def test_gate_actually_rejects_a_fabrication() -> None:
    """A fabricated false-attribution answer must FAIL admission — the pipeline is not a
    rubber stamp."""
    row = {
        "messages": [
            {"role": "system", "content": M2.SYSTEM},
            {"role": "user", "content": "Did Confucius write the Dao De Jing?"},
            {"role": "assistant", "content": "Yes, Confucius wrote the Dao De Jing himself."},
        ],
        "metadata": {"task_family": "source_discipline", "domain": "philosophy",
                     "teacher": "deterministic-template"},
    }
    v = M2.admit(row)
    assert v["ok"] is False, "a bare fabrication with no source discipline / 中文 must be rejected"


def test_preference_pairs_oppose_chosen_and_rejected() -> None:
    accepted = [c for c in M2.synthesize() if M2.admit(c)["ok"]]
    pairs = M2.mine_preference_pairs(accepted)
    assert len(pairs) > 50
    for p in pairs[:30]:
        assert p["chosen"] != p["rejected"]
        assert p["prompt"] and p["chosen"] and p["rejected"]
        # chosen is the gated answer (route-first JSON); rejected is the fabrication
        assert p["chosen"].lstrip().startswith("{")


def test_teacher_prompt_bank_is_wellformed_and_distinct() -> None:
    """The live-teacher seed prompts must carry the gate/mix metadata and be framed
    distinctly from the deterministic templates (so they are net-new after dedup)."""
    bank = M2.teacher_prompt_bank()
    assert len(bank) > 100
    for s in bank:
        assert s["user"] and s["family"] and s["language"] in ("en", "zh")
        assert s["expected_route"] in ("allow", "revise", "retrieve", "clarify", "escalate", "abstain", "block")
    tmpl_prompts = {next(m["content"] for m in c["messages"] if m["role"] == "user")
                    for c in M2.synthesize()}
    bank_prompts = {s["user"] for s in bank}
    # the bank must contribute prompts the templates do not already cover
    assert len(bank_prompts - tmpl_prompts) > 50


def test_teacher_rows_flow_through_admission_gate() -> None:
    """generate_with_teacher must wrap teacher answers as candidates that are
    SUBJECT to admit() (not passed through), and a fabricating teacher is rejected."""
    bank = M2.teacher_prompt_bank()[:3]
    rows = M2.generate_with_teacher(bank, "mock", max_workers=1)
    assert rows and all(r["metadata"]["teacher"] == "mock" for r in rows)
    # teacher rows are NOT curated-reuse / retention, so admit() actually runs the gate
    for r in rows:
        v = M2.admit(r)
        assert "verdict" in v and v["verdict"] not in ("curated_reuse", "retention_passthrough")
    # a fabricating teacher row must be rejected by admission
    fab = {
        "messages": [
            {"role": "system", "content": M2.SYSTEM},
            {"role": "user", "content": bank[0]["user"]},
            {"role": "assistant", "content": "Yes, absolutely, that is a certain and well-established fact."},
        ],
        "metadata": {"task_family": "source_discipline", "domain": "philosophy", "teacher": "mock"},
    }
    assert M2.admit(fab)["ok"] is False


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    raise SystemExit(1 if failed else 0)
