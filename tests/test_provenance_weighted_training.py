# SPDX-License-Identifier: Apache-2.0
"""Tests for W3 provenance-weighted training. Binds to the REAL agent.source_ranking."""
import importlib.util
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "provenance_weighted_training.py"


def _load():
    sys.modules.pop("w3tool", None)
    spec = importlib.util.spec_from_file_location("w3tool", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _examples():
    return [
        {"id": "e1", "text": "a", "source": "okf://belief/1", "domain": "science"},
        {"id": "e2", "text": "b", "source": "randomblog.example.com/x", "domain": "news"},
        {"id": "e3", "text": "c", "source": "training/corpus.jsonl", "domain": "math"},
    ]


def _skip_if_no_repo(m):
    if not m._REPO_OK:
        import pytest
        pytest.skip("agent.source_ranking unavailable")


def test_high_provenance_gets_higher_weight():
    m = _load(); _skip_if_no_repo(m)
    r = m.run(_examples(), floor=0.1)
    assert r["ok"] is True
    w = {x["id"]: x["weight"] for x in r["weights"]}
    # OKF belief-graph source must outrank a random blog
    assert w["e1"] > w["e2"]


def test_curriculum_orders_high_provenance_first():
    m = _load(); _skip_if_no_repo(m)
    r = m.run(_examples(), floor=0.1)
    assert r["curriculumOrder"][0] == "e1"  # highest trust first


def test_floor_bounds_low_trust_weight():
    m = _load(); _skip_if_no_repo(m)
    r = m.run(_examples(), floor=0.3)
    assert all(x["weight"] >= 0.3 for x in r["weights"])  # nothing below the floor


def test_influence_fingers_shared_low_trust_source():
    m = _load(); _skip_if_no_repo(m)
    r = m.run(_examples(), floor=0.1,
              eval_item={"id": "q", "source": "randomblog.example.com/x"})
    top = r["influence"]["implicatedTrainRows"][0]
    assert top["id"] == "e2"  # the blog-sourced row is most implicated
    assert top["sharesEvalSource"] is True


def test_bad_floor_fails_closed():
    m = _load(); _skip_if_no_repo(m)
    assert m.run(_examples(), floor=1.5)["ok"] is False


def test_empty_examples_fail_closed():
    m = _load(); _skip_if_no_repo(m)
    assert m.run([])["ok"] is False