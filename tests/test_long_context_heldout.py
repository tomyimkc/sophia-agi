#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the held-out, non-synthetic long-context recall eval."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_hidden_eval_sophia import RunConfig  # noqa: E402
from tools import run_long_context_heldout as heldout  # noqa: E402
from tools.run_long_context_sophia import GATED_PACKED_MODE, long_context_modes  # noqa: E402


def test_corpus_is_nonsynthetic_and_integrity_holds() -> None:
    items = heldout.load_items()
    assert len(items) >= 5
    integrity = heldout.corpus_integrity(items)
    assert integrity["clean"], integrity["failures"]
    # Real prose, not templated synthetic filler: no LC_NEEDLE_* tokens anywhere.
    for item in items:
        blob = item["goldDoc"]["text"] + " ".join(d["text"] for d in item["distractorDocs"])
        assert "LC_NEEDLE" not in blob and "LC_DISTRACTOR" not in blob


def test_gold_doc_is_the_unique_answer_source() -> None:
    # Each gold answer token appears in the gold doc and in NONE of the distractors,
    # so correct recall requires surfacing the gold doc, not guessing from a distractor.
    for item in heldout.load_items():
        distractor_text = " ".join(d["text"] for d in item["distractorDocs"])
        for token in item["answerTokens"]:
            assert token in item["goldDoc"]["text"]
            assert token not in distractor_text


def test_heldout_runs_offline_under_mock_and_emits_valid_cards() -> None:
    items = heldout.load_items()
    modes = long_context_modes(GATED_PACKED_MODE)
    config = RunConfig(backend="mock", timeout_sec=5)
    case = heldout.build_case(items[0], budget_tokens=140, relevance_source="lexical")
    # gold passage carries the answer tokens; distractor passages do not.
    gold_passages = [p for p in case["longContextPassages"] if p["answerTokens"]]
    assert len(gold_passages) == 1
    ablation = next(iter(modes.values()))
    row = heldout.score_item(case, GATED_PACKED_MODE, ablation, config)
    assert row["cardErrors"] == []
    assert 0.0 <= row["recall"] <= 1.0  # under mock, recall ~0 (mock cannot read prose)


def test_report_is_candidate_with_mock_boundary() -> None:
    items = heldout.load_items()
    integrity = heldout.corpus_integrity(items)
    report = heldout.build_report(items, [], backend="mock", relevance_source="lexical",
                                  integrity=integrity, budget_tokens=140)
    assert report["reportStatus"] == "candidate"
    assert report["candidateOnly"] is True
    assert report["canClaimAGI"] is False
    assert "MOCK backend cannot read prose" in report["backendClaimBoundary"]


def test_adapter_skips_when_it_resolves_to_mock_fallback() -> None:
    # With no SOPHIA_MODEL_PROVIDER, --backend adapter resolves to adapter:mock. A real-model
    # run must NOT present those mock numbers as real: it must skip and write no report.
    import json
    import os
    import subprocess
    import tempfile

    env = dict(os.environ)
    env.pop("SOPHIA_MODEL_PROVIDER", None)
    env.pop("SOPHIA_MODEL_BASE_URL", None)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "should-not-exist.json"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "run_long_context_heldout.py"),
             "--backend", "adapter", "--timeout-sec", "10", "--out", str(out)],
            cwd=ROOT, text=True, capture_output=True, timeout=60, env=env, check=False,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout
        payload = json.loads(proc.stdout)
        assert payload["localModelUnavailable"] is True
        assert "mock" in payload["resolvedBackend"]
        assert not out.exists()


def main() -> int:
    test_corpus_is_nonsynthetic_and_integrity_holds()
    test_gold_doc_is_the_unique_answer_source()
    test_heldout_runs_offline_under_mock_and_emits_valid_cards()
    test_report_is_candidate_with_mock_boundary()
    test_adapter_skips_when_it_resolves_to_mock_fallback()
    print("test_long_context_heldout: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
