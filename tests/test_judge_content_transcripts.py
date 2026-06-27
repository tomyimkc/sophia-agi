# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The 2-judge content-transcript labeler (tools/judge_content_transcripts)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json  # noqa: E402
from tools import judge_content_transcripts as J  # noqa: E402


def test_parse_content_pass():
    assert J.parse_content_pass('{"content_pass": true}') is True
    assert J.parse_content_pass('blah {"content_pass": false} trailing') is False
    assert J.parse_content_pass("not json") is False
    assert J.parse_content_pass("") is False


def test_family_keys_match_aggregator():
    # The dict keys this tool emits must match the aggregator's _family_key on the same spec.
    from tools.run_lora_uplift_validation import _family_key
    for spec in ("deepseek:deepseek-chat", "openrouter:meta-llama/llama-3.1-70b-instruct"):
        assert J._family(spec) == _family_key(spec)


def test_judge_seeds_orchestration_with_injected_judges():
    d = Path(tempfile.mkdtemp())
    ids = []
    for dom in J.DOMAINS:
        cases = load_json(DOMAIN_BENCH[dom]).get("cases", [])[:2]
        ids += [c["id"] for c in cases]
        (d / f"local-base-{dom}.json").write_text(
            json.dumps({"responses": {c["id"]: c["question"] + " |PASS" for c in cases}}))
        (d / f"local-adp-{dom}.json").write_text(
            json.dumps({"responses": {c["id"]: c["question"] + (" |PASS" if i == 0 else " |FAIL")
                                      for i, c in enumerate(cases)}}))
    judges = ["deepseek:deepseek-chat", "openrouter:meta-llama/llama-3.1-70b-instruct"]
    fns = {J._family(s): (lambda q, r, a: a.endswith("|PASS")) for s in judges}
    out = J.judge_seeds([{"seed": 0, "dir": str(d)}], judges=judges,
                        base_label="base", adapter_label="adp", judge_fns=fns)
    items = out["seeds"][0]["items"]
    assert len(items) == len(ids)
    # base answers all PASS; adapter passes the first case per domain only (4 of 8).
    assert all(all(it["baseContent"].values()) for it in items)
    assert sum(1 for it in items if all(it["adapterContent"].values())) == len(J.DOMAINS)
    # keys are the two distinct vendor families (gateway-aware: meta-llama, not llama)
    assert set(items[0]["baseContent"]) == {"deepseek", "meta-llama"}


def test_mock_run_shape():
    rep = J._mock_run()
    assert len(rep["seeds"]) == 3
    assert rep["subjectModel"].startswith("mock")
    assert len(rep["judges"]) == 2
