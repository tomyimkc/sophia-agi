#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for LLM thinking-step logging + A2A message logging + distillation (offline)."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import a2a_distill as ad  # noqa: E402
from agent import model as m  # noqa: E402
from agent import thinking_trace as tt  # noqa: E402


def _reset_capture(value: str | None) -> None:
    if value is None:
        os.environ.pop("SOPHIA_CAPTURE_THINKING", None)
    else:
        os.environ["SOPHIA_CAPTURE_THINKING"] = value


# --------------------------------------------------------------------------- #
# ModelResult: reasoning fields surface in to_log() (sizes, never verbatim)
# --------------------------------------------------------------------------- #
def test_model_result_to_log_reports_reasoning_size_not_text() -> None:
    r = m.ModelResult(text="answer", provider="p", model="x", reasoning_text="because A then B", reasoning_tokens=4)
    log = r.to_log()
    assert log["reasoningTokens"] == 4
    assert log["hasReasoning"] is True
    assert log["reasoningChars"] == len("because A then B")
    # The decision log never carries the raw reasoning text — that is the trace's job.
    assert "reasoning" not in log and "because" not in json.dumps(log)


# --------------------------------------------------------------------------- #
# Anthropic thinking-param shape is model-aware and opt-in
# --------------------------------------------------------------------------- #
def test_thinking_param_off_by_default() -> None:
    _reset_capture(None)
    cfg = m.ModelConfig(kind="anthropic", model="claude-sonnet-4-6")
    assert m._anthropic_thinking_param(cfg) is None


def test_thinking_param_adaptive_for_modern_claude() -> None:
    _reset_capture("1")
    try:
        cfg = m.ModelConfig(kind="anthropic", model="claude-sonnet-4-6")
        param = m._anthropic_thinking_param(cfg)
        assert param == {"type": "adaptive", "display": "summarized"}
    finally:
        _reset_capture(None)


def test_thinking_param_budgeted_for_legacy_claude() -> None:
    _reset_capture("1")
    try:
        cfg = m.ModelConfig(kind="anthropic", model="claude-3-5-sonnet-20241022", max_tokens=2400)
        param = m._anthropic_thinking_param(cfg)
        assert param["type"] == "enabled" and 1024 <= param["budget_tokens"] < 2400
    finally:
        _reset_capture(None)


# --------------------------------------------------------------------------- #
# Trace writer: hash-only by default, verbatim under SOPHIA_CAPTURE_THINKING
# --------------------------------------------------------------------------- #
def test_record_generation_hash_only_by_default() -> None:
    _reset_capture(None)
    with tempfile.TemporaryDirectory() as tmp:
        tt.set_context(trace_id="t-hash")
        res = m.ModelResult(text="the answer", provider="mock", model="mock-1", reasoning_text="secret cot")
        rec = tt.record_generation("sys", "user", res, role="generate", runs_dir=Path(tmp))
        assert rec["kind"] == "llm_call" and rec["traceId"] == "t-hash"
        assert rec["reasoningHash"] and rec["userHash"]
        # No verbatim text when capture is off.
        assert "reasoning" not in rec and "user" not in rec and "answer" not in rec
        path = Path(tmp) / "t-hash.jsonl"
        assert path.exists() and "secret cot" not in path.read_text(encoding="utf-8")


def test_record_generation_verbatim_when_capture_on() -> None:
    _reset_capture("1")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tt.set_context(trace_id="t-verbatim")
            res = m.ModelResult(text="the answer", provider="mock", model="mock-1", reasoning_text="step1 step2")
            rec = tt.record_generation("sys", "user prompt", res, runs_dir=Path(tmp))
            assert rec["reasoning"] == "step1 step2" and rec["answer"] == "the answer"
            assert rec["user"] == "user prompt"
    finally:
        _reset_capture(None)


# --------------------------------------------------------------------------- #
# ModelClient choke-point sink fires once per successful generate
# --------------------------------------------------------------------------- #
def test_model_client_trace_sink_fires_on_generate() -> None:
    captured: list[dict] = []

    def sink(system: str, user: str, result) -> None:  # noqa: ANN001
        captured.append({"system": system, "ok": result.ok})

    client = m.ModelClient(m.resolve_config("mock"), trace_sink=sink)
    out = client.generate("be helpful", "do the thing")
    assert out.ok and len(captured) == 1 and captured[0]["ok"] is True


# --------------------------------------------------------------------------- #
# A2A message logging is opt-in (no-op unless SOPHIA_THINKING_LOG set)
# --------------------------------------------------------------------------- #
def test_maybe_record_a2a_is_noop_when_disabled() -> None:
    os.environ.pop("SOPHIA_THINKING_LOG", None)
    with tempfile.TemporaryDirectory() as tmp:
        tt.set_context(trace_id="t-off")
        tt.maybe_record_a2a(sender="a", receiver="b", prompt="p", response="r", runs_dir=Path(tmp))
        assert not list(Path(tmp).glob("*.jsonl"))


def test_record_a2a_message_writes_span() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tt.set_context(trace_id="t-a2a")
        rec = tt.record_a2a_message(sender="parent", receiver="child", prompt="review odds",
                                    response="ok", ok=True, gate="accept", kind="result", runs_dir=Path(tmp))
        assert rec["kind"] == "a2a_message" and rec["a2aKind"] == "result" and rec["gate"] == "accept"
        assert (Path(tmp) / "t-a2a.jsonl").exists()


# --------------------------------------------------------------------------- #
# Swarm delegate() emits delegate/result/synthesis A2A spans (integration)
# --------------------------------------------------------------------------- #
def test_delegate_logs_a2a_messages() -> None:
    from agent import harness as h
    from agent import subagent as sa

    saved_runs = h.RUNS_DIR  # restore the module global so later tests aren't affected
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)  # keep harness child traces out of the repo
        log_dir = Path(tmp) / "thinking"
        os.environ["SOPHIA_THINKING_LOG"] = str(log_dir)
        try:
            client = m.ModelClient(m.resolve_config("mock"))
            specs = [sa.SubagentSpec(goal="review the gacha odds", label="legal", max_steps=1, max_retries=0)]
            sa.delegate("audit the launch", specs, client=client, parent_id="p1")
            events = []
            for path in log_dir.glob("*.jsonl"):
                events += [json.loads(li) for li in path.read_text(encoding="utf-8").splitlines() if li.strip()]
            kinds = {e.get("a2aKind") for e in events if e.get("kind") == "a2a_message"}
            assert {"delegate", "result", "synthesis"} <= kinds
        finally:
            os.environ.pop("SOPHIA_THINKING_LOG", None)
            h.RUNS_DIR = saved_runs


def _run_openai_payload(payload: dict):
    """Drive _call_openai_compatible against a faked HTTP response (real parse path)."""
    import urllib.request

    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    orig = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=None: _R(json.dumps(payload).encode())
    try:
        cfg = m.ModelConfig(kind="openai", model="r1", base_url="http://x/v1", api_key_default="k")
        return m._call_openai_compatible("s", "u", cfg, tools=None, on_token=None)
    finally:
        urllib.request.urlopen = orig


def test_openai_capture_off_strips_think_and_drops_reasoning() -> None:
    _reset_capture(None)  # capture OFF
    payload = {"model": "r1", "choices": [{"finish_reason": "stop", "message": {
        "content": "<think>secret cot</think>Final answer.", "reasoning_content": "rc"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5}}
    res = _run_openai_payload(payload)
    # <think> is ALWAYS stripped from the answer (no CoT leak), but reasoning is NOT retained.
    assert res.text == "Final answer." and "<think>" not in res.text and "secret cot" not in res.text
    assert res.reasoning_text == "" and res.reasoning_tokens == 0


def test_openai_capture_on_retains_reasoning() -> None:
    _reset_capture("1")
    try:
        payload = {"model": "r1", "choices": [{"finish_reason": "stop", "message": {
            "content": "<think>t</think>ans", "reasoning_content": "rc"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5,
                      "completion_tokens_details": {"reasoning_tokens": 12}}}
        res = _run_openai_payload(payload)
        assert res.text == "ans" and "rc" in res.reasoning_text and "t" in res.reasoning_text
        assert res.reasoning_tokens == 12
    finally:
        _reset_capture(None)


def test_thinking_param_budget_never_below_floor() -> None:
    _reset_capture("1")
    try:
        # max_tokens just above 1024: budget must be clamped to >=1024 and stay < max_tokens.
        p = m._anthropic_thinking_param(m.ModelConfig(kind="anthropic", model="claude-3-5-sonnet-20241022", max_tokens=1101))
        assert p["type"] == "enabled" and p["budget_tokens"] == 1024 and p["budget_tokens"] < 1101
        # max_tokens too small to hold a >=1024 budget under it -> omit thinking, don't send junk.
        assert m._anthropic_thinking_param(m.ModelConfig(kind="anthropic", model="claude-3-5-sonnet-20241022", max_tokens=1024)) is None
    finally:
        _reset_capture(None)


def test_trace_id_matches_filename_without_context() -> None:
    # Regression for the per-call-UUID bug: with no context set, a record's traceId must
    # equal the file it lands in (stable memoized default), not a fresh id per call.
    _reset_capture(None)
    with tempfile.TemporaryDirectory() as tmp:
        tt._CTX.set(("", None))  # clear any context from earlier tests
        r1 = tt.record_generation("s", "u", m.ModelResult(text="a", provider="mock", model="m"), runs_dir=Path(tmp))
        r2 = tt.record_generation("s", "u2", m.ModelResult(text="b", provider="mock", model="m"), runs_dir=Path(tmp))
        files = list(Path(tmp).glob("*.jsonl"))
        assert len(files) == 1 and files[0].stem == r1["traceId"] == r2["traceId"]


# --------------------------------------------------------------------------- #
# A2A distillation: fail-closed, training rows + skill candidates
# --------------------------------------------------------------------------- #
def _a2a_events() -> list[dict]:
    return [
        # two successful, accepted "review odds" delegations to a legal agent
        {"kind": "a2a_message", "a2aKind": "delegate", "traceId": "t", "sender": "p", "receiver": "p.sub1-legal",
         "ok": True, "gate": None, "prompt": "review odds policy", "response": ""},
        {"kind": "a2a_message", "a2aKind": "result", "traceId": "t", "sender": "p.sub1-legal", "receiver": "p",
         "ok": True, "gate": "accept", "prompt": "review odds policy", "response": "Odds disclosure required."},
        {"kind": "a2a_message", "a2aKind": "delegate", "traceId": "t", "sender": "p", "receiver": "p.sub2-legal",
         "ok": True, "gate": None, "prompt": "review odds refunds", "response": ""},
        {"kind": "a2a_message", "a2aKind": "result", "traceId": "t", "sender": "p.sub2-legal", "receiver": "p",
         "ok": True, "gate": "accept", "prompt": "review odds refunds", "response": "Refund window applies."},
        # a blocked exchange must NOT become training data (fail-closed)
        {"kind": "a2a_message", "a2aKind": "result", "traceId": "t", "sender": "x", "receiver": "p",
         "ok": False, "gate": "block", "prompt": "fabricate a citation", "response": "made up"},
    ]


def test_training_rows_are_fail_closed() -> None:
    rows = ad.a2a_training_rows(_a2a_events())
    completions = {r.completion for r in rows}
    assert "Odds disclosure required." in completions and "Refund window applies." in completions
    assert "made up" not in completions  # blocked exchange excluded
    assert all(r.a2a_kind in ad._ANSWER_LEGS for r in rows)


def test_skill_candidates_need_recurring_support() -> None:
    cands = ad.skill_candidates(_a2a_events(), min_support=2)
    # 'review-odds-*' to a 'legal' agent recurs -> one candidate; one-offs are dropped.
    assert any(c.receiver_role.startswith("legal") and c.support >= 2 for c in cands)
    rec = cands[0].to_record()
    assert rec["status"] == "candidate" and rec["support"] >= 2


def test_distill_report_counts_hash_only_skips() -> None:
    events = _a2a_events() + [
        {"kind": "a2a_message", "a2aKind": "result", "traceId": "t", "ok": True, "gate": "accept",
         "promptHash": "abc", "responseHash": "def"},  # hash-only span (no verbatim)
    ]
    report = ad.distill_events(events)
    assert report.hash_only_skipped == 1 and report.total_messages == len(events)
    assert report.rows  # the verbatim ones still distilled


def main() -> int:
    test_model_result_to_log_reports_reasoning_size_not_text()
    test_thinking_param_off_by_default()
    test_thinking_param_adaptive_for_modern_claude()
    test_thinking_param_budgeted_for_legacy_claude()
    test_record_generation_hash_only_by_default()
    test_record_generation_verbatim_when_capture_on()
    test_model_client_trace_sink_fires_on_generate()
    test_maybe_record_a2a_is_noop_when_disabled()
    test_record_a2a_message_writes_span()
    test_delegate_logs_a2a_messages()
    test_openai_capture_off_strips_think_and_drops_reasoning()
    test_openai_capture_on_retains_reasoning()
    test_thinking_param_budget_never_below_floor()
    test_trace_id_matches_filename_without_context()
    test_training_rows_are_fail_closed()
    test_skill_candidates_need_recurring_support()
    test_distill_report_counts_hash_only_skips()
    print("test_thinking_trace: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
