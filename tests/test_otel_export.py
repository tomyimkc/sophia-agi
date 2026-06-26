#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the OTLP/HTTP exporter (sophia_contract.otel_export).

The contract already records Langfuse-shaped spans; this exporter re-maps them to
OpenTelemetry's OTLP/JSON shape and adds an ``agent.decision``-aligned span event
for each verdict. Deterministic, offline, no new dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_contract import SophiaContract  # noqa: E402
from sophia_contract.otel_export import (  # noqa: E402
    build_payload,
    export_spans,
    export_traces_file,
)

_CLK = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731


def _attrs(span: dict) -> dict:
    """Flatten an OTLP span's attribute list to a {key: scalar} map for asserts."""
    out = {}
    for kv in span.get("attributes", []):
        (vt, val), = kv["value"].items()
        out[kv["key"]] = val
    return out


def test_payload_shape_is_otlp() -> None:
    spans = [{"id": "trace_x", "name": "verify_claim", "startTime": "2026-01-01T00:00:00+00:00",
              "endTime": "2026-01-01T00:00:00+00:00",
              "input": {"claim_id": "clm_1", "clearance": "UNCLASSIFIED"},
              "output": {"verdict": "accepted", "confidence": 0.9}, "level": "DEFAULT", "metadata": {}}]
    payload = build_payload(spans, service_name="svc-test")
    rs = payload["resourceSpans"][0]
    assert _attrs(rs["resource"])["service.name"] == "svc-test"
    otlp_span = rs["scopeSpans"][0]["spans"][0]
    assert rs["scopeSpans"][0]["scope"]["name"] == "sophia.gate"
    assert otlp_span["name"] == "verify_claim"
    # ids are valid OTLP lengths (32 hex trace, 16 hex span)
    assert len(otlp_span["traceId"]) == 32 and len(otlp_span["spanId"]) == 16
    assert _attrs(otlp_span)["sophia.verdict"] == "accepted"


def test_accepted_is_ok_status() -> None:
    spans = [{"id": "a", "name": "verify_claim", "startTime": "", "endTime": "",
              "input": {}, "output": {"verdict": "accepted", "confidence": 1.0},
              "level": "DEFAULT", "metadata": {}}]
    otlp = build_payload(spans)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert otlp["status"]["code"] == 1  # OK


def test_held_is_error_status_and_carries_decision_event() -> None:
    spans = [{"id": "h", "name": "verify_claim", "startTime": "", "endTime": "",
              "input": {}, "output": {"verdict": "held", "held_reason": "no_source", "confidence": 0.0},
              "level": "WARNING", "metadata": {"role": "role_06_content_marketing"}}]
    otlp = build_payload(spans)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert otlp["status"]["code"] == 2  # ERROR — failed closed
    assert _attrs(otlp)["sophia.held_reason"] == "no_source"
    assert _attrs(otlp)["sophia.role"] == "role_06_content_marketing"
    # the decision event mirrors the agent.decision schema (chosen action + alternatives)
    event = otlp["events"][0]
    assert event["name"] == "gate.decision"
    ev_attrs = {kv["key"]: list(kv["value"].values())[0] for kv in event["attributes"]}
    assert ev_attrs["decision"] == "held"
    assert "accepted" in ev_attrs["alternatives.considered"]
    assert ev_attrs["reasoning"] == "held: no_source"


def test_record_claim_span_has_no_decision_event() -> None:
    spans = [{"id": "r", "name": "record_claim", "startTime": "", "endTime": "",
              "input": {"idempotency_key": "k"}, "output": {"claim_id": "clm_1"},
              "level": "DEFAULT", "metadata": {}}]
    otlp = build_payload(spans)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert "events" not in otlp  # not a verdict -> no decision event


def test_export_offline_no_endpoint() -> None:
    spans = [{"id": "t1", "name": "n", "startTime": "", "endTime": "",
              "input": {}, "output": {}, "level": "DEFAULT", "metadata": {}}]
    r = export_spans(spans, endpoint=None)
    assert r["sent"] is False and r["count"] == 1 and "payload" in r


def test_export_dry_run_with_endpoint() -> None:
    r = export_spans([], endpoint="http://localhost:4318", dry_run=True)
    assert r["sent"] is False and r["reason"] == "dry_run"


def test_contract_traces_export_end_to_end() -> None:
    # the contract's own spans must export cleanly through OTLP
    svc = SophiaContract(clock=_CLK)
    c = svc.record_claim({"idempotency_key": "e", "content": "x", "sources": ["s"]})
    svc.verify_claim({"claim_id": c["claim_id"]})
    payload = build_payload(svc.tracer.events())
    otlp_spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(otlp_spans) >= 2
    # exactly the verify_claim span(s) carry a gate.decision event
    decisions = [s for s in otlp_spans if "events" in s]
    assert decisions and all(s["events"][0]["name"] == "gate.decision" for s in decisions)


def test_export_traces_file_roundtrip(tmp_path) -> None:
    import json

    p = tmp_path / "traces.jsonl"
    p.write_text(json.dumps({"id": "z", "name": "verify_claim", "startTime": "", "endTime": "",
                             "input": {}, "output": {"verdict": "accepted"}, "level": "DEFAULT",
                             "metadata": {}}) + "\n", encoding="utf-8")
    r = export_traces_file(str(p), endpoint=None)
    assert r["sent"] is False and r["count"] == 1


def main() -> int:
    import inspect
    import tempfile

    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            if "tmp_path" in inspect.signature(fn).parameters:
                fn(Path(tempfile.mkdtemp()))
            else:
                fn()
    print("test_otel_export: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
