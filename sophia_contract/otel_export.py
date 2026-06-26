# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Export the contract's gate verdicts as OpenTelemetry (OTLP/HTTP) spans.

``sophia_contract.trace.Tracer`` already records each service call as a
Langfuse-shaped span (``id, name, startTime, endTime, input, output, level,
metadata``) in ``traces.jsonl``. This adapter re-maps those same spans into the
OTLP/HTTP JSON shape (``resourceSpans -> scopeSpans -> spans``) and POSTs them to
any OpenTelemetry collector (``POST {endpoint}/v1/traces``), so a Sophia verdict
shows up in Jaeger / Tempo / Datadog / Langfuse-OTel next to the rest of an
agent's trace — no vendor lock-in.

The cross-walk is deliberate: a Sophia verdict *is* a decision with alternatives,
so each ``verify_claim`` span carries a ``gate.decision`` span event whose
attributes line up with the ``agent.decision`` event emitted by OpenTelemetry
agent-instrumentation libraries (``decision`` = chosen action, ``reasoning``,
``alternatives.considered``). Drop both into the same collector and the gate's
ruling reads as one more step in the agent loop.

Dependency-free (urllib, like ``langfuse_export.py`` / ``agent/model.py``) and
offline-testable: with no endpoint or ``dry_run=True`` it builds the OTLP payload
and returns it WITHOUT any network call, so CI can assert the wire shape. Config
comes from the environment (OTel-standard names):
``OTEL_EXPORTER_OTLP_ENDPOINT`` (e.g. http://localhost:4318),
``OTEL_EXPORTER_OTLP_HEADERS`` (``k1=v1,k2=v2``), ``OTEL_SERVICE_NAME``
(default ``sophia-contract``).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import datetime

# OTel status codes: 0 UNSET, 1 OK, 2 ERROR. A clean accept is OK; any
# fail-closed outcome (rejected / superseded / held) is surfaced as ERROR so it
# stands out in a trace UI exactly as the Langfuse "WARNING" level does today.
_STATUS_OK = 1
_STATUS_ERROR = 2
_SPAN_KIND_INTERNAL = 1


def _hex(seed: str, length: int) -> str:
    """Deterministic hex id of ``length`` chars from ``seed`` (trace/span ids)."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:length]


def _unix_nano(ts: str) -> int:
    """ISO-8601 timestamp -> Unix nanoseconds; 0 when absent/unparseable.

    Tracer timestamps are whatever ``clock()`` returns (often an ISO string, or
    "" in tests). We never raise — an unparseable stamp just yields 0 so the
    span still exports."""
    if not ts:
        return 0
    try:
        return int(datetime.fromisoformat(ts).timestamp() * 1_000_000_000)
    except (ValueError, TypeError):
        return 0


def _attr(key: str, value) -> dict:
    """One OTLP KeyValue, typed by the Python value (str/bool/int/float)."""
    if isinstance(value, bool):
        any_value = {"boolValue": value}
    elif isinstance(value, int):
        any_value = {"intValue": str(value)}
    elif isinstance(value, float):
        any_value = {"doubleValue": value}
    else:
        any_value = {"stringValue": "" if value is None else str(value)}
    return {"key": key, "value": any_value}


def _decision_event(span: dict, ts_nano: int) -> "dict | None":
    """Map a ``verify_claim`` span to an ``agent.decision``-shaped span event.

    Returns None for spans that are not verdicts (e.g. ``record_claim``), so the
    event only appears where there is an actual ruling to report."""
    output = span.get("output") or {}
    verdict = output.get("verdict")
    if not verdict:
        return None
    alternatives = [v for v in ("accepted", "rejected", "superseded", "held") if v != verdict]
    attributes = [
        _attr("decision", verdict),
        _attr("alternatives.considered", ",".join(alternatives)),
    ]
    if output.get("held_reason"):
        attributes.append(_attr("reasoning", f"held: {output['held_reason']}"))
    if output.get("confidence") is not None:
        attributes.append(_attr("decision.confidence", float(output["confidence"])))
    return {"timeUnixNano": str(ts_nano), "name": "gate.decision", "attributes": attributes}


def _span(span: dict) -> dict:
    """One OTLP span for a contract trace event (Langfuse-shaped -> OTLP)."""
    ts_nano = _unix_nano(span.get("startTime") or "")
    end_nano = _unix_nano(span.get("endTime") or "") or ts_nano
    output = span.get("output") or {}
    inp = span.get("input") or {}
    meta = span.get("metadata") or {}

    attributes = [_attr("sophia.span.name", span.get("name", "sophia-contract"))]
    if "verdict" in output:
        attributes.append(_attr("sophia.verdict", output["verdict"]))
    if output.get("confidence") is not None:
        attributes.append(_attr("sophia.confidence", float(output["confidence"])))
    if output.get("held_reason"):
        attributes.append(_attr("sophia.held_reason", output["held_reason"]))
    if inp.get("claim_id"):
        attributes.append(_attr("sophia.claim_id", inp["claim_id"]))
    if inp.get("clearance"):
        attributes.append(_attr("sophia.clearance", inp["clearance"]))
    if meta.get("role"):
        attributes.append(_attr("sophia.role", meta["role"]))

    # A verdict that is anything other than 'accepted' failed closed -> ERROR.
    is_clean_accept = output.get("verdict") in (None, "accepted") and span.get("level") not in ("WARNING", "ERROR")
    out_span = {
        "traceId": _hex(span["id"], 32),
        "spanId": _hex(span["id"] + ":span", 16),
        "name": span.get("name", "sophia-contract"),
        "kind": _SPAN_KIND_INTERNAL,
        "startTimeUnixNano": str(ts_nano),
        "endTimeUnixNano": str(end_nano),
        "attributes": attributes,
        "status": {"code": _STATUS_OK if is_clean_accept else _STATUS_ERROR},
    }
    event = _decision_event(span, ts_nano)
    if event is not None:
        out_span["events"] = [event]
    return out_span


def build_payload(spans: "list[dict]", *, service_name: "str | None" = None) -> dict:
    """The exact OTLP/HTTP JSON body an OpenTelemetry collector's /v1/traces wants."""
    service_name = service_name or os.environ.get("OTEL_SERVICE_NAME") or "sophia-contract"
    return {
        "resourceSpans": [{
            "resource": {"attributes": [_attr("service.name", service_name)]},
            "scopeSpans": [{
                "scope": {"name": "sophia.gate", "version": "1"},
                "spans": [_span(s) for s in spans],
            }],
        }],
    }


def _parse_headers(raw: "str | None") -> dict:
    """OTEL_EXPORTER_OTLP_HEADERS (``k1=v1,k2=v2``) -> a header dict."""
    headers: dict = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()
    return headers


def export_spans(
    spans: "list[dict]",
    *,
    endpoint: "str | None" = None,
    headers: "dict | None" = None,
    service_name: "str | None" = None,
    dry_run: bool = False,
    timeout_sec: int = 30,
) -> dict:
    """POST spans to an OTLP collector. Returns {sent, count, payload?, status?/error?}.

    No-ops safely (sent=False) when no endpoint is configured or ``dry_run`` —
    returning the payload it *would* have sent, so callers/tests can inspect it
    offline (mirrors ``langfuse_export.export_spans``)."""
    endpoint = (endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").rstrip("/")
    payload = build_payload(spans, service_name=service_name)
    count = len(payload["resourceSpans"][0]["scopeSpans"][0]["spans"])

    if dry_run or not endpoint:
        return {"sent": False, "count": count,
                "reason": "dry_run" if dry_run else "no OTEL_EXPORTER_OTLP_ENDPOINT",
                "payload": payload, "endpoint": endpoint or None}

    merged = {"Content-Type": "application/json"}
    merged.update(_parse_headers(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")))
    merged.update(headers or {})
    request = urllib.request.Request(
        f"{endpoint}/v1/traces",
        data=json.dumps(payload).encode("utf-8"),
        headers=merged,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as resp:
            return {"sent": True, "count": count, "status": resp.status}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300] if hasattr(exc, "read") else ""
        return {"sent": False, "count": count, "error": f"HTTP {exc.code}: {body}"}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"sent": False, "count": count, "error": repr(exc)}


def export_traces_file(path: "str", **kw) -> dict:
    """Read a traces.jsonl (one span per line) and export it over OTLP."""
    from pathlib import Path

    spans = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                spans.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return export_spans(spans, **kw)
