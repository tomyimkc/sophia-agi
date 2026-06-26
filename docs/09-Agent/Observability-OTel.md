# Gate verdicts as OpenTelemetry spans

> Export every Sophia verdict to any OTLP collector (Jaeger, Tempo, Datadog,
> Langfuse-OTel) so the gate's ruling appears as one more step in the agent trace —
> no vendor lock-in, no new dependency.

`sophia_contract.trace.Tracer` already records each service call as a span. The
[`langfuse_export`](../../sophia_contract/langfuse_export.py) adapter ships those
spans to Langfuse; [`otel_export`](../../sophia_contract/otel_export.py) ships the
**same** spans to an OpenTelemetry collector in OTLP/HTTP JSON shape.

## Why OTel

A gate verdict *is* a decision with alternatives considered — which is exactly the
`agent.decision` event that OpenTelemetry agent-instrumentation libraries (e.g.
[`chrisipanaque/opentelemetry-ai-agent-observability`](https://github.com/chrisipanaque/opentelemetry-ai-agent-observability))
emit for an agent's own reasoning steps. So a `verify_claim` span carries a
`gate.decision` span event whose attributes line up with that schema:

| `agent.decision` (agent loop) | `gate.decision` (Sophia verdict) |
|---|---|
| `decision` (chosen action) | the verdict (`accepted` / `held` / …) |
| `alternatives` (considered) | `alternatives.considered` (the other verdicts) |
| `reasoning` | `reasoning` (e.g. `held: no_source`) |

Point an agent's instrumentation and Sophia's exporter at the same collector and
the gate's ruling slots into the trace right after the step that produced the
claim. The verdict also sets OTel span **status**: a clean `accepted` is `OK`;
every fail-closed outcome (`held` / `rejected` / `superseded`) is `ERROR`, so it
stands out in any trace UI.

## Usage

Offline / CI (no collector needed — inspect the wire payload):

```python
from sophia_contract import SophiaContract
from sophia_contract.otel_export import build_payload, export_spans

svc = SophiaContract()
c = svc.record_claim({"idempotency_key": "k1", "content": "...", "sources": ["s1"]})
svc.verify_claim({"claim_id": c["claim_id"]})

payload = build_payload(svc.tracer.events())     # OTLP/HTTP JSON, ready to POST
export_spans(svc.tracer.events(), dry_run=True)  # {sent: False, payload: {...}}
```

Live — set the OTel-standard env vars and the exporter POSTs to `{endpoint}/v1/traces`:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_HEADERS="x-api-key=…"   # optional
export OTEL_SERVICE_NAME=sophia-contract           # optional (default)
```

```python
export_spans(svc.tracer.events())            # POST in-memory spans
export_traces_file("traces.jsonl")           # or replay a persisted trace file
```

With no `OTEL_EXPORTER_OTLP_ENDPOINT` set (or `dry_run=True`) the exporter never
touches the network — it returns the payload it *would* have sent. That keeps CI
offline and reproducible, matching the Langfuse adapter's behaviour.

## See also

- [`sophia_contract/otel_export.py`](../../sophia_contract/otel_export.py) — the exporter (urllib-only, dependency-free).
- [`tests/test_otel_export.py`](../../tests/test_otel_export.py) — wire-shape + status-mapping tests.
- [Defense-In-Depth-Governance.md](../11-Platform/Defense-In-Depth-Governance.md) — pair the claim gate with a repo-boundary gate; both decisions land in one collector.
