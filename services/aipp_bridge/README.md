# Sophia ↔ AIpp Bridge

An **authenticated** HTTP surface that lets the [AIpp](https://github.com/tomyimkc/aipp)
iOS boss-cockpit consume Sophia's epistemic gate, RAG grounding, and conscience
kernel.

The raw RAG API (`services/rag_api`) is intentionally unauthenticated and exposes
Sophia's internal gate dictionaries verbatim. This bridge sits in front of that
machinery and:

1. **Adds Bearer-token auth** (`SOPHIA_AIPP_TOKEN`) and **fails closed** — with no
   token configured, every authed endpoint returns `503`.
2. **Normalizes** Sophia's rich gate / conscience dicts into the compact
   **AIpp verdict contract** the app maps directly onto its governance states.

## Verdict contract

Every authed endpoint returns:

```jsonc
{
  "verdict": "accepted | held | rejected | abstained",
  "confidence": 0.0,            // 0–1, derived from gate/conscience severity
  "reasons": ["..."],           // violations + warnings + conscience reason
  "abstained": false,           // true on the honest "I don't know" path
  "sources": [{"path": "...", "title": "...", "score": 0.0}],
  "answer": "...",              // /ask only
  "gatePassed": true,           // /ask, /verify
  "conscienceVerdict": "allow", // /conscience only (raw kernel verdict)
  "zhSummary": "present|null"   // whether a 中文 discipline section is present
}
```

| AIpp verdict | Meaning | Maps to AIpp governance |
|---|---|---|
| `accepted` | Gate passed, safe to act on | `verified` |
| `held` | Soft fail (warnings / revise / retrieve / escalate) | `held` |
| `rejected` | Hard provenance/attribution/legal/numeric violation, or conscience `block` | `rejected` |
| `abstained` | Sophia declined rather than fabricate | `abstained` |

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | no | Liveness + whether auth is configured |
| POST | `/ask` | yes | Grounded, gated answer with sources (Research/Knowledge agents) |
| POST | `/verify` | yes | Run the epistemic gate over an existing draft → verdict |
| POST | `/conscience` | yes | Run the conscience kernel over a draft → verdict |

## Run locally

```bash
export SOPHIA_AIPP_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
uvicorn services.aipp_bridge.main:app --port 8081
```

Then in AIpp → Provider Settings → Sophia, set the endpoint (e.g. your Mac's
Tailscale host) and paste the same token.

## Deploy

```bash
docker build -f services/aipp_bridge/Dockerfile -t sophia-aipp-bridge .
docker run -p 8081:8081 -e SOPHIA_AIPP_TOKEN=... sophia-aipp-bridge
```

Front it with TLS (Tailscale Serve, Caddy, or Cloud Run) before exposing it to
the device — the bearer token is the only credential.

## Tests

```bash
python3 -m pytest services/aipp_bridge/test_verdict.py -q
```

The verdict normalization is pure and dependency-free, so these run without the
RAG index or any model.
