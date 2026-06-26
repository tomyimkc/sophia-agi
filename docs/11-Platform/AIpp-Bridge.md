# Milestone — Sophia ↔ AIpp Bridge (Authenticated Gate Surface)

**Date:** 2026-06-24
**Branch:** `claude/sophia-agi-aipp-integration-57x7ip`
**Status:** ✅ Implemented + unit-tested

## Goal

Give the AIpp iOS boss-cockpit a safe, authenticated way to consume Sophia's
core capabilities — the epistemic gate, RAG grounding, and the conscience
kernel — so that agent output flowing through AIpp can be **verified, grounded,
or honestly abstained** before the boss ever sees it.

## What shipped

A new service at `services/aipp_bridge/`:

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: `/health`, `/ask`, `/verify`, `/conscience` with Bearer auth |
| `verdict.py` | Pure normalization: Sophia gate/conscience dicts → AIpp verdict contract |
| `test_verdict.py` | 9 dependency-free unit tests (all passing) |
| `Dockerfile` | Container targeting port 8081 |
| `.env.example` | `SOPHIA_AIPP_TOKEN` and inherited RAG config |
| `README.md` | Contract, endpoints, run/deploy instructions |

## Design decisions

- **Fail closed on auth.** The bridge returns `503` if `SOPHIA_AIPP_TOKEN` is
  unset and `401` on a bad token (constant-time compare). This mirrors Sophia's
  fail-closed philosophy — no token, no answers.
- **Separate from `rag_api`.** The existing RAG API stays unauthenticated and
  shape-rich for internal use; the bridge is the only surface meant to face a
  device, and it normalizes output rather than leaking internal gate dicts.
- **Four-state verdict contract.** Sophia's varied internals (gate `passed` +
  `violations` + `warnings`, conscience `allow|revise|retrieve|clarify|escalate|abstain|block`)
  collapse into `accepted | held | rejected | abstained`, the exact set AIpp's
  governance model needs. Abstention is first-class: an honest "I don't know"
  is detected from the answer text and never reported as a failure.
- **Conservative combination.** When both the gate and the conscience kernel
  weigh in, `combine()` always keeps the most conservative verdict.

## How AIpp uses it

- **Verify any draft** (`/verify`) — every agent proposal (Claude, OpenClaw,
  Apple on-device, mock) is gated before entering the Decision Queue.
- **Grounded answers** (`/ask`) — Research/Knowledge agents get cited,
  abstention-capable answers instead of free-form generation.
- **Conscience gate** (`/conscience`) — upgrades AIpp's regex "irreversible
  four" check into a real deliberative decision.

## Follow-ups

- Optional per-claim routing (`route_claims=True`) for finer-grained provenance.
- Streaming `/ask` for long answers.
- Rate limiting / per-device tokens if the bridge ever leaves a private tailnet.
