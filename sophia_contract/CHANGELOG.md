# Sophia Governance Contract — Changelog

The contract is versioned independently of the repository (`VERSION`). It follows
semver: **MAJOR** = breaking, **MINOR** = additive only, **PATCH** = bugfix. A field
may be deprecated for one full MAJOR before removal. aihk-os pins against the version
returned by `describe()`.

## [1.0.0] — 2026-06-22

Initial published contract.

### Added
- **Handshake** — `describe() -> {version, capabilities[], schema_url, deprecations[]}`.
- **Required methods**
  - `record_claim({idempotency_key, content, sources[], parents[], blp_level}) -> Claim`
    — deterministic `claim_id` from `idempotency_key` (idempotent; fail-closed on a
    key reused with different content); BLP no-write-down enforced at record time.
  - `verify_claim({claim_id}) -> Verdict{verdict, confidence, reasons[], cited_evidence[],
    suggested_fix?, supersedes?, held_reason?}` — deterministic, fail-closed pipeline;
    only `accepted` is publishable.
- **Optional capabilities** — `explain_verdict`, `batch_verify`, `health` (all implemented).
- **Error model** — `{error:{code, message, retryable}}` over the closed code set
  `{BAD_REQUEST, UNAUTHENTICATED, BLP_VIOLATION, OVER_BUDGET, UNAVAILABLE, INTERNAL}`.
- **Risk-tiered auto-approve** — low-risk + high-confidence + cited → `accepted`;
  ambiguous / higher-risk → `held(needs_human)`.
- **Preference feedback loop** — `record_human_verdict(...)` to an inspectable, editable
  store; future verifies short-circuit to the human ruling.
- **Guardrails & memory** — durable decision log, budget caps (stop-and-report),
  supersession registry, idempotent claim store.
- **JSON Schema** — `schema/contract-1.0.0.json`; **golden vectors** —
  `schema/golden-vectors.json` (15 vectors run in CI).

### Deprecated
- None.

### Notes
- BLP split: no-write-**down** is a `BLP_VIOLATION` error at `record_claim`; no-read-**up**
  is a `held(blp_violation)` verdict at `verify_claim`. Both fail closed.
