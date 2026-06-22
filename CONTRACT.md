# Sophia Governance Contract — v1.0.0

The stable, versioned seam between **Sophia** (the governance brain / control plane)
and **aihk-os** (the control plane + 9 role pipelines that consume Sophia).

Sophia exists to be a *trustworthy, legible, low-maintenance* governance service for a
solo founder: **no team to catch errors → fail closed; scarce founder attention →
approve-by-exception; trust → everything provenance-tracked and explainable.**

- **Machine-readable schema:** [`schema/contract-1.0.0.json`](schema/contract-1.0.0.json)
- **Golden vectors (conformance):** [`schema/golden-vectors.json`](schema/golden-vectors.json) — run by `tests/test_contract_conformance.py` on every release.
- **Changelog:** [`sophia_contract/CHANGELOG.md`](sophia_contract/CHANGELOG.md)
- **Version:** `1.0.0` (the contract version, independent of the repo `VERSION`).

```python
from sophia_contract import SophiaContract
svc = SophiaContract(store_dir="var/sophia")   # durable JSONL stores; omit for in-memory
svc.describe()
```

---

## 1. Handshake

`describe() -> { version, capabilities[], schema_url, deprecations[] }`

```json
{
  "version": "1.0.0",
  "capabilities": ["describe", "record_claim", "verify_claim", "explain_verdict", "batch_verify", "health"],
  "schema_url": "schema/contract-1.0.0.json",
  "deprecations": []
}
```

aihk-os pins against `version` (semver) and may hard-fail on a MAJOR bump. `capabilities`
lists only what is actually implemented. Each `deprecations[]` entry is
`{field, since, remove_in, note}`.

## 2. Semver + deprecation (hard rule)

| Bump | Meaning | aihk-os impact |
|------|---------|----------------|
| **MAJOR** | breaking (shape/semantics change, field removal) | fails closed; must adopt deliberately |
| **MINOR** | additive only (new optional field / capability / error code value) | safe |
| **PATCH** | bugfix, no shape change | safe |

- **A field may be deprecated for one full MAJOR before removal.** It keeps working,
  is announced in `deprecations[]`, and is only removed in the *next* MAJOR.
- **Never break shape in a MINOR.** Field names here *are* the contract.
- Ship a CHANGELOG entry every release.

## 3. Required methods

### `record_claim`
Request — `{ idempotency_key, content, sources[], parents[], blp_level }`
Returns a **Claim** — `{ claim_id, content, sources[], parents[], blp_level, created_at, signature? }`

- **Idempotency (hard rule):** the same `idempotency_key` MUST return the same `claim_id`.
  `claim_id` is derived deterministically from the key. Re-recording the same key with
  the **same** content returns the stored claim; with **different** content it fails
  closed with `BAD_REQUEST` (no silent divergence).
- `blp_level ∈ {UNCLASSIFIED, CONFIDENTIAL, SECRET, TOP_SECRET}`.
- **BLP no-write-down (record time):** a derived claim must be classified at least as
  high as every parent. Otherwise → `BLP_VIOLATION` (never silently downgraded).

### `verify_claim`
Request — `{ claim_id }` (optional `clearance`, default `UNCLASSIFIED`)
Returns a **Verdict** — `{ verdict, confidence, reasons[], cited_evidence[], suggested_fix?, supersedes?, held_reason? }`

- `verdict ∈ {accepted, rejected, superseded, held}`. **Only `accepted` may be published.**
- `held_reason ∈ {no_source, stale_source, needs_human, over_budget, blp_violation}` (present iff `verdict == held`).
- **Decision pipeline (deterministic, strict order, fail-closed):**
  1. **lookup** — unknown id → `BAD_REQUEST`.
  2. **BLP no-read-up** — clearance must dominate the claim level, else `held / blp_violation`.
  3. **budget** — cap exhausted → `held / over_budget` (stop-and-report).
  4. **superseded** — a successor exists → `superseded` with `supersedes`.
  5. **human preference (feedback loop)** — a prior human ruling short-circuits review.
  6. **evidence** — no sources → `held / no_source`; a refuted/invalid source → `rejected`; all-stale → `held / stale_source`.
  7. **confidence** — deterministic from #valid sources (+0.25 first, +0.15 corroboration), provenance lineage (+0.05).
  8. **risk-tiered auto-approve** — low-risk (`UNCLASSIFIED`) **and** confidence ≥ 0.75 **and** cited → `accepted`.
  9. **else escalate** — `held / needs_human` (ambiguous or higher-risk).

## 4. Optional capabilities (advertised only when implemented)

- `explain_verdict({claim_id}) -> Verdict + { explanation }` — the verdict plus a one-line rule-path trace.
- `batch_verify({claim_ids[]}) -> { results[] }` — independent verdicts; one bad id never fails the batch.
- `health() -> { status, version, checks{} }` — liveness + self-diagnostics for unattended operation.

## 5. Error model

`{ "error": { code, message, retryable } }`, `code ∈ {BAD_REQUEST, UNAUTHENTICATED, BLP_VIOLATION, OVER_BUDGET, UNAVAILABLE, INTERNAL}`.

`retryable` is authoritative — only `UNAVAILABLE` is retryable unchanged. Errors are
returned in-band as the wire shape above; the caller switches on `code`.

> **record vs verify on BLP:** a no-write-**down** breach is rejected at `record_claim`
> as a `BLP_VIOLATION` *error* (it must never enter the store); a no-read-**up** breach
> at `verify_claim` returns a `held / blp_violation` *verdict* (the claim exists, but
> the caller may not act on it). Both fail closed.

## 6. Conformance

`schema/golden-vectors.json` fixes claims → expected verdicts across every verdict,
held_reason, and error path. `tests/test_contract_conformance.py` runs them on every
release and cross-checks that the schema enums match the implementation.

---

## Guardrails & memory (behind the seam)

- **Durable decision log** — every verdict appended to `decisions.jsonl`.
- **Idempotency everywhere** — deterministic ids; safe retries.
- **Budget caps** — `verify_budget` triggers stop-and-report (`over_budget`), not a crash.
- **Preference feedback loop** — `record_human_verdict(...)` writes to an inspectable,
  hand-editable `preferences.jsonl`; future verifies of the same claim/content return the
  human's decision, so review burden shrinks over time.
- **Provenance** — claims carry `sources[]` and `parents[]`; verdicts cite the evidence used.

## Compatibility promise

Do not rename or remove a v1 field without a MAJOR bump. Additive changes (new optional
fields, capabilities, error-code *values*) are MINOR. aihk-os may rely on: deterministic
`claim_id` from `idempotency_key`, "only `accepted` is publishable", and the closed
`held_reason` / `error.code` sets growing only additively within v1.
