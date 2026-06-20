# Design: Source-Discipline Finalize Gate (Sophia → OpenClaw)

**Status:** approved (design) — 2026-06-20
**Branch:** `feat/source-discipline-gate`

## Context & goal

OpenClaw is a multi-channel AI gateway whose agents can assert hallucinated
attributions or lineage merges (e.g. *"Confucius wrote the Dao De Jing"*). Sophia's
whole thesis is **source discipline** — a machine-checked "never merge lineages" rule
(`agent/verifiers.py:provenance_faithful` / `source_discipline`, line 147 / alias 221).

**Goal:** make Sophia's source-discipline rule *govern* OpenClaw replies — every OpenClaw
agent reply is checked against Sophia's verifier before it is sent, and a draft that
asserts a forbidden attribution is sent back for one correction pass. Detection is a
pure local-regex check (~2 ms, **no model call**), so it runs on every turn regardless
of which model produced the draft (works on this install's xAI/grok and ACP-Claude paths;
the broken Anthropic key is irrelevant).

This is the literal "Sophia helps OpenClaw" direction, and it revives the
verify-before-send capability that was disabled for latency — but as a near-free,
high-precision check instead of a blind same-model reflexion.

## Scope

**In scope (v1):** output gating only — check the agent's *draft reply* on
`before_agent_finalize`.

**Out of scope (deliberately, may follow):**
- Input-boundary gating of retrieved/untrusted text (`tool_result_persist`) — a later
  add that reuses the same checker; noted as a follow-up, not built here.
- A warm long-lived sidecar — rejected for v1 (see Decisions).
- Exposing Sophia as a full OpenClaw model/agent provider — out of scope.
- Any change to Sophia's knowledge-write path or the provenance gate itself.

## Settled decisions

| Decision | Choice | Rationale |
|---|---|---|
| Plugin→checker transport | **Spawn-per-call** | No infra to keep alive; pays ~109 ms cold start + spawn per gated reply. Simpler than a warm sidecar. |
| Fail mode (checker unreachable/errors) | **Fail-open + loud alert** | Availability over a missed narrow (~31-record) check; a downed checker must never stall replies. Every bypass is logged loudly. |
| On a violation | **`revise`** (native `before_agent_finalize` contract) | Asks the producing model for one corrected pass. Needs a working model — fine on xAI/grok + ACP-Claude. |
| Corpus | Sophia's existing `doNotAttributeTo` records (~31, from `data/*.json`) | Reuses the curated, tested corpus; high precision, honestly narrow. |

## Architecture

Two small artifacts in two homes, connected by a stdin→stdout subprocess call (the same
inter-process pattern Sophia already uses in reverse at `sophia_mcp/tools_impl.py`).

### Component A — Sophia CLI: `tools/source_discipline_cli.py` (sophia-agi repo)

A dependency-free, offline CLI that exposes the verifier. Follows Sophia conventions:
`sys.path.insert(0, ROOT)`, a `check(text) -> dict` core (dict result, like the repo's
other tool CLIs), and `main() -> int`. (Note: not the no-arg `run_validation()` pattern —
this CLI takes input text.)

- **Input:** draft text on **stdin** (or `--text "..."`).
- **Core:** `from agent.verifiers import provenance_faithful` →
  `provenance_faithful()(text, task=None, step={})` → its `{passed, reasons, detail}`.
- **Output:** one line of JSON to stdout:
  `{"passed": bool, "reasons": [...], "violations": [...]}` (violations lifted from
  `detail`). Exit `0` always on a successful check (pass or fail is in the JSON);
  non-zero only on its own internal error (so the plugin can distinguish "checked,
  violated" from "couldn't check").
- **Dependency-free:** stdlib only (`sys`, `json`, `argparse`); reuses `agent/verifiers.py`.
  No FastAPI/uvicorn (Sophia core stays dependency-light). `okf/` untouched.
- **Reusable:** the same core function backs a future `tool_result_persist` input gate.

### Component B — OpenClaw plugin: `sophia-source-discipline` (`~/.openclaw/plugins/`)

A plain-JS `definePluginEntry` plugin (same shape as the `selfimprove-capture` /
`verify-before-send` plugins already on this box).

- **Hook:** `before_agent_finalize` (requires `plugins.entries.<id>.hooks.allowConversationAccess=true`).
- **Gate:** skip drafts shorter than `minChars` (default ~80 — greetings/acks rarely carry
  attributions) to avoid spawning on trivial turns.
- **Check:** `child_process.spawn(pythonBin, [cliPath], { cwd: sophiaRepoPath })`, write the
  draft (`event.lastAssistantMessage`) to stdin, read stdout JSON, with a hard timeout.
- **Decide:**
  - `passed === false` → return
    `{action:"revise", reason:"source-discipline: "+violations, retry:{instruction:"Remove or correct the forbidden attribution; if unsure, state who did NOT author it.", idempotencyKey: hash(draft), maxAttempts: 2}}`.
  - `passed === true` → return `undefined` (finalize as-is).
  - **any error** (spawn fail / timeout / non-zero exit / bad JSON) → **fail-open**: return
    `undefined` AND log a loud `SOURCE-DISCIPLINE GATE BYPASSED: <reason>` warning (and,
    when a route exists, an owner alert).
- **Config** (`plugins.entries.sophia-source-discipline.config`): `pythonBin`
  (default `python3`), `sophiaRepoPath`, `cliPath` (default `tools/source_discipline_cli.py`),
  `minChars` (default 80), `timeoutMs` (default 4000).

### Data flow

```
Telegram turn → OpenClaw agent (xai/grok) drafts a reply
  → before_agent_finalize fires (plugin)
    → draft length ≥ minChars?  no → finalize as-is
    → yes → spawn python3 tools/source_discipline_cli.py  (draft on stdin)
              → provenance_faithful(draft) → {passed, violations} on stdout
    → passed       → finalize as-is
    → not passed   → {action:"revise", retry:{...}}  (model corrects, ≤2 passes)
    → spawn error  → finalize as-is + loud "GATE BYPASSED" log/alert  (fail-open)
```

## Error handling

Every failure mode resolves to **fail-open** so a reply is never blocked by gate
infrastructure: spawn `ENOENT` (wrong python/path), timeout (>`timeoutMs`), CLI non-zero
exit, or unparseable stdout all → `undefined` + a loud log. The CLI itself never raises on
expected input (mirrors Sophia's tool/impl convention of returning structured results).

## Testing (all offline)

- **Sophia — `tests/test_source_discipline_cli.py`** (plain functions + `main()->int`,
  `sys.path.insert(ROOT)`, no model, no network):
  - a forbidden assertion (e.g. *"Confucius wrote the Dao De Jing"*) → `passed=false`,
    non-empty `violations`;
  - the negation/contrast (*"Confucius did **not** write the Dao De Jing"*) → `passed=true`
    (proves the carve-out survives the CLI boundary);
  - benign text → `passed=true`;
  - the `run_validation()` core returns the documented dict shape.
  Wire into `.github/workflows/ci.yml`.
- **OpenClaw plugin:** load cleanly (`openclaw plugins doctor`); a `before_agent_finalize`
  probe confirms it fires on a substantive draft and skips a short one (same validation
  method used for the earlier critic); a stubbed-CLI path asserts fail-open on spawn error.
  (Plugin tests are operational, not CI — the plugin lives in `~/.openclaw`, not the repo.)

## Shipping

- **Sophia CLI + test + CI + docs** → committed on `feat/source-discipline-gate`, opened as
  its own PR (independent of the OpenClaw-provider PR #9). VERSION bump + CHANGELOG +
  `docs/11-Platform/` reference.
- **OpenClaw plugin** → `~/.openclaw/plugins/sophia-source-discipline/`, installed via
  `openclaw plugins install -l`, enabled with `allowConversationAccess=true`, restart, verify.

## Honest limits & risks

- **Narrow corpus:** ~31 curated `doNotAttributeTo` records. High precision, but NOT a
  general hallucination catcher — false-negatives on unlisted lineages are expected.
- **Spawn latency:** ~109 ms cold start + process spawn per gated reply (the cost of the
  spawn-per-call choice). Mitigated by `minChars` gating; revisit a warm sidecar only if it
  becomes a felt cost.
- **`revise` needs a working model.** Detection always runs; the *correction* pass uses the
  same session model — fine on xAI/grok and ACP-Claude, dead on anthropic-API turns (which
  already fail anyway).
- **Fail-open by design:** a downed checker means replies go out **unchecked** (loudly
  logged). Accepted trade-off for availability on a narrow gate.
- **Provenance unchanged:** this only inspects OpenClaw drafts; it writes no knowledge and
  does not touch Sophia's gate. Adds nothing to, and claims nothing about, the AGI-proof
  package.
