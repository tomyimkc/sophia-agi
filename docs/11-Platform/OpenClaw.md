# OpenClaw Integration

*Added in v0.7.1.* How Sophia composes with [OpenClaw](https://github.com/openclaw/openclaw),
and — just as importantly — what it deliberately does **not** do.

## What OpenClaw is

OpenClaw (npm `openclaw`, MIT) is a self-hosted **multi-channel AI gateway**. It runs as a
local loopback daemon with stored per-provider auth profiles (OAuth or API key) and exposes a
unified inference surface across many backends (Anthropic, OpenAI-compatible incl. GLM / vLLM /
SGLang / Ollama / llama.cpp / DeepSeek, xAI/grok, …), handling auth, fallback, and cost *inside*
the gateway. It is a **Node CLI, not a Python library**, so Sophia shells out to it — never
imports it. The integration-relevant surface is one command:

```
openclaw infer model run --model <provider>/<model> --prompt <text> --json
# -> {"ok": true, "provider": "...", "model": "...", "outputs": [{"text": "..."}]}
```

OpenClaw also offers image/embedding inference, a side-effecting `agent` turn, multi-channel
`message send`, an MCP layer, and a plugin SDK. Only the read-only text-inference path is wired
here (see *Non-goals*).

## Why this shape: a model backend behind `agent/model.py`

Sophia's `agent/model.py` is already a **unified model adapter** over four transport "kinds"
(`anthropic`, `openai`, `grok`, `mock`). OpenClaw is itself a unified inference gateway, so it
maps 1:1 onto that abstraction — it drops into the existing `_TRANSPORTS` dispatch with **no
architectural change**, mirroring the existing `grok` CLI/subprocess transport almost exactly
(`_call_grok`).

This is the lowest-surface, highest-leverage seam. Because every higher layer (harness,
librarian ingest, eval) consumes models only through `resolve_config()` / `default_client()`,
one preset transparently lights up the other suggested use-cases **without opening a second
code path**:

- **Ingestion (for free):** `python tools/wiki_ingest.py raw/<id>.txt --provider openclaw`
  drives `agent/wiki_librarian.ingest_text()` unchanged. Every resulting write still terminates
  at `wiki_store.upsert → gate() → provenance_faithful`, so **no new knowledge-write path is
  created** and the provenance gate stays the sole chokepoint.
- **MCP (thin, opt-in):** a read-only audited tool `sophia_openclaw_infer` (`risk="low"`) exposes
  the same inference to MCP clients. It writes no knowledge and is not a provenance path.

## Seams touched

| File | Change |
|------|--------|
| `agent/model.py` | `openclaw` preset (default route `xai/grok-4.3`) + `_call_openclaw` transport + `_TRANSPORTS` registration. Stdlib-only (`subprocess`, `json`, `os`). |
| `sophia_mcp/tools_impl.py`, `sophia_mcp/server.py` | read-only `sophia_openclaw_infer` MCP tool, `@audited(risk="low")`. |
| `tests/test_model_openclaw.py`, `tests/test_mcp_openclaw.py` | offline tests (stub `subprocess.run`); wired into CI alongside the previously-unwired `test_model_adapter.py`. |
| `.env.example` | `openclaw` preset + `SOPHIA_OPENCLAW_BIN`. |

`okf/`, the provenance gate, `@audited`'s write semantics, and `_auto_provider()` are **untouched**.

## Usage

```bash
# explicit provider (opt-in only — never auto-selected)
SOPHIA_MODEL_PROVIDER=openclaw python tools/agent_harness.py run "hello"

# route a specific provider/model through OpenClaw's gateway
python tools/agent_harness.py run "hello" --provider openclaw:anthropic/claude-sonnet-4-6

# stay offline-safe with a fallback
SOPHIA_MODEL_PROVIDER=openclaw SOPHIA_MODEL_FALLBACKS=mock python tools/agent_harness.py run "hi"
```

`SOPHIA_OPENCLAW_BIN` overrides the binary (default `openclaw` on `PATH`). An alternative,
**zero-code** wiring is the OpenAI-compatible preset pointed at the loopback gateway
(`kind=openai`, `base_url=http://localhost:18789/v1`) — documented as a config-only fallback, not
the primary path.

## Non-goals (honest limits)

- **Strictly opt-in.** `_auto_provider()` never selects OpenClaw; the offline/CI default remains
  `mock`. When the binary is absent, the transport degrades to `ok=False` (never crashes) so the
  `mock` fallback keeps the stack offline-testable.
- **No side-effecting wiring.** OpenClaw's `agent` (tool-running turns) and `message send`
  (multi-channel delivery) are *not* integrated — they sit outside Sophia's evidence/provenance
  discipline and would expand the autonomy surface. Wiring them later would require
  `@audited(risk="high")` + `SOPHIA_MCP_APPROVE_WRITES` and a separate, explicitly-scoped proposal.
- **No provenance bypass.** OpenClaw text only enters the knowledge base through the existing
  librarian → `wiki_store.upsert` → source-discipline gate, which independently rejects lineage
  merges **even when writes are approved**. (Caveat, pre-existing: the gate's body check catches
  forbidden attributions for traditions with a registered `doNotAttributeTo` record and known
  phrasings; the frontmatter self-merge check is absolute. If OpenClaw *fetches* a remote source,
  persist its bytes under `raw/<id>` before citing — `build_page` does not verify the citation
  physically exists.)
- **Plumbing only.** No real token streaming (CLI is request/response, like `grok`); native
  tool-calling is not passed through; cost is Sophia's estimate, not OpenClaw's; auth lives in
  OpenClaw's profiles (opaque to Sophia — failures surface only as `ok=False`). The CLI argv /
  JSON-envelope shape is pinned from a working install, not formal API docs; a future OpenClaw
  release could change it (guards degrade to `ok=False` but do not auto-detect a schema change).
- **This integration adds nothing to, and makes no claim about, the AGI-candidate proof package.**
