# Cross-Harness Packaging

Sophia's operator-facing surfaces — its **skills**, its **verifier gate**, and
its **MCP server** — are portable artifacts. This document describes a manifest
that names those surfaces once and a DRY-RUN validator/installer that reports
where each surface *would* land in any supported harness.

## Inspiration: ECC

[`affaan-m/ECC`](https://github.com/affaan-m/ECC) ships a single manifest plus
per-harness adapters, making one set of capabilities portable across Claude
Code, Codex, Cursor, Gemini, OpenCode, Zed and Copilot. Each harness stores
skills / MCP config in a slightly different location and format; the manifest is
the single source of truth and the adapters translate it per harness.

Sophia already had per-harness directories (`.claude/`, `.agents/`, `.grok/`,
`.cursor/`) but **no manifest** tying them together. This feature adds one.

## The manifest

`packaging/operator_manifest.json` (`schema: sophia.operator.manifest.v1`) has
two parts:

- **`surfaces`** — each a portable operator surface with an `id`, a `kind`
  (`skill` | `mcp` | `gate`), a real repo-relative `source`, and a description.
  Today this covers the portable skills under `skills/portable/`, the
  `prompt-author` skill under `.claude/skills/`, the MCP config (`.mcp.json`,
  which launches `sophia_mcp/server.py`), and the verifier gate
  (`agent/gate.py`).
- **`harnesses`** — for each harness id (`claude`, `cursor`, `codex`,
  `opencode`, `grok`, `agents`) the target `skillsDir` and `mcpFile` that an
  installer would write into.

Every `source` is a **real path in the repo**. A manifest pointing at a missing
file is a bug, and `tests/test_install_sophia_layer.py` fails the build if any
source is absent.

## The verifier gate, installable in any harness

The most load-bearing surface is the **verifier gate** (`agent/gate.py`): the
epistemic self-check that screens responses for source-discipline and
attribution failures. Treating it as a manifest surface means the same gate can
be carried into Codex, Cursor, OpenCode, etc., rather than living only inside
Claude Code. Portability of the *verifier* — not just the skills — is the point:
the discipline travels with the agent.

## The installer (DRY-RUN only, today)

`tools/install_sophia_layer.py` (stdlib only, offline, deterministic):

- `validate_manifest() -> (ok, problems)` — fails on a wrong schema id, a
  missing/duplicate surface, an unknown kind, a missing source on disk, or a
  malformed harness entry.
- `plan_install(harness_id)` — returns the per-surface plan (source, computed
  target under that harness's dir, whether the source exists). **Writes
  nothing.**
- `offline_invariants() -> (ok, detail)` and a `__main__` that prints
  `PASS`/`FAIL`.

Run it:

```bash
python -m tools.install_sophia_layer            # validate + invariants
python tools/install_sophia_layer.py --harness claude   # print a dry-run plan
python tests/test_install_sophia_layer.py       # tests
```

## Honest limits (OPEN work)

- **Dry-run only.** The installer prints planned copy actions; it does **not**
  copy anything. Real install (with backup/idempotency/uninstall) is OPEN.
- **Per-harness adapter formats are OPEN.** Different harnesses expect different
  skill/MCP layouts and metadata. The manifest currently records only target
  *directories*; the format translation (the ECC "adapter" layer) is not yet
  implemented. The `.codex/` and `.opencode/` directories do not exist in the
  repo yet — their harness entries are forward-looking targets.
- **No claim of cross-harness parity.** Listing a surface in the manifest does
  not prove it behaves identically in another harness; that requires per-harness
  verification still to be built.

`canClaimAGI` stays **false**. This is packaging plumbing, not a capability
claim.
