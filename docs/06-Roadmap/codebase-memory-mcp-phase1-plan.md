# codebase-memory-mcp — Phase 1 plan (pin, verify, wire)

> Phase 0 (git-crypt security controls) is on `main` (PR #330). Phase 1 makes the MCP
> **invokable safely**. `canClaimAGI:false`. This doc is plaintext on purpose — it is NOT
> secret, and it must stay readable when the tree is locked (unlike `docs/superpowers/**`).
>
> **Indexing stays DISABLED until every step below is done AND the tree is locked.** Dev
> boxes are unlocked by default, so the guard (`index_guard.py`) refuses there regardless.

## Why two independent gates
The binary is third-party (`DeusData/codebase-memory-mcp`) and git-crypt-BLIND: it reads
code snippets live off disk, so on an unlocked tree it would index/echo decrypted secrets.
Phase 1 therefore enforces **two** things at launch, both already implemented:
- **Byte-pin** — `tools/cbm/fetch_cbm.py --verify <binary>` refuses unless `sha256(binary)`
  matches `cbm.pin.json` (supply-chain: run the exact audited bytes, nothing tampered).
- **Locked-tree preflight** — `tools/cbm/index_guard.py` refuses to `exec` the binary unless
  the working tree is git-crypt LOCKED and no `GITCRYPT_KEY_B64` is present.

## Install steps (human, once — each is a real gate, not a formality)
1. **Fetch + build + audit the binary.** Clone the `repo` recorded in `cbm.pin.json`, checkout
   an audited commit, build the `binary_rel` path, and inspect it (this is the trust decision).
   Put the built binary OUT of the repo, e.g. `~/.cache/cbm/codebase-memory-mcp`.
2. **Pin it.** `python tools/cbm/fetch_cbm.py --init ~/.cache/cbm/codebase-memory-mcp --ref <audited-commit>`
   — records the binary `sha256` (and the `--ref` you pass; the ref is NOT auto-inferred) into
   `cbm.pin.json` (commit that pin update). Until this runs, `--verify` refuses and indexing is off
   (test `test_committed_pin_is_uninitialized` enforces the committed pin ships empty).
3. **Set the cache OUT of the repo.** `export CBM_CACHE_DIR=~/.cache/cbm` (never inside the
   worktree). The L2 sink (`.gitignore` + `check_no_index_artifacts.py`) is belt-and-suspenders;
   the primary rule is the DB lives out-of-repo. Purge any pre-adoption `.codebase-memory/`.
4. **Lock the tree** (`git-crypt lock`) — confirm `python tools/cbm/lockcheck.py` exits 0.
5. **Wire `.mcp.json`** (see the exact entry below) — the ONLY launch path, chaining the
   verify + the locked-tree preflight before the binary.
6. **First locked-tree index + leak-absence check.** Run once, then grep/`strings`/sqlite the
   produced DB for any known canary secret; abort adoption if anything leaks.

## The `.mcp.json` entry (add ONLY after steps 1–4)
Do not add this until the binary is pinned — a `.mcp.json` `command` that points at a missing
or unpinned binary would fail on every session start. The entry launches through BOTH gates:

```jsonc
// add under "mcpServers" in .mcp.json:
"codebase-memory": {
  "command": "python",
  "args": [
    "tools/cbm/index_guard.py", "--",           // (1) refuse unless tree is LOCKED + no key
    "python", "tools/cbm/verify_then_exec.py",  // (2) refuse unless sha256 matches the pin
    "${CBM_CACHE_DIR}/codebase-memory-mcp"       // the pinned binary (out-of-repo)
  ],
  "env": { "CBM_CACHE_DIR": "${CBM_CACHE_DIR}" }
}
```
`index_guard.py -- <cmd...>` execs `<cmd...>` only on a locked tree; the inner
`fetch_cbm.py --verify` (wrap as a tiny `verify_then_exec` shim, or call `--verify` in a
pre-step) gates on the byte-pin. Net: a raw invocation cannot bypass either gate.

## What is landed now (this PR) vs what is a human step
- **Landed:** `fetch_cbm.py` (pin+verify, 7 tests), `cbm.pin.json` (uninitialized on purpose),
  this plan, and the wiring recipe. All offline, no binary needed, no session breakage.
- **Human step (blocked on the binary + a locked tree):** steps 1–6 above — pin the real
  bytes, set the out-of-repo cache, lock, wire `.mcp.json`, first index + leak check.

## Residual risks (carried from the Phase-0 audit)
- **R1** the locked-tree check is at startup only; do not build a shareable artifact from a
  session unlocked mid-run.
- The `.claude/hooks/session_start.sh` lock primitive was **fixed** (deterministic 10-byte
  magic compare, #329) — the same class of bug that Phase 0's `lockcheck.py` avoids.
