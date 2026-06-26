<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 tomyimkc -->
# Rust Workspace Consolidation — one build for the compatible trees, honest separation for the rest

> **Scope, stated plainly.** This repo had **three independent Rust trees** that
> each built separately (three `cargo build`s, three `Cargo.lock`s, three
> `target/`s). [HANDOVER.md](../../HANDOVER.md) flagged `storage/kvcache` vs
> `sophia-storage/crates/sophia-kvcache` as a possible duplicate/drift. This
> change consolidates the trees that are **genuinely compatible** into one
> top-level virtual workspace, and documents why the one tree that is
> *intentionally* separate stays separate. It does **not** force-merge
> incompatible trees, and it does **not** delete code that turned out not to be a
> duplicate. The guiding rule is the same as everywhere else here: no overclaim,
> fail-closed, measured.

## The three trees, as found

| Tree | Kind | Edition / resolver | MSRV | Members | Deps |
|---|---|---|---|---|---|
| `storage/` | workspace | 2021 / resolver `"2"` | (none) | `kvcache`, `diskstore`, `miniraft`, `infcache`, `raftkv` | `tokio`, `libc`, `io-uring` (opt) |
| `sophia-storage/` | **separate** workspace | 2024 / resolver `"3"` | `1.85` | `sophia-lsm`, `sophia-kvcache` | none (std-only libs) |
| `services/ann_serving/` | standalone package | 2021 | (none) | `sophia-ann-serving` | **zero** |

`sophia-storage/`'s own header says it out loud: *"an isolated, optional Rust
workspace … Nothing in the Python codebase depends on this; it is feature-gated
and built separately."* It uses `[workspace.package]` field inheritance, declares
an MSRV for crates.io consumers, and deliberately omits `panic = "abort"` so it
stays embeddable as a published library.

## The duplicate verdict: NOT duplicates — distinct crates that share a name fragment

`storage/kvcache` and `sophia-storage/crates/sophia-kvcache` are **different
crates solving different problems**. They are not copies, not drift of one
original, and should both be kept.

Evidence:

| | `storage/kvcache` | `sophia-storage/crates/sophia-kvcache` |
|---|---|---|
| crate name | `kvcache` | `sophia-kvcache` (lib `sophia_kvcache`) |
| one-liner | "Sharded async **in-memory** KV cache" | "Disaggregated, **prefix-sharing** KV-cache for LLM inference — paged blocks **tiered across HBM/DRAM/NVMe**" |
| problem | a network cache *server* (TCP, Tokio) | an in-process inference KV-cache *library* |
| modules | `cache`, `client`, `lru`, `protocol`, `server` | `block`, `eviction`, `prefix`, `store`, `tier` |
| public API | `ShardedCache`, `Client`, `serve`, `Request/Response` | `KvCache`, `Config`, `AdmitResult`, `block::Block`, `tier::TierStack` |
| dependencies | `tokio` (async runtime, sockets) | none (std-only, `#![forbid(unsafe_code)]`) |
| edition | 2021 | 2024 |
| LOC (`.rs`) | 1100 | 1115 |

There is **zero module overlap and zero shared type**. The only identifiers the
two crates have in common are generic collection-method names (`new`, `get`,
`insert`, `len`, `remove`, `is_empty`) — the coincidence any two container types
share, not shared structure. One speaks a length-prefixed TCP wire protocol and
routes keys to sharded LRU maps by FNV-1a; the other has no network at all and
implements content-addressed paged blocks with reference-counted prefix reuse and
a real on-disk NVMe tier. The shared substring `kvcache` is the *only* thing they
share. **Verdict: keep both; nothing to de-duplicate.** (The names are clear
enough in context — `kvcache` the server vs `sophia-kvcache` the inference cache
— that a rename was not warranted; this doc is the disambiguation.)

## Why `sophia-storage` cannot and should not be merged in

Four blockers, any one of which is sufficient; together they make a forced merge
both impossible and wrong:

1. **No nested workspaces.** A Cargo workspace cannot contain another
   `[workspace]`. `sophia-storage/Cargo.toml` *is* a workspace root; pulling it
   under a top-level workspace is not a thing Cargo allows.
2. **Edition / resolver mismatch.** It is edition 2024 + resolver `"3"`; the
   storage crates are edition 2021 + resolver `"2"`. A single workspace has one
   resolver. (rustc 1.94 in this environment supports both, but the feature
   unification semantics differ — silently changing a published library's
   resolver is exactly the kind of unmeasured behaviour change we avoid.)
3. **`[workspace.package]` inheritance + MSRV.** Its members do
   `version.workspace = true`, `edition.workspace = true`, `rust-version =
   "1.85"`, etc. Those inherited fields and the deliberate MSRV are part of its
   **publish** contract; folding it into a root workspace with different shared
   metadata would break that contract.
4. **Publish story.** It is meant to be `cargo publish`-able as standalone
   library crates, with no `panic = "abort"` leaking into downstream consumers.
   A repo-wide workspace profile would override that intent.

So `sophia-storage/` **stays its own workspace**, exactly as its header always
intended. This is honest separation, not fragmentation: it is a different layer
(a publishable library) with a different lifecycle.

## What changed — the chosen layout

A **top-level virtual workspace** at the repo root (`/Cargo.toml`) that unifies
the trees that are actually compatible:

```
/Cargo.toml                      ← NEW: virtual workspace (resolver "2")
  members = storage/{kvcache,diskstore,miniraft,infcache,raftkv}
          + services/ann_serving
  exclude = ["sophia-storage"]
  [profile.release]              ← opt-level 3 + thin LTO + 1 codegen-unit
/Cargo.lock                      ← NEW: single lockfile for all six members
storage/…                        ← unchanged crates; storage/Cargo.toml REMOVED
services/ann_serving/…           ← unchanged crate; its package-level
                                   [profile.release] removed (now governed by root)
sophia-storage/…                 ← UNTOUCHED, still its own workspace
```

Membership was chosen by what composes cleanly, verified with `cargo metadata` /
`cargo check` (below) — not assumed. All six members are edition 2021 and were
already resolver-`"2"` compatible, so unifying them changes no build behaviour.

### Files created
- **`Cargo.toml`** (repo root) — the virtual workspace manifest. Lists the six
  members, `exclude`s `sophia-storage`, and carries the workspace `[profile.release]`.
- **`Cargo.lock`** (repo root) — single generated lockfile for the unified workspace.

### Files modified
- **`services/ann_serving/Cargo.toml`** — removed its now-ignored package-level
  `[profile.release]` (Cargo only honours profiles at the workspace root; a
  member profile is silently ignored with a warning). Replaced with a comment
  pointing here. Behaviour note: that profile used fat LTO (`lto = true`); the
  workspace uses **thin** LTO. Thin LTO builds faster and keeps nearly all the
  inlining win; a consumer who wants fat LTO for the ann binaries can add a
  dedicated profile.
- **`storage/README.md`** — the line that called `storage/` the workspace root is
  now accurate: these crates are members of the root workspace.

### Files removed
- **`storage/Cargo.toml`** — was a competing workspace root. With the root
  workspace owning these members, leaving it in place created a *dual-root*
  ambiguity: `cargo` run from inside `storage/` resolved to `storage/` as the
  root (innermost wins → its own `Cargo.lock`/`target/`), while `cargo` at the
  repo root resolved to the root workspace. Removing it makes the root the single
  workspace for these crates from anywhere. Its `[profile.release]` (identical:
  opt-level 3 + thin LTO + 1 codegen-unit) was lifted verbatim to the root.
- **`storage/Cargo.lock`** — subsumed by the single root `Cargo.lock`.

### Files deliberately NOT changed
- **All of `sophia-storage/`** — separate workspace by design (see above).
- **The two kvcache crates' source** — not duplicates; nothing to merge.
- **`RESULTS.md` reproduce command** (`cd storage && cargo run … --bin
  kvcache-bench`) — still works: `cargo` walks up from `storage/` to the root
  workspace and resolves `--bin kvcache-bench` there. Verified.

## How to build it

```bash
# The unified workspace (storage/* + services/ann_serving), one lockfile:
cargo check --workspace            # from the repo root
cargo test  --workspace            # all six members
cargo build --release              # opt-level 3 + thin LTO

# A single member:
cargo test -p kvcache
cargo run --release --bin kvcache-server -- --addr 127.0.0.1:7070 --shards 16 --capacity 1000000

# The intentionally-separate publishable workspace, built on its own:
cargo check --manifest-path sophia-storage/Cargo.toml --workspace
```

## Verification (commands run, outputs)

Environment: `rustc 1.94.1`, `cargo 1.94.1`.

**Unified root workspace — clean, all six members, no warnings:**
```
$ cargo check --workspace        # from repo root
    Checking sophia-ann-serving v0.1.0 (services/ann_serving)
    Checking miniraft v0.1.0 (storage/miniraft)
    Checking diskstore v0.1.0 (storage/diskstore)
    Checking raftkv v0.1.0 (storage/raftkv)
    Checking kvcache v0.1.0 (storage/kvcache)
    Checking infcache v0.1.0 (storage/infcache)
    Finished `dev` profile [unoptimized + debuginfo] target(s)
```
(Before removing `ann_serving`'s member-level profile, this printed
`warning: profiles for the non root package will be ignored …`; after the
removal the build is warning-free.)

**Single root, resolved from inside a member** — proves the dual-root ambiguity is gone:
```
$ cd storage/kvcache && cargo metadata --no-deps --format-version 1 | jq -r .workspace_root
/home/…/sophia-agi          # the repo root, not storage/
```

**Release profile honoured at root:**
```
$ cargo build --release -p sophia-ann-serving -p kvcache
    Finished `release` profile [optimized] target(s)
```

**`sophia-storage` still independent and intact:**
```
$ cd sophia-storage && cargo metadata --no-deps | jq -r '.workspace_root, (.packages[].name)'
/home/…/sophia-agi/sophia-storage
sophia-lsm
sophia-kvcache
$ cargo check --workspace
    Checking sophia-lsm v0.1.0
    Checking sophia-kvcache v0.1.0
    Finished `dev` profile target(s)
```

## Net effect

- Before: 3 trees, 3 `cargo build`s, 3 lockfiles.
- After: **2** trees — one unified virtual workspace (6 members, 1 lockfile, 1
  `cargo build --workspace`) plus the deliberately-separate `sophia-storage`
  library workspace. No source code merged or deleted under a false "duplicate"
  premise; the only deletions are the redundant `storage/Cargo.toml` /
  `Cargo.lock` whose role the root now fills. Fully reversible: restore
  `storage/Cargo.toml` + `storage/Cargo.lock`, drop the root `Cargo.toml`, and
  re-add `ann_serving`'s `[profile.release]` to revert.
