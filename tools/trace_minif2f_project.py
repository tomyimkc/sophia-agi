# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""One-off: trace a miniF2F-v2 Lean 4 project so its trace is cached for the eval.

This is option (b) of preregistration.json -> tracedProject.cacheMatchRisk: when the
port's mathlib4 dependency is NOT in lean-dojo's remote cache, the trace must be
produced ONCE on a big Linux box (lean-dojo docs: mathlib-scale trace ~1 hr / 32 GB
RAM) and then persisted (actions/cache of ~/.cache/lean_dojo) so per-run verification
only touches the small miniF2F files. The macOS-arm64 box CANNOT do this (trace()
deadlocks — Lean-L0-Trace-Deadlock.md §1); run it on Linux only.

It is NOT an eval and produces NO capability number — it populates a cache and (best
effort) confirms the traced project resolves a Theorem through the real kernel without
fabricating. canClaimAGI stays false.

Driven by env vars (set by .github/workflows/formal-proofs-trace-cache.yml):
    MINIF2F_REPO     (required)  e.g. https://github.com/yangky11/miniF2F-lean4
    MINIF2F_COMMIT   (required)  the pinned commit SHA
    VERIFY_FILE      (optional)  a .lean file path inside the repo, for the sanity check
    VERIFY_THEOREM   (optional)  a theorem full-name in VERIFY_FILE
    VERIFY_WRONG_PROOF (optional, default "exact 1") a deliberately WRONG proof: the
                     sanity check asserts it is NOT accepted (no fabrication). We use a
                     wrong proof on purpose because miniF2F theorems are unproved stubs,
                     so there is no bundled correct proof to assert `accepted` on.

Exit 0 = trace completed (cache populated) and any requested sanity check passed.
Exit 1 = trace failed / sanity check fabricated an accept / bad config.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(f"ERROR: {name} is required", file=sys.stderr)
        raise SystemExit(2)
    return val


def main() -> int:
    repo_url = _require("MINIF2F_REPO")
    commit = _require("MINIF2F_COMMIT")

    try:
        from lean_dojo import LeanGitRepo, trace  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only with lean-dojo installed
        print(f"ERROR: lean-dojo not importable: {exc!r}", file=sys.stderr)
        return 1

    print(f"[trace] LeanGitRepo({repo_url}, {commit})", flush=True)
    repo = LeanGitRepo(repo_url, commit)

    # This is the heavy step: trace the dependency closure (mathlib4-scale). lean-dojo
    # consults its remote cache per dependency, so an already-cached mathlib4 is
    # downloaded rather than re-extracted; anything uncached is extracted locally.
    print("[trace] tracing (this can take ~1 hr on first run; cache makes reruns fast)...",
          flush=True)
    traced = trace(repo)  # writes into ~/.cache/lean_dojo
    print(f"[trace] DONE — traced repo at: {getattr(traced, 'root_dir', traced)}", flush=True)

    verify_file = os.environ.get("VERIFY_FILE", "").strip()
    verify_thm = os.environ.get("VERIFY_THEOREM", "").strip()
    if verify_file and verify_thm:
        from agent import lean_backend  # local import: only needed for the sanity check
        from lean_dojo import Theorem  # type: ignore
        wrong = os.environ.get("VERIFY_WRONG_PROOF", "exact 1")
        print(f"[verify] check_proof on {verify_thm} in {verify_file} with a WRONG proof "
              f"({wrong!r}) — must NOT be accepted", flush=True)
        thm = Theorem(repo, Path(verify_file), verify_thm)
        r = lean_backend.check_proof_in_repo(thm, wrong)
        print(f"[verify] verdict={r.verdict} reason={r.reason}", flush=True)
        if r.verdict == "accepted":
            print("ERROR: a deliberately WRONG proof was ACCEPTED — fabrication bug",
                  file=sys.stderr)
            return 1
        print("[verify] OK — traced project resolves a Theorem and the kernel path is live "
              "(no fabrication).", flush=True)
    else:
        print("[verify] skipped (set VERIFY_FILE + VERIFY_THEOREM to run the sanity check)",
              flush=True)

    print("[trace] SUCCESS — ~/.cache/lean_dojo is populated; persist it via actions/cache.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
