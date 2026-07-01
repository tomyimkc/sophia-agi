#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Seal an externally-authored eval pack — a stable manifest hash that proves the pack the
runner scored is the exact, unspent pack that was committed BEFORE the run (independence-eval-plan
.md §1: "commit only the manifest hash until the run is complete; reveal items post-hoc").

The hash is a sha256 over the CANONICALIZED items (each line parsed as JSON, re-serialized with
sorted keys + compact separators, joined by '\\n'), so it is invariant to key order / whitespace /
trailing-newline noise but changes if any item content changes -> tamper detection. Deterministic,
stdlib only.

    # seal a pack (write <pack>.manifest.json next to it)
    python3 tools/seal_eval_pack.py --pack PACK.jsonl --author "ext-reviewer:jane" \\
        --sealed-at 2026-06-29T00:00:00Z

    # verify a committed pack is unspent / untampered against its manifest
    python3 tools/seal_eval_pack.py --pack PACK.jsonl --verify

Exit: 0 = sealed / verified OK, 1 = verify MISMATCH (tampered or wrong pack), 2 = bad inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _canonical_items(pack_path: Path) -> "list[str]":
    """Parse a JSONL pack into a list of canonical (sorted-key, compact) JSON strings.

    Blank lines are skipped; a non-JSON line raises ValueError so a corrupt pack can never be
    silently sealed. Canonicalization makes the hash invariant to formatting but sensitive to
    any content change."""
    items: list[str] = []
    for ln, raw in enumerate(pack_path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{pack_path}:{ln}: not valid JSON ({exc})") from exc
        items.append(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return items


def manifest_hash(pack_path: Path) -> "tuple[str, int]":
    """Return (sha256_hex, item_count) over the canonicalized pack. Pure function of pack content."""
    items = _canonical_items(pack_path)
    h = hashlib.sha256()
    for it in items:
        h.update(it.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest(), len(items)


def manifest_path(pack_path: Path) -> Path:
    return pack_path.with_name(pack_path.name + ".manifest.json")


def seal(pack_path: Path, author: str, sealed_at: str) -> dict:
    """Compute the manifest and write <pack>.manifest.json. `sealed_at` is PASSED IN (never
    auto-dated) so the artifact is reproducible and the commit timestamp — not wall-clock — is the
    source of truth for ordering."""
    digest, count = manifest_hash(pack_path)
    manifest = {
        "pack": pack_path.name,
        "algo": "sha256",
        "hash": digest,
        "count": count,
        "author": author,
        "sealedAt": sealed_at,
        "canonicalization": "per-line json.loads -> json.dumps(sort_keys=True, separators=(',',':')) joined by '\\n'",
        "boundary": "Sealed manifest of an externally-authored pack. One pack = one spend; "
                    "--verify enforces the scored pack is unspent/untampered (independence-eval-plan.md).",
    }
    return manifest


def verify(pack_path: Path) -> dict:
    """Re-hash the pack and compare to its committed manifest. Returns a result dict with `ok`."""
    mpath = manifest_path(pack_path)
    if not mpath.exists():
        return {"ok": False, "reason": f"no manifest at {mpath.name} — pack was never sealed",
                "pack": pack_path.name}
    committed = json.loads(mpath.read_text(encoding="utf-8"))
    digest, count = manifest_hash(pack_path)
    ok = (digest == committed.get("hash")) and (count == committed.get("count"))
    return {
        "ok": ok,
        "pack": pack_path.name,
        "expectedHash": committed.get("hash"),
        "actualHash": digest,
        "expectedCount": committed.get("count"),
        "actualCount": count,
        "author": committed.get("author"),
        "sealedAt": committed.get("sealedAt"),
        "reason": "match (unspent/untampered)" if ok else "MISMATCH — pack tampered or wrong pack",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, required=True, help="the eval pack JSONL")
    ap.add_argument("--author", default="", help="pack author identity (independent of this repo)")
    ap.add_argument("--sealed-at", default="", help="ISO8601 seal timestamp (PASSED IN, not auto-dated)")
    ap.add_argument("--verify", action="store_true", help="re-hash and check against committed manifest")
    ap.add_argument("--out", type=Path, default=None, help="manifest path (default <pack>.manifest.json)")
    args = ap.parse_args()

    if not args.pack.exists():
        print(json.dumps({"ok": False, "reason": f"pack not found: {args.pack}"}))
        return 2

    if args.verify:
        res = verify(args.pack)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res["ok"] else 1

    if not args.author or not args.sealed_at:
        print(json.dumps({"ok": False, "reason": "sealing requires --author and --sealed-at "
                                                 "(sealed-at is passed in, never auto-dated)"}))
        return 2
    try:
        manifest = seal(args.pack, args.author, args.sealed_at)
    except ValueError as exc:
        print(json.dumps({"ok": False, "reason": str(exc)}))
        return 2
    out = args.out or manifest_path(args.pack)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "wrote": str(out), "hash": manifest["hash"],
                      "count": manifest["count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
