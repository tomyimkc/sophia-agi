# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Seal held-out math/code benchmark splits → public SHA-256 manifest only.

Full prompt/answer payloads are copied under gitignored ``private/``; the committed
manifest records file-level and per-item hashes so third parties can verify the
split was fixed before training. Curriculum generators must not read sealed paths
(see ``tools/heldout_seal_guard.py``).

    python tools/seal_math_code_heldout.py
    python tools/seal_math_code_heldout.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST_OUT = ROOT / "agi-proof" / "sophia-math-code-curriculum" / "heldout-seal.manifest.json"
PRIVATE_DIR = ROOT / "private" / "math-code-heldout"

# Surfaces cited as evidence-oracle held-out (style samples + code uplift tasks).
SEAL_PATHS: list[str] = [
    "eval/external/math-style-sample.jsonl",
    "eval/external/gsm8k-style-sample.jsonl",
    "eval/coding/mbpp-style-sample.jsonl",
    "eval/coding/humaneval-style-sample.jsonl",
    "benchmark/code_tasks.json",
    "provenance_bench/data/math_problems.json",
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _item_digest(row: dict) -> str:
    return _sha256_text(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def _items_from_file(rel: str, path: Path) -> list[dict]:
    if rel.endswith(".jsonl"):
        return _load_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if rel.endswith("math_problems.json"):
        return [p for p in data.get("problems", []) if p.get("split") == "eval"]
    if rel.endswith("code_tasks.json"):
        return list(data.get("tasks", []))
    if isinstance(data, list):
        return data
    return []


def build_manifest() -> dict:
    sealed_at = datetime.now(timezone.utc).isoformat()
    files: list[dict] = []
    for rel in SEAL_PATHS:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"seal path missing: {rel}")
        raw = path.read_bytes()
        items = _items_from_file(rel, path)
        files.append({
            "path": rel,
            "sha256": _sha256_bytes(raw),
            "itemCount": len(items),
            "items": [
                {"id": str(it.get("id", i)), "sha256": _item_digest(it)}
                for i, it in enumerate(items)
            ],
        })
    return {
        "schema": "sophia.math_code_heldout_seal.v1",
        "packId": "math-code-curriculum-heldout-2026-06-25",
        "sealedAt": sealed_at,
        "visibility": "public-hash-only",
        "privateCopy": "private/math-code-heldout/ (gitignored)",
        "generatorPolicy": "Curriculum generators MUST NOT read sealed paths; train on sympy/exec-verified synthetic packs only.",
        "files": files,
    }


def _copy_private(manifest: dict) -> None:
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        src = ROOT / entry["path"]
        dst = PRIVATE_DIR / Path(entry["path"]).name
        shutil.copy2(src, dst)
    (PRIVATE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def check_manifest() -> int:
    if not MANIFEST_OUT.exists():
        print(f"MISSING manifest: {MANIFEST_OUT}", file=sys.stderr)
        return 1
    on_disk = json.loads(MANIFEST_OUT.read_text(encoding="utf-8"))
    fresh = build_manifest()
    if on_disk.get("files") != fresh["files"]:
        print("held-out seal manifest is STALE — run tools/seal_math_code_heldout.py", file=sys.stderr)
        return 1
    print(f"held-out seal OK ({len(fresh['files'])} files)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify manifest matches on-disk splits")
    args = ap.parse_args(argv)
    if args.check:
        return check_manifest()
    manifest = build_manifest()
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _copy_private(manifest)
    print(f"wrote {MANIFEST_OUT} ({len(manifest['files'])} sealed files)")
    print(f"copied payloads to {PRIVATE_DIR} (gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
