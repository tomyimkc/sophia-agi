# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fail closed when a curriculum generator reads sealed held-out benchmark paths."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "agi-proof" / "sophia-math-code-curriculum" / "heldout-seal.manifest.json"
PRIVATE_PREFIX = ROOT / "private" / "math-code-heldout"


def sealed_paths(*, root: Path = ROOT) -> set[Path]:
    """Resolved paths that generators must not load for training data."""
    out: set[Path] = set()
    manifest_path = root / MANIFEST.relative_to(ROOT)
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in data.get("files", []):
            out.add((root / entry["path"]).resolve())
    out.add((root / PRIVATE_PREFIX.relative_to(ROOT)).resolve())
    return out


def assert_generator_safe(path: Path | str, *, root: Path = ROOT) -> None:
    """Raise RuntimeError if ``path`` is a sealed held-out benchmark surface."""
    resolved = Path(path).resolve()
    blocked = sealed_paths(root=root)
    for b in blocked:
        try:
            resolved.relative_to(b)
            raise RuntimeError(
                f"generator blocked from reading sealed held-out path: {resolved} "
                f"(under {b}). Train on sympy/exec-verified synthetic packs only."
            )
        except ValueError:
            continue
    for b in blocked:
        if resolved == b:
            raise RuntimeError(
                f"generator blocked from reading sealed held-out path: {resolved}. "
                "Train on sympy/exec-verified synthetic packs only."
            )
