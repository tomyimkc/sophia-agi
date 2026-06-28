# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic test for tools/verify_replication_manifest.py (no network).

Asserts the offline manifest checker reports PASS on the committed repo: every module/test/
bench/pack/report listed in EXPECTED-RESULTS.json exists, and every referenced report (and the
manifest itself) has canClaimAGI=false.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "tools" / "verify_replication_manifest.py"
    spec = importlib.util.spec_from_file_location("verify_replication_manifest", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_exists():
    assert (ROOT / "agi-proof" / "verification-replication" / "EXPECTED-RESULTS.json").exists()


def test_check_returns_pass():
    mod = _load_module()
    ok, problems = mod.check()
    assert problems == [], f"manifest check reported problems: {problems}"
    assert ok, "expected at least one ok check"


def test_main_exits_zero():
    mod = _load_module()
    assert mod.main() == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok {name}")
    print("all passed")
