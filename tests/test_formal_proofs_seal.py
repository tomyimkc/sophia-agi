# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the formal-proofs held-out seal tool + guard (Phase-1 leakage firewall).

Exercises the hash/seal/check logic and the proposer guard against a TINY fixture, so
the firewall is validated without needing the real miniF2F-v2 split (which is pinned +
sealed at registration time — see agi-proof/formal-proofs-curriculum/preregistration.json
openChecklist).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import seal_formal_proofs_heldout as seal
from tools.heldout_seal_guard import assert_generator_safe, sealed_paths

ROOT = Path(__file__).resolve().parents[1]

_FIXTURE_ITEMS = [
    {"claim_id": "mathd_fixture_1", "proposition": "1 + 1 = 2",
     "lean_statement": "theorem mathd_fixture_1 : 1 + 1 = 2 := by"},
    {"claim_id": "mathd_fixture_2", "proposition": "∀ (n : Nat), n + 0 = n",
     "lean_statement": "theorem mathd_fixture_2 (n : Nat) : n + 0 = n := by"},
]


def _write_fixture(src: Path, commit: str = "deadbeefcafe") -> None:
    src.mkdir(parents=True, exist_ok=True)
    (src / seal.SOURCE_FILE).write_text(
        json.dumps({"repo": "https://github.com/facebookresearch/minif2f",
                    "commit": commit, "split": "test"}), encoding="utf-8")
    with (src / seal.SPLIT_FILE).open("w", encoding="utf-8") as fh:
        for it in _FIXTURE_ITEMS:
            fh.write(json.dumps(it) + "\n")


def test_build_manifest_hashes_and_pins_commit(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_fixture(src, commit="abc123")
    m = seal.build_manifest(src)
    assert m["schema"] == "sophia.formal_proofs_heldout_seal.v1"
    assert m["visibility"] == "public-hash-only"
    assert m["source"]["commit"] == "abc123"
    assert m["source"]["benchmark"] == "miniF2F-v2"
    f = m["files"][0]
    assert f["itemCount"] == 2
    assert [it["id"] for it in f["items"]] == ["mathd_fixture_1", "mathd_fixture_2"]
    # Hashes are real sha256 hex digests, and per-item digests are distinct.
    assert all(len(it["sha256"]) == 64 for it in f["items"])
    assert f["items"][0]["sha256"] != f["items"][1]["sha256"]
    # The committed manifest carries only hashes/ids — never the raw lean statements.
    assert "lean_statement" not in json.dumps(m)


def test_seal_then_check_roundtrip_and_tamper(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "manifest.json"
    _write_fixture(src)
    private_split = seal.PRIVATE_DIR / seal.SPLIT_FILE
    before_private = private_split.read_bytes() if private_split.exists() else None
    assert seal.main(["--source", str(src), "--out", str(out)]) == 0
    assert out.exists()
    after_private = private_split.read_bytes() if private_split.exists() else None
    assert after_private == before_private
    # A fresh check against the same source passes.
    assert seal.check_manifest(src, out) == 0
    # Tampering with the split (adding an item) makes the committed manifest STALE.
    with (src / seal.SPLIT_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"claim_id": "sneaked_in", "proposition": "0 = 0"}) + "\n")
    assert seal.check_manifest(src, out) == 1


def test_check_reports_not_sealed_when_source_missing(tmp_path: Path) -> None:
    # No split staged → check must fail closed (the openChecklist seal step isn't done).
    assert seal.check_manifest(tmp_path / "absent", tmp_path / "absent-manifest.json") == 1


def test_guard_blocks_formal_proofs_private_prefix() -> None:
    blocked = sealed_paths()
    assert any("formal-proofs-heldout" in str(p) for p in blocked)
    sealed_file = ROOT / "private" / "formal-proofs-heldout" / "minif2f-v2-test.jsonl"
    with pytest.raises(RuntimeError, match="blocked"):
        assert_generator_safe(sealed_file)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
