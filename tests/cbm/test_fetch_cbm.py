"""Tests for the Phase-1 pin+verify mechanism (tools/cbm/fetch_cbm.py).

Exercises the SECURITY-critical part — verify refuses unless the pin is initialized AND
the binary's sha256 matches exactly — without needing the real third-party binary.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.cbm.fetch_cbm import load_pin, sha256_file, verify, main  # noqa: E402


def _write_binary(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_sha256_file_matches_hashlib(tmp_path):
    b = tmp_path / "bin"
    expected = _write_binary(b, b"\x7fELF fake binary bytes")
    assert sha256_file(b) == expected


def test_verify_refuses_when_pin_uninitialized(tmp_path):
    b = tmp_path / "bin"
    _write_binary(b, b"anything")
    ok, msg = verify(b, {"sha256": "", "ref": "x"})
    assert ok is False and "not initialized" in msg.lower()


def test_verify_refuses_missing_binary(tmp_path):
    ok, msg = verify(tmp_path / "nope", {"sha256": "a" * 64})
    assert ok is False and "not found" in msg


def test_verify_refuses_on_mismatch(tmp_path):
    b = tmp_path / "bin"
    _write_binary(b, b"real bytes")
    ok, msg = verify(b, {"sha256": "b" * 64, "ref": "r"})
    assert ok is False and "MISMATCH" in msg


def test_verify_passes_on_exact_match(tmp_path):
    b = tmp_path / "bin"
    digest = _write_binary(b, b"the audited, pinned binary")
    ok, msg = verify(b, {"sha256": digest, "ref": "r1"})
    assert ok is True and "verified" in msg


def test_init_then_verify_roundtrip(tmp_path):
    # a pin file with empty sha256 + a binary
    pin_file = tmp_path / "cbm.pin.json"
    pin_file.write_text(json.dumps({"repo": "x", "ref": "r", "binary_rel": "b", "sha256": ""}))
    b = tmp_path / "bin"
    _write_binary(b, b"pin me")

    # --init records the sha256
    rc = main(["--init", str(b), "--pin", str(pin_file)])
    assert rc == 0
    assert load_pin(pin_file)["sha256"] == sha256_file(b)

    # --verify now passes (exit 0)
    assert main(["--verify", str(b), "--pin", str(pin_file)]) == 0

    # tamper the binary -> --verify fails (exit 1)
    b.write_bytes(b"pin me + tampered")
    assert main(["--verify", str(b), "--pin", str(pin_file)]) == 1


def test_committed_pin_is_uninitialized(tmp_path):
    """The repo-committed cbm.pin.json must ship with an EMPTY sha256 (indexing disabled by default)."""
    pin = load_pin(ROOT / "cbm.pin.json")
    assert pin.get("sha256", "") == "", "committed pin must be uninitialized (empty sha256)"
    assert "codebase-memory-mcp" in pin.get("repo", "")
