"""Tests for the verify-then-exec shim (tools/cbm/verify_then_exec.py).

The security property: it REFUSES (never execs) unless the binary's sha256 matches the pin;
on a match it execs the binary. os.execvp is mocked so the test process is not replaced.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.cbm.verify_then_exec as vte  # noqa: E402


def test_refuses_when_pin_uninitialized(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.write_bytes(b"anything")
    monkeypatch.setattr(vte, "load_pin", lambda *a, **k: {"sha256": ""})
    called = {}
    monkeypatch.setattr(vte.os, "execvp", lambda *a, **k: called.setdefault("hit", True))
    assert vte.main([str(b)]) == 1
    assert not called, "must NOT exec when the pin is uninitialized"


def test_refuses_on_sha256_mismatch(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.write_bytes(b"real")
    monkeypatch.setattr(vte, "load_pin", lambda *a, **k: {"sha256": "b" * 64})
    called = {}
    monkeypatch.setattr(vte.os, "execvp", lambda *a, **k: called.setdefault("hit", True))
    assert vte.main([str(b)]) == 1
    assert not called, "must NOT exec on a sha256 mismatch"


def test_execs_when_verified(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.write_bytes(b"the audited bytes")
    digest = hashlib.sha256(b"the audited bytes").hexdigest()
    monkeypatch.setattr(vte, "load_pin", lambda *a, **k: {"sha256": digest, "ref": "r1"})
    captured = {}
    monkeypatch.setattr(vte.os, "execvp", lambda file, args: captured.update(file=file, args=args))
    vte.main([str(b), "--serve", "--port", "9"])
    assert captured["file"] == str(b)
    assert captured["args"] == [str(b), "--serve", "--port", "9"]


def test_usage_when_no_binary():
    assert vte.main([]) == 2
