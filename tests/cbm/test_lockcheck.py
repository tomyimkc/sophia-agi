# tests/cbm/test_lockcheck.py
import subprocess, sys
from pathlib import Path
from tools.cbm.lockcheck import is_locked, GIT_CRYPT_MAGIC

def test_ciphertext_is_locked(tmp_path):
    f = tmp_path / "AGENTS.md"
    f.write_bytes(GIT_CRYPT_MAGIC + b"\xf3\xbb\xecrest-of-ciphertext")
    assert is_locked(str(f)) is True

def test_plaintext_is_unlocked(tmp_path):
    f = tmp_path / "AGENTS.md"
    f.write_bytes(b"# Sophia AGI\n")
    assert is_locked(str(f)) is False

def test_missing_file_is_not_locked(tmp_path):
    assert is_locked(str(tmp_path / "nope.md")) is False

def test_plaintext_containing_word_GITCRYPT_is_unlocked(tmp_path):
    # guards against the substring false-positive the audit flagged
    f = tmp_path / "AGENTS.md"
    f.write_bytes(b"GITCRYPT is mentioned here as plain text")
    assert is_locked(str(f)) is False

def test_cli_exit_codes(tmp_path):
    locked = tmp_path / "a"; locked.write_bytes(GIT_CRYPT_MAGIC + b"x")
    plain = tmp_path / "b"; plain.write_bytes(b"plain")
    root = Path(__file__).resolve().parents[2]
    r_lock = subprocess.run([sys.executable, "tools/cbm/lockcheck.py", str(locked)], cwd=root)
    r_plain = subprocess.run([sys.executable, "tools/cbm/lockcheck.py", str(plain)], cwd=root)
    assert r_lock.returncode == 0 and r_plain.returncode == 1
