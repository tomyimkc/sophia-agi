import tools.cbm.index_guard as g


def test_preflight_ok_when_locked_and_no_key():
    assert g.preflight(env={}, locked=True) is None


def test_preflight_refuses_when_unlocked():
    msg = g.preflight(env={}, locked=False)
    assert msg and "UNLOCKED" in msg


def test_preflight_refuses_when_key_present_even_if_locked():
    msg = g.preflight(env={"GITCRYPT_KEY_B64": "x"}, locked=True)
    assert msg and "GITCRYPT_KEY_B64" in msg


def test_main_returns_1_and_does_not_exec_when_unlocked(monkeypatch):
    calls = []
    monkeypatch.setattr(g.os, "execvp", lambda f, a: calls.append((f, a)))
    monkeypatch.setattr(g, "is_locked", lambda *a, **k: False)
    rc = g.main(["index_guard.py", "--", "true"])
    assert rc == 1 and calls == []


def test_main_execs_when_locked(monkeypatch):
    calls = []
    monkeypatch.setattr(g.os, "execvp", lambda f, a: calls.append((f, a)))
    monkeypatch.setattr(g, "is_locked", lambda *a, **k: True)
    monkeypatch.setattr(g.os, "environ", {})
    rc = g.main(["index_guard.py", "--", "mybin", "serve"])
    assert calls == [("mybin", ["mybin", "serve"])]


def test_main_usage_error_without_dashdash():
    assert g.main(["index_guard.py", "mybin"]) == 2


def test_script_mode_imports_cleanly():
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, "tools/cbm/index_guard.py"], cwd=str(root),
                       capture_output=True, text=True)
    # usage error (exit 2), NOT an ImportError/ModuleNotFoundError from script-mode import
    assert r.returncode == 2, r.stderr
    assert "ImportError" not in r.stderr and "ModuleNotFoundError" not in r.stderr
