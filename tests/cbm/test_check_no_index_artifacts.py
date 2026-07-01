# tests/cbm/test_check_no_index_artifacts.py
import subprocess
from pathlib import Path
from tools.cbm.check_no_index_artifacts import SINK_RE, find_violations

def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

def test_sink_regex_positive():
    for p in [".codebase-memory/graph.db.zst", "graph.db", "graph.db.zst",
              "sub/dir/x.db.zst", ".codebase-memory/nested/a"]:
        assert SINK_RE.search(p), p

def test_sink_regex_negative():
    for p in ["tools/cbm/lockcheck.py", "docs/graph.dbx.md", "notes.zst.txt", "database.py"]:
        assert not SINK_RE.search(p), p

def _init_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "ok.py").write_text("x = 1\n")
    _git(tmp_path, "add", "ok.py")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path

def test_clean_repo_has_no_violations(tmp_path):
    r = _init_repo(tmp_path)
    assert find_violations(staged_only=False, cwd=str(r)) == []

def test_force_added_artifact_is_caught_by_path(tmp_path):
    r = _init_repo(tmp_path)
    (r / ".codebase-memory").mkdir()
    (r / ".codebase-memory" / "graph.db.zst").write_bytes(b"\x28\xb5\x2f\xfdCANARY")
    _git(r, "add", "-f", ".codebase-memory/graph.db.zst")  # force past a would-be gitignore
    viol = find_violations(staged_only=True, cwd=str(r))
    assert any("graph.db.zst" in v for v in viol)

def test_magic_bytes_caught_outside_codebase_memory_when_staged(tmp_path):
    r = _init_repo(tmp_path)
    (r / "renamed_blob").write_bytes(b"\x28\xb5\x2f\xfdzstd-body")  # innocent name, not under .codebase-memory/
    _git(r, "add", "-f", "renamed_blob")
    viol = find_violations(staged_only=True, cwd=str(r))
    assert any("renamed_blob" in v and "index blob magic bytes" in v for v in viol)

def test_magic_bytes_not_scanned_in_tracked_mode(tmp_path):
    r = _init_repo(tmp_path)
    (r / "renamed_blob").write_bytes(b"\x28\xb5\x2f\xfdzstd-body")
    _git(r, "add", "-f", "renamed_blob")
    _git(r, "commit", "-qm", "add blob")
    # tracked/full scan is path-only by design; an out-of-dir magic blob is not flagged
    assert find_violations(staged_only=False, cwd=str(r)) == []

def test_cli_exit_code(tmp_path):
    r = _init_repo(tmp_path)
    (r / ".codebase-memory").mkdir()
    (r / ".codebase-memory" / "graph.db.zst").write_bytes(b"x")
    _git(r, "add", "-f", ".codebase-memory/graph.db.zst")
    root = Path(__file__).resolve().parents[2]
    res = subprocess.run(["python", str(root / "tools/cbm/check_no_index_artifacts.py"),
                          "--staged"], cwd=str(r))
    assert res.returncode == 1
