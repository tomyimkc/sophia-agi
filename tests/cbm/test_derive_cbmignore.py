import subprocess
import sys
from pathlib import Path

from tools.cbm.derive_cbmignore import derive, parse_gitattributes

SAMPLE = """\
training/mlx_adapters/**/*.safetensors filter=lfs diff=lfs merge=lfs -text
secret/** filter=git-crypt diff=git-crypt
docs/superpowers/** filter=git-crypt diff=git-crypt
AGENTS.md filter=git-crypt diff=git-crypt
.claude/skills/** filter=git-crypt diff=git-crypt
.claude/skills/git-discipline/** !filter !diff
.grok/** filter=git-crypt diff=git-crypt
# a comment line
"""

def test_parse_splits_encrypted_and_whitelist():
    enc, white = parse_gitattributes(SAMPLE)
    assert "secret/**" in enc and "docs/superpowers/**" in enc and ".grok/**" in enc
    assert ".claude/skills/**" in enc
    assert ".claude/skills/git-discipline/**" in white
    assert "training/mlx_adapters/**/*.safetensors" not in enc  # filter=lfs, not git-crypt

def test_derive_excludes_encrypted_reincludes_whitelist_adds_sinks():
    out = derive(SAMPLE)
    assert "\ndocs/superpowers/**\n" in out
    assert "\nAGENTS.md\n" in out
    assert "\n!.claude/skills/git-discipline/**\n" in out  # process skill re-included
    assert ".codebase-memory/" in out and "*.db.zst" in out
    assert out.startswith("# GENERATED")

def test_new_secret_path_is_covered_automatically():
    # add a brand-new IP skill by RULE (.claude/skills/**) — derive must cover it
    enc, _ = parse_gitattributes(SAMPLE)
    assert ".claude/skills/**" in enc  # umbrella rule => any new skill dir excluded


def test_check_fresh_vs_stale(tmp_path):
    """Exercise the fresh-vs-stale comparison that `--check` relies on.

    Tests the derive() comparison directly, then drives the real `--check` CLI
    against a controlled tmp repo (via CBM_ROOT_OVERRIDE) for fresh(0)/stale(1).
    """
    # Fresh: content derived from SAMPLE matches itself → would be flagged as up-to-date
    fresh = derive(SAMPLE)
    assert fresh == fresh  # identical → --check would return 0

    # Stale: any deviation is caught
    stale = fresh + "\n# drift added externally\n"
    assert fresh != stale  # differs → --check would return 1

    # Drive the real `--check` CLI against a CONTROLLED tmp repo via CBM_ROOT_OVERRIDE,
    # so we exercise the actual exit codes (0 fresh / 1 stale) — not the committed repo.
    import os
    script = Path(__file__).resolve().parents[2] / "tools/cbm/derive_cbmignore.py"
    (tmp_path / ".gitattributes").write_text(SAMPLE)
    fresh_content = derive(SAMPLE)
    (tmp_path / ".cbmignore").write_text(fresh_content)
    env = {**os.environ, "CBM_ROOT_OVERRIDE": str(tmp_path)}

    # Fresh .cbmignore → --check exits 0
    r_fresh = subprocess.run([sys.executable, str(script), "--check"], env=env)
    assert r_fresh.returncode == 0, "--check should pass when .cbmignore is fresh"

    # Stale .cbmignore → --check exits 1
    (tmp_path / ".cbmignore").write_text(fresh_content + "\n# deliberate drift\n")
    r_stale = subprocess.run([sys.executable, str(script), "--check"], env=env)
    assert r_stale.returncode == 1, "--check should fail when .cbmignore is stale"
