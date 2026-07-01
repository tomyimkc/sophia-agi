#!/usr/bin/env python3
"""Offline tests for make_independent_hidden_pack.py — all fail-closed paths."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_independent_hidden_pack.py"
SCHEMA = Path("agi-proof/hidden-reviewer-packs/schema.json")


def _run(args: list[str], tmp: Path) -> subprocess.CompletedProcess:
    # The tool binds to the runner's real validate_pack (review D3), so the repo
    # must be importable. Real invocation runs from repo root with PYTHONPATH=.;
    # replicate that here (repo root is two levels up from this test file).
    import os
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True, text=True, cwd=str(tmp), env=env,
    )


def _four_domain_items() -> dict:
    return {"cases": [
        {"id": "p1", "domain": "philosophy", "prompt": "zzq novelstring alpha bravo charlie",
         "scoring": {"maxPoints": 1, "rubric": ["r"], "mustInclude": ["qux"]}},
        {"id": "l1", "domain": "logic", "prompt": "delta echo foxtrot golf hotel india",
         "scoring": {"maxPoints": 1, "rubric": ["r"]}},
        {"id": "c1", "domain": "coding", "prompt": "juliet kilo lima mike november oscar",
         "scoring": {"maxPoints": 1, "rubric": ["r"]}},
        {"id": "h1", "domain": "history", "prompt": "papa quebec romeo sierra tango uniform",
         "scoring": {"maxPoints": 1, "rubric": ["r"]}},
    ]}


def test_happy_path_writes_pack(tmp_path: Path):
    inp = tmp_path / "in.json"; inp.write_text(json.dumps(_four_domain_items()))
    (tmp_path / "agi-proof" / "hidden-reviewer-packs").mkdir(parents=True)
    (tmp_path / "agi-proof" / "hidden-reviewer-packs" / "schema.json").write_text(
        SCHEMA.read_text() if SCHEMA.exists() else _fallback_schema())
    corpus = tmp_path / "corpus"; corpus.mkdir()
    (corpus / "unrelated.md").write_text("completely different words here nothing overlaps xyz")
    out = tmp_path / "pack.json"
    r = _run(["--input", "in.json", "--schema", "agi-proof/hidden-reviewer-packs/schema.json",
              "--corpus", "corpus", "--out", "pack.json"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert out.exists() and (tmp_path / "pack.json.checksums.sha256").exists()
    pack = json.loads(out.read_text())
    assert pack["reviewer"]["status"] == "third-party"
    assert len(pack["cases"]) == 4


def test_fewer_than_four_domains_refused(tmp_path: Path):
    items = {"cases": _four_domain_items()["cases"][:2]}
    inp = tmp_path / "in.json"; inp.write_text(json.dumps(items))
    (tmp_path / "schema.json").write_text(_fallback_schema())
    (tmp_path / "corpus").mkdir()
    r = _run(["--input", "in.json", "--schema", "schema.json", "--corpus", "corpus",
              "--out", "pack.json"], tmp_path)
    assert r.returncode != 0
    assert "domains" in (r.stderr + r.stdout).lower()


def test_contamination_refuses_to_write(tmp_path: Path):
    items = _four_domain_items()
    inp = tmp_path / "in.json"; inp.write_text(json.dumps(items))
    (tmp_path / "schema.json").write_text(_fallback_schema())
    corpus = tmp_path / "corpus"; corpus.mkdir()
    # make a corpus doc that IS the first prompt -> Jaccard ~1.0
    (corpus / "leak.md").write_text(items["cases"][0]["prompt"] + " qux")
    out = tmp_path / "pack.json"
    r = _run(["--input", "in.json", "--schema", "schema.json", "--corpus", "corpus",
              "--out", "pack.json"], tmp_path)
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert not out.exists(), "pack must NOT be written on contamination"


def test_empty_corpus_fails_closed(tmp_path: Path):
    items = _four_domain_items()
    (tmp_path / "in.json").write_text(json.dumps(items))
    (tmp_path / "schema.json").write_text(_fallback_schema())
    (tmp_path / "corpus").mkdir()
    r = _run(["--input", "in.json", "--schema", "schema.json", "--corpus", "corpus",
              "--out", "pack.json"], tmp_path)
    assert r.returncode == 3
    assert "cannot prove decontamination" in (r.stderr + r.stdout)


def _fallback_schema() -> str:
    return json.dumps({
        "type": "object", "required": ["packId", "createdAt", "visibility", "cases"],
        "properties": {"cases": {"type": "array", "minItems": 1}},
    })


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
