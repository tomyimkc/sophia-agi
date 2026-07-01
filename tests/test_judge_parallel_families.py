#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cross-family judge concurrency must change ONLY wall-clock, never the verdicts.

judge_pilot_answers.py judges each family on its own box; running families concurrently
(distinct boxes) instead of sequentially reclaims the cross-box idle gap. This test pins the
invariant that matters: concurrent and sequential produce IDENTICAL per-item verdicts, and the
concurrency is correctly suppressed when two families share a box (which would only oversubscribe
one endpoint). Offline — uses the `mock` provider, no network, no GPU.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "judge_pilot_answers.py"


def _fixture(d: Path) -> Path:
    rows = [{"id": f"source_discipline/{i}", "task_family": "source_discipline",
             "base_answer": f"Base answer {i}", "adapter_answer": f"Adapter answer {i}",
             "reference": "gold", "prompt": "q"} for i in range(16)]
    p = d / "answers.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def _run(answers: Path, judges: str, raw: Path, out: Path, *extra: str) -> str:
    r = subprocess.run([sys.executable, str(TOOL), "--answers", str(answers),
                        "--judges", judges, "--forced-choice", "--seed", "1",
                        "--raw-out", str(raw), "--out", str(out), *extra],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    return r.stdout


def _verdicts(raw: Path) -> list:
    d = json.loads(raw.read_text(encoding="utf-8"))
    return [(it["id"], it["verdicts"]) for it in d["items"]]


def test_concurrent_equals_sequential() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        ans = _fixture(d)
        judges = "mock:a@http://box1:1/v1,mock:b@http://box2:2/v1"
        log_par = _run(ans, judges, d / "par-raw.json", d / "par.json")
        log_seq = _run(ans, judges, d / "seq-raw.json", d / "seq.json", "--no-parallel-families")
        assert "CONCURRENTLY" in log_par, "distinct boxes should judge concurrently by default"
        assert "CONCURRENTLY" not in log_seq, "--no-parallel-families must force sequential"
        assert _verdicts(d / "par-raw.json") == _verdicts(d / "seq-raw.json"), \
            "concurrency must not change any verdict"


def test_shared_box_suppresses_concurrency() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        ans = _fixture(d)
        judges = "mock:a@http://samebox:9/v1,mock:b@http://samebox:9/v1"
        log = _run(ans, judges, d / "raw.json", d / "out.json")
        assert "CONCURRENTLY" not in log, "families on the same box must NOT be parallelized"
        assert "share a box" in log


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} judge_parallel_families tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
