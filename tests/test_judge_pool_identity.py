#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Judge POOL routing must change ONLY which lane serves a request, NEVER the verdict.

The judge pool serves each judge FAMILY by MULTIPLE endpoint REPLICAS (lanes) so judge load
distributes instead of queueing on one box. The invariant that makes this safe: replicas of the
SAME model are the SAME family, so spreading a family's per-item requests across its lanes is
timing/routing ONLY — every per-item verdict is IDENTICAL to the single-endpoint path.

This test pins that. It runs tools/judge_pilot_answers.py twice on the same mock answers file:
  (1) single-endpoint per family (the unchanged default), and
  (2) with --judge-pool giving each family 3 mock REPLICAS (distinct @urls, same model),
and asserts the per-item per-judge verdicts are byte-identical. Offline — `mock` provider, no
network, no GPU. The mock's output depends on the model + prompt, NOT the base_url, so the two
runs MUST agree; if they ever diverge, the pool changed a verdict and this test fails.
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
             "reference": "gold", "prompt": "q"} for i in range(24)]
    p = d / "answers.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def _pool_config(d: Path) -> Path:
    """Each family = 3 mock REPLICAS (same model, distinct @urls). Two families -> 2-family gate.
    Replicas share the model so they key to the same family AND give the same verdict."""
    cfg = {
        "schema": "sophia.judge_pool.local.v1",
        "families": {
            "deepseek": {"replicas": [
                "openrouter:deepseek/deepseek-chat@http://ds1:1/v1",
                "openrouter:deepseek/deepseek-chat@http://ds2:2/v1",
                "openrouter:deepseek/deepseek-chat@http://ds3:3/v1",
            ]},
            "meta-llama": {"replicas": [
                "openrouter:meta-llama/llama-3.3-70b-instruct@http://ml1:1/v1",
                "openrouter:meta-llama/llama-3.3-70b-instruct@http://ml2:2/v1",
                "openrouter:meta-llama/llama-3.3-70b-instruct@http://ml3:3/v1",
            ]},
        },
    }
    p = d / "pool.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _run(answers: Path, judges: str, raw: Path, out: Path, *extra: str) -> str:
    # mock provider keys verdicts off the model; the @url only selects the (mock) endpoint. We use
    # openrouter:<vendor>/<model> so _family_key keys to the VENDOR (deepseek / meta-llama) and the
    # single-endpoint and pooled runs cover the SAME two families.
    env = {"SOPHIA_MODEL_PROVIDER": "mock"}
    import os
    full_env = {**os.environ, **env}
    r = subprocess.run([sys.executable, str(TOOL), "--answers", str(answers),
                        "--judges", judges, "--forced-choice", "--seed", "1",
                        "--raw-out", str(raw), "--out", str(out), *extra],
                       capture_output=True, text=True, cwd=str(ROOT), env=full_env)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _verdicts(raw: Path) -> list:
    d = json.loads(raw.read_text(encoding="utf-8"))
    return [(it["id"], it["verdicts"]) for it in d["items"]]


def test_pool_is_verdict_identical() -> None:
    """The KEY proof: single-endpoint vs 3-replica-per-family pool => IDENTICAL per-item verdicts."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        ans = _fixture(d)
        pool = _pool_config(d)
        # Single-endpoint: one spec per family (the unchanged default path).
        single_judges = ("openrouter:deepseek/deepseek-chat@http://ds1:1/v1,"
                         "openrouter:meta-llama/llama-3.3-70b-instruct@http://ml1:1/v1")
        log_single = _run(ans, single_judges, d / "single-raw.json", d / "single.json")
        # Pooled: SAME two families, 3 lanes each. The --judges flag still names the two families;
        # --judge-pool fans each family across its replica lanes.
        log_pool = _run(ans, single_judges, d / "pool-raw.json", d / "pool.json",
                        "--judge-pool", str(pool))
        assert "judge-pool ON" in log_pool, "pool run must announce pool routing"
        assert "judge-pool ON" not in log_single, "single run must not route through a pool"
        v_single = _verdicts(d / "single-raw.json")
        v_pool = _verdicts(d / "pool-raw.json")
        assert v_single == v_pool, (
            "judge pool changed a verdict! routing across replicas must be timing-only.\n"
            f"single={v_single[:3]}\npool={v_pool[:3]}")
        assert len(v_pool) == 24


def test_pool_rejects_under_two_families() -> None:
    """A pool collapsing to <2 families (one family on many lanes) must be REFUSED, not silently run."""
    import os
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        ans = _fixture(d)
        bad = {"schema": "sophia.judge_pool.local.v1",
               "families": {"deepseek": {"replicas": [
                   "openrouter:deepseek/deepseek-chat@http://ds1:1/v1",
                   "openrouter:deepseek/deepseek-chat@http://ds2:2/v1"]}}}
        badp = d / "bad.json"
        badp.write_text(json.dumps(bad), encoding="utf-8")
        r = subprocess.run(
            [sys.executable, str(TOOL), "--answers", str(ans),
             "--judges", "openrouter:deepseek/deepseek-chat@http://ds1:1/v1",
             "--forced-choice", "--seed", "1", "--raw-out", str(d / "r.json"),
             "--out", str(d / "o.json"), "--judge-pool", str(badp)],
            capture_output=True, text=True, cwd=str(ROOT),
            env={**os.environ, "SOPHIA_MODEL_PROVIDER": "mock"})
        assert r.returncode != 0, "a <2-family pool must be refused"
        assert "2" in (r.stderr + r.stdout), "error should mention the 2-family requirement"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} judge_pool_identity tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
