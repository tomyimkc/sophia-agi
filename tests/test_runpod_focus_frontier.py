# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Focus-frontier RunPod launcher — no-cost paths only (arg parsing, remote script, payload).

Never creates a pod: only the pure, offline-safe surface is exercised.
"""
from __future__ import annotations

import json

from tools import runpod_focus_frontier as L


def test_offline_remote_script_runs_harness_not_a_fake_eval():
    args = L.parse_args(["--remote-mode", "offline"])
    script = L._remote_focus_script(args)
    assert "tools/run_focus_efficiency_frontier.py --write" in script
    assert "tools/run_focus_efficiency_frontier.py --check" in script
    # offline must not inject any live measured-eval language
    assert "LIVE mode" not in script


def test_live_remote_script_is_honest_about_model_gating():
    args = L.parse_args(["--remote-mode", "live"])
    script = L._remote_focus_script(args)
    assert "LIVE:" in script
    assert "model-gated" in script
    assert "no number is fabricated" in script.lower()
    # still runs the honest harnesses + the 3-arm eval
    assert "run_focus_efficiency_frontier.py --write" in script
    assert "run_focus_frontier_eval.py --write" in script


def test_default_eval_entrypoint_is_the_three_arm_eval():
    args = L.parse_args([])
    assert args.eval_entrypoint == "tools/run_focus_frontier_eval.py"
    script = L._remote_focus_script(L.parse_args(["--remote-mode", "offline"]))
    assert "tools/run_focus_frontier_eval.py --write" in script


def test_live_passes_model_when_supplied():
    args = L.parse_args(["--remote-mode", "live", "--model", "Qwen/Qwen2.5-7B-Instruct"])
    script = L._remote_focus_script(args)
    assert '--model "Qwen/Qwen2.5-7B-Instruct"' in script


def test_dry_run_builds_payload_without_secret(capsys):
    rc = L.main(["--dry-run", "--remote-mode", "live"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RunPod create payload" in out
    # The payload env must NOT carry a RUNPOD_API_KEY *value* (it is popped before print);
    # the dry-run pubkey placeholder proves no real key/pubkey was used.
    assert '"RUNPOD_API_KEY":' not in out
    assert "DRYRUN" in out


def test_refuses_pod_without_yes():
    # No --yes and no --dry-run/--local -> must refuse (exit 2), never create a pod.
    assert L.main(["--remote-mode", "live"]) == 2


def test_local_writes_nogo_report(tmp_path, monkeypatch):
    rc = L.main(["--local", "--remote-mode", "offline"])
    assert rc == 0
    from tools.run_focus_efficiency_frontier import REPORT
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    assert data["verdict"] == "NO-GO"
