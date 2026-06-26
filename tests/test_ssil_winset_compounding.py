# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the win-set compounding driver (no torch, no GPU).

Exercises the FULL gate pipeline (win-set build, contamination re-check against the
held-out eval split, multi-seed aggregate, compounding_proof, z3 attestation) on
synthetic generations, and proves the negative control: a deliberately-contaminated
generation is REJECTED by the gate but ADMITTED by the ungated control.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_ssil_winset_compounding as drv  # noqa: E402


def _run(extra: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ssil.json"
        code = drv.main(["--mode", "mock", "--task", "code", "--generations", "4", "--out", str(out), *extra])
        assert code == 0, f"driver exited {code}"
        return json.loads(out.read_text(encoding="utf-8"))


def test_mock_driver_produces_monotone_gated_curve() -> None:
    proof = _run([])
    assert proof["mode"] == "mock"
    assert proof["canClaimAGI"] is False
    inv = proof["invariants"]
    assert inv["gated_curve_monotone_rising"], inv
    assert inv["converges_at_ceiling"], inv
    assert inv["win_set_eval_disjoint_honest_gens"], inv
    # every honest generation carries the real-data audit fields
    honest = [r for r in proof["gated"]["generations"] if not r["anyContaminated"]]
    assert honest and all("winSetSize" in r and "contaminationStatus" in r for r in honest)
    assert all((r["contaminationStatus"] or {}).get("clean") for r in honest)


def test_negative_control_gate_rejects_contaminated_gen() -> None:
    proof = _run(["--negative-control"])
    inv = proof["invariants"]
    assert inv["gate_rejects_contaminated_gen"], inv
    assert inv["negative_control_would_admit_it"], inv
    assert inv["gate_made_a_difference"], inv
    # the injected contaminated gen's win-set leak is DETECTED (clean=False) and gate-rejected
    last = proof["gated"]["generations"][-1]
    assert last["anyContaminated"] is True
    assert (last["contaminationStatus"] or {}).get("clean") is False
    assert last["gateVerdict"] == "rejected"
    assert proof["liveClaimStatus"].startswith("Open")


def test_win_set_excludes_heldout_eval_split() -> None:
    """Direct check: every honest generation's win set is disjoint from the eval split."""
    from provenance_bench import code_dataset
    from provenance_bench.dataset_guard import normalize

    data = code_dataset.build_code_rl_dataset(eval_frac=0.34, seed=0)
    eval_prompts = {normalize(r["prompt"]) for r in data["eval_rows"]}
    for g in range(1, 5):
        win = drv._win_set(data["train_rows"], g)
        # win set is train-split only
        assert all(normalize(r["prompt"]) not in eval_prompts for r in win)
        assert len(win) > 0


def test_live_mode_refuses_without_gpu() -> None:
    """--mode live is wired but not executable from a no-GPU session; it must refuse clearly."""
    try:
        drv.run("live", "code", 4, [0, 1, 2], min_delta=0.03, ci_k=1.0,
                base_after=0.0, negative_control=False, out=Path("/tmp/x.json"))
    except SystemExit as exc:
        msg = str(exc).lower()
        assert "not executed" in msg or "runpod" in msg, msg
        return
    raise AssertionError("--mode live should have refused with a SystemExit in a no-GPU session")


if __name__ == "__main__":
    for name in list(globals()):
        if name.startswith("test_"):
            globals()[name]()
    print("SSIL win-set compounding offline invariants PASS")
