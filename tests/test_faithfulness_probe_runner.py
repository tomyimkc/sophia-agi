#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the faithfulness probe runner (tools/run_faithfulness_probe.py).

v4: the probe-POWER upgrade the v3 findingScope called for — 30 binary-gold
probes (vs 16), >=4-sentence CoTs, 6 reasoning-only perturbs (vs 3, so each probe
yields nAttempted>=3), and a bootstrap CI + sign test on top of Cohen's d. These
tests lock in that the probe produces a LARGE effect AND a CI that excludes 0 on
the mock scorer (the precondition for the probe to be able to detect a real
adapter signal at all — a probe that can't separate a known-strong synthetic
signal has no chance on a real adapter).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_mock_run_produces_v4_report() -> None:
    from tools.run_faithfulness_probe import run
    out = Path(tempfile.mkdtemp()) / "fp.json"
    report = run(mode="mock", out=out)
    assert report["schema"] == "sophia.faithfulness_probe.v4"
    assert report["mode"] == "mock"
    assert report["candidateOnly"] is True
    assert report["validated"] is False
    assert "not proof of AGI" in report["boundary"]
    # v4 shape: 30 binary-gold probes, 6 perturbs, Cohen's d + bootstrap CI + sign test
    assert report["nProbes"] == 30
    assert report["nLoadBearing"] == 15 and report["nPostHoc"] == 15
    assert report["nPerturbs"] == 6
    assert "cohensD" in report and "effectVerdict" in report
    assert "bootstrapCI" in report and "signTest" in report
    for hint in ("load-bearing", "post-hoc"):
        h = report["perHint"][hint]
        assert {"mean", "std", "n"} <= set(h), h
    # every probe is binary gold (v2's ill-posed "possibly" is gone)
    assert all(p["gold"] in ("yes", "no") for p in report["probes"])
    # stdDrop present per probe (v3 addition)
    assert all("stdDrop" in p for p in report["probes"])
    # v4 power requirement: each probe yields nAttempted>=3 (the v3 limit was <=2)
    assert all(p["nAttempted"] >= 3 for p in report["probes"]), (
        [(p["id"], p["nAttempted"]) for p in report["probes"] if p["nAttempted"] < 3]
    )
    assert report["meanAttempted"] >= 3.0
    # artifact written and matches the returned report
    assert out.exists()
    assert json.loads(out.read_text()) == report


def test_v4_probe_shows_large_effect_on_mock() -> None:
    """THE regression test for the probe (was test_v3_probe_shows_large_effect_on_mock).
    The mock scorer embeds a strong signal (named support tokens raise the gold
    logprob; filler does not). A probe that CANNOT produce a large Cohen's d AND a
    bootstrap CI excluding 0 here has no chance of detecting a real adapter effect
    — so this is a probe-power precondition, not an adapter claim. If this fails,
    the probe design is broken (not the model)."""
    from tools.run_faithfulness_probe import run
    report = run(mode="mock", out=None)
    d = report["cohensD"]
    assert d is not None, "Cohen's d undefined — probe produced no variance"
    assert abs(d) >= 0.8, (
        f"v4 probe does not reach large effect on the mock (d={d}); "
        f"perHint={report['perHint']}. The probe cannot detect a signal it can't "
        f"see in a synthetic case with a known-strong signal."
    )
    # bootstrap CI on the mean difference must exclude 0 (direction reliable)
    boot = report["bootstrapCI"]
    assert boot is not None and boot["excludesZero"], boot
    # per-probe sign test must be lopsidedly positive (load-bearing drops more)
    sign = report["signTest"]
    assert sign is not None and sign["nPos"] > sign["nNeg"], sign
    assert sign["pValue"] < 0.05, sign
    # direction: load-bearing drops MORE than post-hoc
    lb = report["perHint"]["load-bearing"]["mean"]
    ph = report["perHint"]["post-hoc"]["mean"]
    assert lb > ph, (lb, ph)
    assert "large effect" in report["effectVerdict"]


def test_v1_artifact_is_on_record_as_falsified() -> None:
    """The falsified v1 artifact must remain committed (not hidden) with an
    honest interpretation. Hiding a falsified probe would be the exact
    overclaim the layer exists to prevent."""
    v1 = ROOT / "agi-proof" / "verified-traces" / "faithfulness-probe.v1-FALSIFIED.public-report.json"
    assert v1.exists(), "v1 FALSIFIED artifact must remain on record"
    art = json.loads(v1.read_text())
    assert "FALSIFIED" in art.get("version", "").upper() or "falsified" in art.get("interpretation", "").lower()
    assert "falsificationNote" in art


def test_real_mode_fail_closes_without_mlx() -> None:
    """--mode real must refuse with a clear error when mlx_lm is unavailable."""
    import subprocess
    try:
        import mlx_lm  # noqa: F401
        return  # Apple Silicon with mlx installed -> skip
    except ImportError:
        pass
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_faithfulness_probe.py"), "--mode", "real"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "mlx-lm" in r.stdout or "mlx-lm" in r.stderr


def test_interpretation_carries_honest_caveat() -> None:
    """The v3 interpretation must state that small |d| is inconclusive at this
    power, NOT by itself 'decorative CoT' — the framing fix from the v2 reframe."""
    from tools.run_faithfulness_probe import run
    report = run(mode="mock", out=None)
    interp = report["interpretation"].lower()
    assert "not by itself" in interp or "not proof" in interp


def main() -> int:
    test_mock_run_produces_v4_report()
    print(f"ok {test_mock_run_produces_v4_report.__name__}")
    test_v4_probe_shows_large_effect_on_mock()
    print(f"ok {test_v4_probe_shows_large_effect_on_mock.__name__}")
    test_v1_artifact_is_on_record_as_falsified()
    print(f"ok {test_v1_artifact_is_on_record_as_falsified.__name__}")
    test_real_mode_fail_closes_without_mlx()
    print(f"ok {test_real_mode_fail_closes_without_mlx.__name__}")
    test_interpretation_carries_honest_caveat()
    print(f"ok {test_interpretation_carries_honest_caveat.__name__}")
    print("PASS faithfulness probe runner tests (v4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
