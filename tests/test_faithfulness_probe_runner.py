#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the faithfulness probe runner (tools/run_faithfulness_probe.py).

v3: the v2 boolean `discriminates` (under-powered at n=2, one ill-posed gold) is
replaced by a Cohen's d effect size over 16 binary-gold probes. These tests lock
in that the probe produces a LARGE effect on the mock scorer (the precondition
for the probe to be able to detect a real adapter signal at all).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_mock_run_produces_v3_report() -> None:
    from tools.run_faithfulness_probe import run
    out = Path(tempfile.mkdtemp()) / "fp.json"
    report = run(mode="mock", out=out)
    assert report["schema"] == "sophia.faithfulness_probe.v3"
    assert report["mode"] == "mock"
    assert report["candidateOnly"] is True
    assert report["validated"] is False
    assert "not proof of AGI" in report["boundary"]
    # v3 shape: 16 binary-gold probes, Cohen's d, per-hint mean+std+n
    assert report["nProbes"] == 16
    assert report["nLoadBearing"] == 8 and report["nPostHoc"] == 8
    assert "cohensD" in report and "effectVerdict" in report
    for hint in ("load-bearing", "post-hoc"):
        h = report["perHint"][hint]
        assert {"mean", "std", "n"} <= set(h), h
    # every probe is binary gold (v2's ill-posed "possibly" is gone)
    assert all(p["gold"] in ("yes", "no") for p in report["probes"])
    # stdDrop present per probe (v3 addition)
    assert all("stdDrop" in p for p in report["probes"])
    # artifact written and matches the returned report
    assert out.exists()
    assert json.loads(out.read_text()) == report


def test_v3_probe_shows_large_effect_on_mock() -> None:
    """THE regression test for v3. The mock scorer embeds a strong signal
    (named support tokens raise the gold logprob; filler does not). A probe that
    CANNOT produce a large Cohen's d here has no chance of detecting a real
    adapter effect — so this is a probe-power precondition, not an adapter claim.
    If this fails, the probe design is broken (not the model)."""
    from tools.run_faithfulness_probe import run
    report = run(mode="mock", out=None)
    d = report["cohensD"]
    assert d is not None, "Cohen's d undefined — probe produced no variance"
    assert abs(d) >= 0.8, (
        f"v3 probe does not reach large effect on the mock (d={d}); "
        f"perHint={report['perHint']}. The probe cannot detect a signal it can't "
        f"see in a synthetic case with a known-strong signal."
    )
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
    test_mock_run_produces_v3_report()
    print(f"ok {test_mock_run_produces_v3_report.__name__}")
    test_v3_probe_shows_large_effect_on_mock()
    print(f"ok {test_v3_probe_shows_large_effect_on_mock.__name__}")
    test_v1_artifact_is_on_record_as_falsified()
    print(f"ok {test_v1_artifact_is_on_record_as_falsified.__name__}")
    test_real_mode_fail_closes_without_mlx()
    print(f"ok {test_real_mode_fail_closes_without_mlx.__name__}")
    test_interpretation_carries_honest_caveat()
    print(f"ok {test_interpretation_carries_honest_caveat.__name__}")
    print("PASS faithfulness probe runner tests (v3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
