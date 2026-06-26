#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the faithfulness probe runner (tools/run_faithfulness_probe.py).

v2: the probe was FALSIFIED in v1 (uniform 0.5 flip-rate; conflated answer-token
removal with reasoning perturbation). v2 uses gold-logprob drop + reasoning-only
perturbs so the categories separate. These tests lock in that discrimination.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_mock_run_produces_honest_report() -> None:
    from tools.run_faithfulness_probe import run
    out = Path(tempfile.mkdtemp()) / "fp.json"
    report = run(mode="mock", out=out)
    assert report["schema"] == "sophia.faithfulness_probe.v2"
    assert report["mode"] == "mock"
    assert report["candidateOnly"] is True
    assert report["validated"] is False
    assert "not proof of AGI" in report["boundary"]
    assert len(report["probes"]) == 3
    # v2 fields: meanDrop (not meanFlipRate), discriminates flag, perHint breakdown
    assert "meanDrop" in report
    assert "discriminates" in report
    assert "perHint" in report
    # each probe carries the gold answer + contrast hint
    hints = {p["hint"] for p in report["probes"]}
    assert hints == {"load-bearing", "hedged", "post-hoc"}
    assert all("gold" in p and "meanDrop" in p for p in report["probes"])
    # artifact written and matches the returned report
    assert out.exists()
    assert json.loads(out.read_text()) == report


def test_v2_probe_discriminates_load_bearing_from_post_hoc() -> None:
    """THE regression test for the v2 fix. v1 returned uniform 0.5 across all
    categories (falsified). v2 must SEPARATE them: load-bearing CoT -> larger
    gold-logprob drop than post-hoc CoT. If this fails, the probe still does not
    measure faithfulness — the falsification stands."""
    from tools.run_faithfulness_probe import run
    report = run(mode="mock", out=None)
    assert report["discriminates"] is True, (
        f"v2 probe does not discriminate: perHint={report['perHint']}. "
        f"load-bearing must drop more than post-hoc."
    )
    ph = report["perHint"]
    assert ph["load-bearing"] > ph["post-hoc"], ph
    # post-hoc should be ~0 (decoration, no support to remove)
    assert ph["post-hoc"] == 0.0, ph
    # load-bearing should be clearly positive
    assert ph["load-bearing"] > 0.0, ph


def test_v1_artifact_is_on_record_as_falsified() -> None:
    """The falsified v1 artifact must remain committed (not hidden) with an
    honest interpretation. Hiding a falsified probe would be the exact
    overclaim the layer exists to prevent."""
    v1 = ROOT / "agi-proof" / "verified-traces" / "faithfulness-probe.v1-FALSIFIED.public-report.json"
    assert v1.exists(), "v1 FALSIFIED artifact must remain on record"
    art = json.loads(v1.read_text())
    assert "FALSIFIED" in art.get("version", "").upper() or "falsified" in art.get("interpretation", "").lower()
    assert "falsificationNote" in art, "must carry the falsification diagnosis"


def test_real_mode_fail_closes_without_mlx() -> None:
    """--mode real must refuse with a clear error when mlx_lm is unavailable.
    Run as a subprocess so the import check is honest (not patched)."""
    import subprocess
    try:
        import mlx_lm  # noqa: F401
        return  # Apple Silicon with mlx installed -> skip (real mode would work)
    except ImportError:
        pass
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_faithfulness_probe.py"), "--mode", "real"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "mlx-lm" in r.stdout or "mlx-lm" in r.stderr


def test_interpretation_string_is_present() -> None:
    """The report must carry the honest interpretation: large drop = more faithful,
    but not proof; ~0 could be post-hoc OR robustly-certain."""
    from tools.run_faithfulness_probe import run
    report = run(mode="mock", out=None)
    interp = report["interpretation"].lower()
    assert "load-bearing" in interp or "faithful" in interp
    assert "not proof" in interp


def main() -> int:
    test_mock_run_produces_honest_report()
    print(f"ok {test_mock_run_produces_honest_report.__name__}")
    test_v2_probe_discriminates_load_bearing_from_post_hoc()
    print(f"ok {test_v2_probe_discriminates_load_bearing_from_post_hoc.__name__}")
    test_v1_artifact_is_on_record_as_falsified()
    print(f"ok {test_v1_artifact_is_on_record_as_falsified.__name__}")
    test_real_mode_fail_closes_without_mlx()
    print(f"ok {test_real_mode_fail_closes_without_mlx.__name__}")
    test_interpretation_string_is_present()
    print(f"ok {test_interpretation_string_is_present.__name__}")
    print("PASS faithfulness probe runner tests (v2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
