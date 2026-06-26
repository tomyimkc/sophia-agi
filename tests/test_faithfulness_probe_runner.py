#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the faithfulness probe runner (tools/run_faithfulness_probe.py).

Exercises the CI-safe mock path end to end (report shape, honesty fields,
aggregation) and the --mode real fail-closed behavior. Does NOT require MLX.
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
    assert report["schema"] == "sophia.faithfulness_probe.v1"
    assert report["mode"] == "mock"
    assert report["candidateOnly"] is True
    assert report["validated"] is False
    assert "not proof of AGI" in report["boundary"]
    assert len(report["probes"]) == 3
    # mean flip-rate is a number in [0,1] (mock produces a synthetic value)
    assert report["meanFlipRate"] is None or 0.0 <= report["meanFlipRate"] <= 1.0
    # each probe carries the contrast hint for human interpretation
    hints = {p["hint"] for p in report["probes"]}
    assert hints == {"load-bearing", "hedged", "post-hoc"}
    # artifact written and matches the returned report
    assert out.exists()
    assert json.loads(out.read_text()) == report


def test_real_mode_fail_closes_without_mlx() -> None:
    """--mode real must refuse with a clear error when mlx_lm is unavailable.
    Run as a subprocess so the import check is honest (not patched)."""
    import subprocess
    # mlx_lm is absent on CI / non-Apple-Silicon; if it IS present this test is a no-op.
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
    """The report must carry the honest interpretation: HIGH = more faithful, but
    not proof; LOW could be post-hoc OR robustly-correct."""
    from tools.run_faithfulness_probe import run
    report = run(mode="mock", out=None)
    interp = report["interpretation"].lower()
    assert "load-bearing" in interp
    assert "not proof" in interp


def main() -> int:
    test_mock_run_produces_honest_report()
    print(f"ok {test_mock_run_produces_honest_report.__name__}")
    test_real_mode_fail_closes_without_mlx()
    print(f"ok {test_real_mode_fail_closes_without_mlx.__name__}")
    test_interpretation_string_is_present()
    print(f"ok {test_interpretation_string_is_present.__name__}")
    print("PASS faithfulness probe runner tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
