#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""T10: nano-scale Per-Layer-Embedding probe — reference invariants + matched-compute probe."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.architecture import ple  # noqa: E402


def test_offline_invariants_pass() -> None:
    ok, detail = ple.offline_invariants()
    assert ok, detail["checks"]
    # the gradient check is the load-bearing correctness invariant
    assert detail["checks"]["embedding_gradcheck"] is True
    assert detail["checks"]["ple_off_equals_dense"] is True


def test_ple_adds_lookup_params_at_matched_matmul() -> None:
    from pretraining.nano.model import NanoLM
    dense = NanoLM(vocab=8, context=2, hidden=8, seed=0)
    p = ple.PLELM(vocab=8, context=2, hidden=8, seed=0, ple=True)
    assert p.num_params() == dense.num_params() + 8 * 8     # +V*h lookup params
    assert p.active_matmul_flops() == dense.h * dense.V     # identical compute-bound matmul


def test_probe_runs_and_reports_matched_compute() -> None:
    import tempfile
    # write to a throwaway path so running tests never dirties the committed artifact
    with tempfile.TemporaryDirectory() as td:
        r = ple.run(quick=True, out=Path(td) / "ple.json")
    assert r["matched_compute"] is True
    assert r["verdict"] in {"ple_better", "dense_better", "tie"}
    assert r["ple"]["total_params"] > r["dense"]["total_params"]


def main() -> int:
    test_offline_invariants_pass()
    test_ple_adds_lookup_params_at_matched_matmul()
    test_probe_runs_and_reports_matched_compute()
    print("test_ple_architecture: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
