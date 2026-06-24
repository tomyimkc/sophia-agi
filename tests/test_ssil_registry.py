#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_registry import Registry  # noqa: E402

SPEC_A = {"min_sources": 1, "min_quality": 0.4}
SPEC_B = {"min_sources": 2, "min_quality": 0.5}


def _reg() -> Registry:
    r = Registry(path=None, canonical_n=2)
    r.record(entry_id="a0", round_idx=0, spec=SPEC_A, metric=0.825, parent=None, gate_verdicts={})
    return r


def test_canonical_requires_n_replications() -> None:
    r = _reg()
    assert r.is_canonical(SPEC_A) is False  # 1 < N
    r.record(entry_id="a1", round_idx=1, spec=SPEC_A, metric=0.825, parent=None, gate_verdicts={})
    assert r.is_canonical(SPEC_A) is True   # 2 >= N
    assert r.canonical_best()["spec"] == SPEC_A


def test_compounding_best_rises() -> None:
    r = _reg()
    r.record(entry_id="a1", round_idx=1, spec=SPEC_A, metric=0.825, parent=None, gate_verdicts={})
    r.record(entry_id="b0", round_idx=2, spec=SPEC_B, metric=0.875, parent="a1", gate_verdicts={})
    r.record(entry_id="b1", round_idx=3, spec=SPEC_B, metric=0.875, parent="a1", gate_verdicts={})
    assert r.canonical_best()["spec"] == SPEC_B  # higher metric, canonical


def test_counterfactual_revert() -> None:
    r = _reg()
    r.record(entry_id="a1", round_idx=1, spec=SPEC_A, metric=0.825, parent=None, gate_verdicts={})
    r.record(entry_id="b0", round_idx=2, spec=SPEC_B, metric=0.875, parent="a1", gate_verdicts={})
    r.record(entry_id="b1", round_idx=3, spec=SPEC_B, metric=0.875, parent="a1", gate_verdicts={})
    # Revert the B improvement -> canonical best falls back to A.
    cf = r.counterfactual_best({"b0", "b1"})
    assert cf is not None and cf["spec"] == SPEC_A


def test_file_backed_roundtrip(tmp_path_factory=None) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "reg.jsonl"
        r1 = Registry(path=p, canonical_n=2)
        r1.record(entry_id="a0", round_idx=0, spec=SPEC_A, metric=0.825, parent=None, gate_verdicts={})
        r1.record(entry_id="a1", round_idx=1, spec=SPEC_A, metric=0.825, parent=None, gate_verdicts={})
        r2 = Registry(path=p, canonical_n=2)  # reload from disk
        assert r2.is_canonical(SPEC_A) is True
        assert len(r2.entries()) == 2


def main() -> int:
    test_canonical_requires_n_replications()
    test_compounding_best_rises()
    test_counterfactual_revert()
    test_file_backed_roundtrip()
    print("test_ssil_registry: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
