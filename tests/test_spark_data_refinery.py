# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline, deterministic invariants for the Spark gate-filtered data refinery (P2).

No GPU, no torch. The intrinsic-gate rule is the load-bearing one: the gate must be
called WITHOUT a question (Feasibility §4) — these tests assert that contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import spark_data_refinery as sdr  # noqa: E402


# --------------------------------------------------------------------------- #
# Mock teacher: deterministic + seeded
# --------------------------------------------------------------------------- #

def test_mock_teacher_deterministic() -> None:
    seeds = [{"id": "b", "prompt": "Q b"}, {"id": "a", "prompt": "Q a"}]
    first = sdr.mock_teacher(seeds, seed=7)
    second = sdr.mock_teacher(seeds, seed=7)
    assert [c.text for c in first] == [c.text for c in second]
    # Sorted by id for stable ordering regardless of input order.
    assert [c.seed_id for c in first] == ["a", "b"]
    # A different seed yields different (still deterministic) text.
    third = sdr.mock_teacher(seeds, seed=8)
    assert [c.text for c in third] != [c.text for c in first]


def test_mock_teacher_offline_no_torch() -> None:
    # The mock path must not *trigger* a torch/transformers import. Assert on the DELTA,
    # not global absence — in the full suite another test may have already loaded torch,
    # which says nothing about whether mock_teacher imports it.
    before = set(sys.modules)
    sdr.mock_teacher([{"id": "x", "prompt": "Q"}], seed=0)
    newly = set(sys.modules) - before
    assert not any(m == "torch" or m.startswith("torch.") for m in newly)
    assert not any(m == "transformers" or m.startswith("transformers.") for m in newly)


# --------------------------------------------------------------------------- #
# Gate filtering: drop violations, keep clean
# --------------------------------------------------------------------------- #

def test_fabricated_candidate_dropped_clean_kept() -> None:
    """A fabricated-citation / false-arithmetic candidate is DROPPED; clean is KEPT.

    Uses the REAL gate (no stub) so this is a true end-to-end invariant.
    """
    seeds = [
        {"id": "clean", "prompt": "Q clean"},  # mock emits a gate-passing answer
        {"id": "arith", "prompt": "Q arith",
         "inject": "source discipline 中文. The figure: 100000 / 5000 = 25 months."},
        {"id": "legal", "prompt": "Q legal",
         "inject": "source discipline 中文. Under Cap. 99999 s.42 of the Fictional Ordinance you may act."},
    ]
    result = sdr.refine(seeds, teacher="mock")
    assert result.candidates == 3
    assert result.dropped == 2  # both fabricated candidates dropped
    assert result.kept == 1
    kept_ids = {row["metadata"]["seedId"] for row in result.rows}
    assert kept_ids == {"clean"}


def test_drops_surfaced_never_silent() -> None:
    seeds = [{"id": "bad", "prompt": "Q",
              "inject": "source discipline 中文. 2 + 2 = 5."}]
    result = sdr.refine(seeds, teacher="mock")
    assert result.dropped == 1
    assert result.kept == 0
    assert result.rows == []


# --------------------------------------------------------------------------- #
# The intrinsic-gate rule: gate called WITHOUT a question
# --------------------------------------------------------------------------- #

def test_real_gate_called_without_question(monkeypatch) -> None:
    """Production default uses the real gate, and calls it intrinsically (no question)."""
    calls: list[dict] = []

    def spy_check_response(text, **kwargs):
        calls.append(kwargs)
        return {"violations": []}

    import agent.gate as gate_mod
    monkeypatch.setattr(gate_mod, "check_response", spy_check_response)

    # gate=None -> production path resolves _real_gate -> agent.gate.check_response.
    result = sdr.refine([{"id": "x", "prompt": "a question?"}], teacher="mock", gate=None)
    assert result.kept == 1
    assert calls, "real gate was not invoked"
    for kw in calls:
        assert "question" not in kw, f"intrinsic rule violated: question passed: {kw}"
        assert kw.get("mode") == "advisor"


def test_default_gate_is_real_gate() -> None:
    # Documented contract: the production default is the repo's intrinsic gate.
    assert sdr._real_gate.__module__ == "tools.spark_data_refinery"
    # And it imports the real check_response (no stub baked in).
    src = Path(sdr.__file__).read_text(encoding="utf-8")
    assert "from agent.gate import check_response" in src


# --------------------------------------------------------------------------- #
# Provenance annotations
# --------------------------------------------------------------------------- #

def test_rows_carry_provenance() -> None:
    result = sdr.refine([{"id": "s1", "prompt": "Q"}], teacher="mock")
    assert result.kept == 1
    meta = result.rows[0]["metadata"]
    assert meta["sparkIteration"] is True
    assert meta["registeredResult"] is False
    assert meta["teacher"] == "mock"
    assert meta["gatePassed"] is True
    assert meta["gateIntrinsic"] is True
    assert meta["gateMode"] == "advisor"
    assert meta["source"] == "spark-data-refinery"
    # SFT/council-trace shape: system + user + assistant.
    roles = [m["role"] for m in result.rows[0]["messages"]]
    assert roles == ["system", "user", "assistant"]


# --------------------------------------------------------------------------- #
# Fail-closed: no gate available -> refuse to emit
# --------------------------------------------------------------------------- #

def test_fail_closed_when_gate_unavailable(monkeypatch) -> None:
    def boom(text):
        raise ImportError("no agent.gate")

    monkeypatch.setattr(sdr, "_real_gate", boom)
    with pytest.raises(RuntimeError, match="fail-closed"):
        sdr.refine([{"id": "x", "prompt": "Q"}], teacher="mock", gate=None)


def test_injected_stub_gate_matches_contract() -> None:
    """Gate is injectable; a stub mimicking check_response's contract works."""
    def stub_gate(text: str) -> list:
        return ["forced violation"] if "BAD" in text else []

    seeds = [{"id": "ok", "prompt": "Q", "inject": "fine 中文 source discipline"},
             {"id": "no", "prompt": "Q", "inject": "BAD 中文 source discipline"}]
    result = sdr.refine(seeds, teacher="mock", gate=stub_gate)
    assert result.kept == 1
    assert result.dropped == 1


# --------------------------------------------------------------------------- #
# local teacher hook never required offline
# --------------------------------------------------------------------------- #

def test_local_teacher_is_a_hook_not_required_offline() -> None:
    with pytest.raises(NotImplementedError):
        sdr.local_teacher([{"id": "x", "prompt": "Q"}])


# --------------------------------------------------------------------------- #
# CLI dry-run
# --------------------------------------------------------------------------- #

def test_cli_dry_run_exit_zero_and_counts(capsys) -> None:
    rc = sdr.main(["--dry-run"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dryRun"] is True
    assert out["out"] is None
    assert out["candidates"] >= 1
    assert out["kept"] + out["dropped"] == out["candidates"]
    assert out["teacher"] == "mock"


def test_cli_writes_jsonl(tmp_path, capsys) -> None:
    out_path = tmp_path / "refined.jsonl"
    rc = sdr.main(["--teacher", "mock", "--out", str(out_path)])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["out"] == str(out_path)
    lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == summary["kept"]
    for row in lines:
        assert row["metadata"]["sparkIteration"] is True
        assert row["metadata"]["registeredResult"] is False
