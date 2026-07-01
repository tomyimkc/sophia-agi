# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate the realtime C1 Phase-1 DATED-QA seed pack + its PENDING framing.

Deterministic — no network, no model. Locks:
  (a) the seed is well-formed (schema fields, unique ids, valid labels) and carries the
      dated fields (sourceTimestamp/validFrom) the Phase-1 temporal machinery needs;
  (b) the seed is temporally CLEAN vs the 2026-07-01 cutoff and valid at as_of (so its
      labels are interpretable — no mislabeled leaked/stale items hidden in the pack);
  (c) the temporal-leakage + stale-fact REJECTION paths are exercised directly on
      streaming_decontam (the machinery the powered pack will lean on);
  (d) the seed is honestly framed: a SEED (N << 393), PENDING/not_run, candidate-only.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent.streaming_decontam as sd  # noqa: E402
from agent.realtime_benchmark import load_pack  # noqa: E402

SEED = ROOT / "eval" / "fact_check" / "phase1_dated_seed_v1.jsonl"
PENDING = ROOT / "data" / "realtime" / "benchmark" / "phase1-online.PENDING.public-report.json"
CUTOFF = "2026-07-01"
AS_OF = "2026-07-01"
_ROWS = load_pack(SEED)


def test_seed_is_well_formed() -> None:
    assert len(_ROWS) == 28
    ids = [r["id"] for r in _ROWS]
    assert len(ids) == len(set(ids)), "ids must be unique"
    for r in _ROWS:
        assert r["label"] in {"true", "false", "unknowable"}
        assert r["claim"].strip()
        assert r.get("sourceTimestamp"), f"{r['id']} missing sourceTimestamp"
        assert r.get("validFrom"), f"{r['id']} missing validFrom"


def test_seed_label_mix_is_balanced() -> None:
    c = Counter(r["label"] for r in _ROWS)
    assert c["true"] >= 10 and c["false"] >= 8 and c["unknowable"] >= 4, c


def test_seed_is_temporally_clean_and_valid_at_as_of() -> None:
    """Every seed item's source predates the cutoff and is valid at as_of (interpretable labels)."""
    for r in _ROWS:
        t = sd.temporal_decontam(r.get("sourceTimestamp", ""), CUTOFF)
        v = sd.valid_time(r.get("validFrom", ""), r.get("validUntil", ""), AS_OF)
        assert t["ok"], f"{r['id']} temporal_decontam failed: {t}"
        assert v["ok"], f"{r['id']} valid_time failed: {v}"


def test_temporal_and_stale_rejection_paths_exist() -> None:
    """The machinery the powered pack relies on: post-cutoff source and stale fact are rejected."""
    assert sd.temporal_decontam("2026-08-01", CUTOFF)["ok"] is False   # source postdates cutoff
    assert sd.temporal_decontam("", CUTOFF)["ok"] is False             # unparseable -> fail-closed
    assert sd.valid_time("2000-01-01", "2010-12-31", AS_OF)["ok"] is False  # expired before as_of


def test_pending_report_is_not_run_and_honest() -> None:
    rep = json.loads(PENDING.read_text(encoding="utf-8"))
    assert rep["status"] == "not_run"
    assert rep["go"] is False and rep["canClaimAGI"] is False
    assert rep["results"] is None
    assert rep["poweredTarget"]["requiredNForMDE_0_10"] == 393
    assert len(_ROWS) < rep["poweredTarget"]["requiredNForMDE_0_10"]  # seed is not powered
    assert rep["ledgerRef"] == "realtime-grounding-loop-candidate-only-2026-07-01"
