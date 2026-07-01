#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the verifier-gated real-time grounding loop.

Offline only (FixtureFactBackend + neutral math/URL claims). Verifies the
fail-closed admission logic: a claim is ingested only when the fact-check verdict,
the conformal decision, and the decontam/valid-time gates ALL pass.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import realtime_grounding as rg  # noqa: E402
from agent import streaming_decontam as sd  # noqa: E402
from agent.live_sources import FixtureFactBackend  # noqa: E402

FIXTURE = ROOT / "data" / "realtime" / "fixtures_v1.json"
AS_OF = "2026-07-01"
CUTOFF = "2026-01-01"


def _backend() -> FixtureFactBackend:
    return FixtureFactBackend.from_file(FIXTURE)


def _ingest(claim: str, **kw):
    kw.setdefault("backend", _backend())
    kw.setdefault("as_of", AS_OF)
    kw.setdefault("eval_cutoff", CUTOFF)
    kw.setdefault("eval_prompts", set())
    kw.setdefault("source_timestamp", "2025-06-01")
    return rg.ingest_one(claim, **kw)


def test_math_accepted_is_ingested() -> None:
    r = _ingest("2 + 2 = 4")
    assert r.verdict == "accepted", r.factCheck
    assert r.ingestState == "ingested", r.reason
    assert r.confidence == 1.0
    assert r.nonconformity == 0.0


def test_math_contradiction_is_rejected() -> None:
    r = _ingest("2 + 2 = 5")
    assert r.verdict == "rejected"
    assert r.ingestState == "rejected"


def test_url_accepted_is_ingested() -> None:
    r = _ingest("See https://example.org/spec-8 for details.")
    assert r.verdict == "accepted", r.factCheck
    assert r.ingestState == "ingested", r.reason


def test_temporal_decontam_vetoes_future_source() -> None:
    r = _ingest("See https://example.org/spec-9 for details.", source_timestamp="2027-03-01")
    assert r.verdict == "accepted"  # fact-check passes
    assert r.ingestState == "quarantined"  # but the temporal gate vetoes
    assert "temporal-decontam" in r.reason


def test_valid_time_vetoes_expired_interval() -> None:
    r = _ingest("3 + 3 = 6", valid_from="2020-01-01", valid_until="2021-01-01")
    assert r.verdict == "accepted"
    assert r.ingestState == "quarantined"
    assert "valid-time" in r.reason


def test_held_claim_is_quarantined() -> None:
    r = _ingest("The service tier ships a new caching layer in build nine.")
    assert r.verdict == "held"
    assert r.ingestState == "quarantined"


def test_content_decontam_vetoes_eval_duplicate() -> None:
    claim = "See https://example.org/spec-8 for details."
    r = _ingest(claim, eval_prompts={sd.normalize(claim)})
    assert r.verdict == "accepted"
    assert r.ingestState == "quarantined"
    assert "content-decontam" in r.reason


def test_fail_closed_without_backend() -> None:
    r = _ingest("The service tier ships a new caching layer in build nine.", backend=None)
    assert r.ingestState != "ingested"


def test_conformal_policy_abstains_on_held() -> None:
    from agent.conformal_gate import fit_conformal_policy
    policy = fit_conformal_policy([{"nonconformity": i / 20, "correct": True} for i in range(6)], alpha=0.1)
    r = _ingest("The service tier ships a new caching layer in build nine.", policy=policy)
    assert r.conformal["verdict"] == "abstain"  # nonconformity 0.6 exceeds any calibrated threshold
    assert r.ingestState == "quarantined"


def test_run_grounding_persists_only_ingested_and_dedups() -> None:
    claims = [
        {"claim": "2 + 2 = 4", "sourceTimestamp": "2025-06-01"},
        {"claim": "2 + 2 = 5", "sourceTimestamp": "2025-06-01"},
        {"claim": "See https://example.org/spec-8 for details.", "sourceTimestamp": "2025-06-01"},
    ]
    with tempfile.TemporaryDirectory() as d:
        store = Path(d) / "belief_store.jsonl"
        rep = rg.run_grounding(claims, backend=_backend(), as_of=AS_OF, eval_cutoff=CUTOFF, root=ROOT, store_path=store)
        assert rep["counts"].get("ingested") == 2, rep["counts"]
        assert rep["counts"].get("rejected") == 1
        assert rep["nWrittenToStore"] == 2
        lines = [ln for ln in store.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2
        # re-run: dedup by claimId -> nothing new written
        rep2 = rg.run_grounding(claims, backend=_backend(), as_of=AS_OF, eval_cutoff=CUTOFF, root=ROOT, store_path=store)
        assert rep2["nWrittenToStore"] == 0


def test_mark_stale_flags_lapsed_beliefs() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = Path(d) / "belief_store.jsonl"
        row = _ingest("2 + 2 = 4", valid_until="2021-01-01")  # expired interval, but force-ingest by clearing gate
        # ingest_one would quarantine an expired one; write it as ingested to test the daemon directly.
        row.ingestState = "ingested"
        rg.append_belief_rows(store, [row])
        out = rg.mark_stale(store, "2026-07-01")
        assert out["nStale"] == 1
        assert '"stale"' in store.read_text(encoding="utf-8")


def test_claim_id_is_stable() -> None:
    assert rg.claim_id("2 + 2 = 4") == rg.claim_id("2 + 2 = 4")


def main() -> int:
    test_math_accepted_is_ingested()
    test_math_contradiction_is_rejected()
    test_url_accepted_is_ingested()
    test_temporal_decontam_vetoes_future_source()
    test_valid_time_vetoes_expired_interval()
    test_held_claim_is_quarantined()
    test_content_decontam_vetoes_eval_duplicate()
    test_fail_closed_without_backend()
    test_conformal_policy_abstains_on_held()
    test_run_grounding_persists_only_ingested_and_dedups()
    test_mark_stale_flags_lapsed_beliefs()
    test_claim_id_is_stable()
    print("test_realtime_grounding: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
