#!/usr/bin/env python3
"""Sophia Skills layer: registration, fail-closed behaviour, in-process MCP bridge.

Deterministic and offline — the bridge defaults to in-process `sophia_mcp.tools_impl`
(no network, no `mcp` package, no `requests`).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import skills  # noqa: E402
from skills.core import run_skill  # noqa: E402

EXPECTED = {
    "provenance_fact_check", "source_discipline_enforce", "conscience_abstain",
    "moral_parliament_decide", "claim_verify_and_record", "belief_revision_explore",
    "wiki_grounded_answer", "moral_public_standard_review", "deception_scan",
    "contradiction_audit", "council_adjudicate", "self_extend_probe",
}


def test_all_skills_registered() -> None:
    assert EXPECTED <= set(skills.SKILLS), EXPECTED - set(skills.SKILLS)
    meta = skills.list_skills()
    assert all(meta[n]["summary"] for n in EXPECTED)  # every skill documents itself


def test_bridge_is_in_process_by_default() -> None:
    st = skills.mcp_status()
    assert st["mode"] == "in-process" and st["available"] is True
    assert skills.mcp_is_running() is True


def test_unknown_skill_fails_closed() -> None:
    r = run_skill("does_not_exist")
    assert r["ok"] is False and r["verdict"] == "held" and r["failClosed"] is True


def test_skill_error_is_fail_closed_not_raised() -> None:
    # missing required kwarg -> the decorator must convert it to a held result, not raise
    r = run_skill("provenance_fact_check")  # no text=
    assert r["ok"] is False and r["verdict"] == "held" and r["failClosed"] is True


def test_provenance_flags_misattribution() -> None:
    r = run_skill("provenance_fact_check", text="Confucius wrote the Dao De Jing.")
    assert r["ok"] and r["verdict"] == "flagged"  # known misattribution must not pass clean


def test_claim_verify_and_record_publishable_flag() -> None:
    r = run_skill("claim_verify_and_record",
                  content="Laozi is associated with the Dao De Jing.",
                  idempotency_key="skills-layer-test-claim-1", sources=["wiki://dao"])
    assert r["ok"] and "publishable" in r and "claimId" in r


def test_wiki_grounded_abstains_out_of_wiki() -> None:
    grounded = run_skill("wiki_grounded_answer", query="dao de jing")
    assert grounded["ok"] and grounded["verdict"] == "grounded"
    held = run_skill("wiki_grounded_answer", query="zzz nonexistent topic 9999")
    assert held["ok"] and held["verdict"] == "held" and held["grounded"] is False


def test_belief_revision_abstains_on_unknown_entity() -> None:
    r = run_skill("belief_revision_explore", entity="nonexistent_entity_xyz_9999")
    assert r["ok"] and r["verdict"] == "held" and r["found"] is False


def test_self_extend_probe_is_candidate_labelled() -> None:
    r = run_skill("self_extend_probe")
    assert r["ok"] and r["candidateOnly"] is True and r["level3Evidence"] is False


def test_conscience_and_moral_skills_return_verdicts() -> None:
    c = run_skill("conscience_abstain", text="The author is unknown.")
    assert c["ok"] and c["verdict"]
    m = run_skill("moral_public_standard_review", text="We should deploy before safety evidence is complete.")
    assert m["ok"] and "parliamentVerdict" in m


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} skills-layer tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
