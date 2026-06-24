"""Sophia Skills — MCP-matched, fail-closed skill layer.

A thin, friendly Python surface over the Sophia MCP tools (`sophia_mcp.tools_impl`).
Every skill is fail-closed: it abstains (`verdict: "held"`) rather than raise or
fabricate. Skills auto-register via the `@sophia_skill` decorator on import.

    from skills import run_skill, list_skills
    run_skill("provenance_fact_check", text="Confucius wrote the Dao De Jing.")

Built by sole author tomyimkc.
"""
from __future__ import annotations

from skills.core import SKILLS, list_skills, run_skill, sophia_skill
from skills.mcp_bridge import call, get_mcp_client, mcp_is_running, mcp_status

# Importing each module triggers @sophia_skill auto-registration.
from skills.provenance_fact_check import provenance_fact_check
from skills.source_discipline import source_discipline_enforce
from skills.conscience_abstain import conscience_abstain
from skills.moral_parliament_decide import moral_parliament_decide
from skills.claim_verify import claim_verify_and_record
from skills.belief_revision import belief_revision_explore
from skills.wiki_grounded_answer import wiki_grounded_answer
from skills.moral_public_standard import moral_public_standard_review
from skills.deception_scan import deception_scan
from skills.contradiction_audit import contradiction_audit
from skills.council_adjudicate import council_adjudicate
from skills.self_extend_probe import self_extend_probe

__all__ = [
    # registry / bridge
    "sophia_skill", "run_skill", "list_skills", "SKILLS",
    "call", "get_mcp_client", "mcp_is_running", "mcp_status",
    # v1 skills
    "provenance_fact_check", "source_discipline_enforce", "conscience_abstain",
    "moral_parliament_decide", "claim_verify_and_record",
    # v2 skills
    "belief_revision_explore", "wiki_grounded_answer", "moral_public_standard_review",
    "deception_scan", "contradiction_audit", "council_adjudicate", "self_extend_probe",
]
