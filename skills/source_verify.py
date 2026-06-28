# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: source_verify_audit — audit an answer for fabricated citations + attribution swaps.

Surfaces the 2026-06-28 verification toolkit (citation-existence via Crossref, attribution-swap
via Wikidata) through the Skills layer. HIGH independence (deterministic external records),
fail-open and coverage-bounded. An answer that cites a non-existent study, or credits a real
work to the wrong creator, is flagged and NOT publishable.
"""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "source_verify_audit",
    summary=("Audit an answer for fabricated citations (Crossref existence) and attribution swaps "
             "(Wikidata creator/author/discoverer). HIGH independence, fail-open; a flagged answer "
             "is not publishable. For viral-claim (Google) and misstated-finding (multi-judge) "
             "checks, run tools/run_source_contamination_bench.py --verifier {hybrid,faithfulness}."),
    uses=("source_verify_tool",),
)
def source_verify_audit(*, answer: str, question: str = "") -> dict:
    res = call("source_verify_tool", answer=answer, question=question)
    findings = res.get("findings", []) if isinstance(res, dict) else []
    clean = bool(res.get("clean")) if isinstance(res, dict) else False
    return {
        "verdict": "clean" if clean else "flagged",
        "publishable": clean,
        "findings": findings,
        "nCitations": res.get("nCitations") if isinstance(res, dict) else None,
        "independence": res.get("independence") if isinstance(res, dict) else None,
        "failClosed": True,
        "canClaimAGI": False,
        "detail": res,
    }
