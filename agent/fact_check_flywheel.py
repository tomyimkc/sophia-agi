"""Quarantine → recheck → promote loop for externally verified claims.

The fact-check gate already emits learning candidates for accepted out-of-wiki
claims. This module keeps the loop honest: candidates are first written to a
quarantine ledger, then independently rechecked before promotion to a provisional
external knowledge file. Nothing here writes into canonical OKF/wiki records.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from agent.fact_check_gate import decision_to_dict, fact_check_text


def extract_learning_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in report.get("cases", []):
        for claim in case.get("claims", []):
            cand = claim.get("learningCandidate")
            if cand:
                out.append(cand)
    # stable de-dup by claimId
    dedup: dict[str, dict[str, Any]] = {}
    for cand in out:
        dedup[str(cand.get("claimId", cand.get("claim")))] = cand
    return list(dedup.values())


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append rows by ``claimId`` without duplicating reruns."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.add(str(obj.get("claimId", obj.get("claim", line))))
    written = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            key = str(row.get("claimId", row.get("claim", json.dumps(row, sort_keys=True))))
            if key in seen:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(key)
            written += 1
    return written


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recheck_candidate(candidate: dict[str, Any], *, retriever=None, entailment=None, doi_resolver=None, url_resolver=None) -> dict[str, Any]:
    decision = decision_to_dict(fact_check_text(
        str(candidate.get("claim", "")),
        retriever=retriever,
        entailment=entailment,
        doi_resolver=doi_resolver,
        url_resolver=url_resolver,
        learn=False,
    ))
    accepted = decision["verdict"] == "accepted"
    promoted = dict(candidate)
    promoted.update({
        "promotionState": "promoted_provisional" if accepted else "stays_quarantined",
        "independentRecheck": {
            "verdict": decision["verdict"],
            "reason": decision["reason"],
            "accepted": accepted,
        },
        "canonicalWikiWrite": False,
    })
    return promoted


def run_flywheel_from_report(
    report: dict[str, Any], *, retriever=None, entailment=None, doi_resolver=None, url_resolver=None,
) -> dict[str, Any]:
    candidates = extract_learning_candidates(report)
    rechecked = [recheck_candidate(c, retriever=retriever, entailment=entailment,
                                   doi_resolver=doi_resolver, url_resolver=url_resolver)
                 for c in candidates]
    promoted = [c for c in rechecked if c.get("promotionState") == "promoted_provisional"]
    return {
        "schema": "sophia.fact_check.flywheel.v1",
        "canonicalWikiWrite": False,
        "nCandidates": len(candidates),
        "nPromotedProvisional": len(promoted),
        "nStillQuarantined": len(rechecked) - len(promoted),
        "candidates": rechecked,
    }


__all__ = ["extract_learning_candidates", "append_jsonl", "load_json", "recheck_candidate", "run_flywheel_from_report"]
