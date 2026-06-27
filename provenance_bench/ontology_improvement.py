# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Inference-uplift A/B harness for the concept-discipline gate.

The falsifiable inference claim (no training needed): wrapping ANY policy in the
concept gate (arm B = guarded) lowers the concept-merge-violation rate and the
confident-wrong rate vs the raw policy (arm A), at a measured, reported
over-abstention cost — on the philosopher-reasoning pack. The gate cannot make a
model *smarter*; it converts a confident cross-tradition merge into an abstention
(or, with a ``repair`` callable, a re-generated disciplined answer). If the
held-out violation-rate delta's bootstrap CI does not exclude 0, the gate brings
no inference uplift.

Model-agnostic: ``policy(item) -> text`` is the raw generator (a mock here, a real
model client in ``tools/run_concept_discipline_bench.py``). The same harness scores
a raw arm and a guarded arm, then compares them with ``concept_metrics``. This is
the SEAM the RLVR run (``tools/run_rlvr.py --task concept``) later plugs a trained
model into. See docs/11-Platform/Ontology-Claim-Boundary.md.
"""
from __future__ import annotations

from typing import Any, Callable

from agent.guarded import check_claim
from agent.philosopher_eval import _abstained, classify_response
from agent.verifiers import _load_concept_traditions
from provenance_bench import concept_metrics

# A safe answer that itself clears the gate (no confabulation). The guarded arm
# falls back to this when the raw output trips the gate and no repair is supplied.
GUARDED_ABSTENTION = (
    "I won't assert that identity: a cross-tradition concept relation needs a stated "
    "respect of comparison and a source. The two are not identical as stated."
)

Policy = Callable[[dict], str]


def to_record(item: dict, text: str) -> dict:
    """Normalize one (item, response) into a ``concept_metrics`` record."""
    outcome = classify_response(item, text)
    return {
        "id": item.get("id"),
        "family": item.get("family"),
        "answerable": bool(item.get("answerable", True)),
        "abstained": _abstained(text),
        "violation": not check_claim(text).get("passed", True),
        "correct": outcome == "correct",
        "confidence": item.get("confidence"),
    }


def guard(text: str, *, item: dict | None = None, repair: Callable[[dict, str], str] | None = None) -> str:
    """The inference-time gate: pass clean text through; on a violation, repair (if
    a repair generator is supplied and it clears the gate) else abstain. Fail-closed:
    a repair that still trips the gate is discarded in favour of the abstention."""
    if check_claim(text).get("passed", True):
        return text
    if repair is not None and item is not None:
        fixed = repair(item, text)
        if check_claim(fixed).get("passed", True):
            return fixed
    return GUARDED_ABSTENTION


def run_arm(items: list[dict], policy: Policy, *, guarded: bool = False,
            repair: Callable[[dict, str], str] | None = None) -> list[dict]:
    """Run a policy over the items (F8 judge-control excluded) -> records."""
    records: list[dict] = []
    for item in items:
        if item.get("family") == "F8":
            continue
        text = policy(item)
        if guarded:
            text = guard(text, item=item, repair=repair)
        records.append(to_record(item, text))
    return records


def run_ab(items: list[dict], policy: Policy, *,
           repair: Callable[[dict, str], str] | None = None, seed: int = 0) -> dict[str, Any]:
    """Baseline (raw policy) vs treatment (same policy + concept gate). Returns the
    ``concept_metrics.compare_arms`` report (per-arm summaries + bootstrapped deltas
    with CIs). A positive uplift: violation rate down (CI excludes 0)."""
    baseline = run_arm(items, policy, guarded=False)
    treatment = run_arm(items, policy, guarded=True, repair=repair)
    report = concept_metrics.compare_arms(baseline, treatment, seed=seed)
    report["arm"] = {"baseline": "raw policy", "treatment": "policy + concept gate (guard)"}
    return report


# --- deterministic reference policies (offline, no model) --------------------- #
def _extract_pair(item: dict) -> tuple[str, str]:
    """Find the two cross-tradition concept terms an F1/F5 prompt is about, so the
    reference policies can construct a gate-tripping (or distinction-drawing)
    sentence with the REAL lexicon terms. Falls back to explicit fields / generics."""
    if item.get("subject") and item.get("object"):
        return str(item["subject"]), str(item["object"])
    try:
        lex = _load_concept_traditions()
    except Exception:
        lex = {}
    low = (item.get("prompt") or "").lower()
    hits = [term for term in sorted(lex, key=len, reverse=True) if term in low]
    # keep first two terms from distinct traditions, preserving prompt order
    hits.sort(key=lambda t: low.find(t))
    chosen: list[str] = []
    seen_trad: set[str] = set()
    for t in hits:
        trad = lex.get(t)
        if trad not in seen_trad:
            chosen.append(t)
            seen_trad.add(trad)
        if len(chosen) == 2:
            break
    if len(chosen) == 2:
        return chosen[0], chosen[1]
    return "the first concept", "the second concept"


def naive_policy(item: dict) -> str:
    """Always asserts a confident answer — including unscoped cross-tradition
    identity on F1/F5. The 'ungated base model' stand-in: high violation rate."""
    fam = item.get("family")
    if fam in ("F1", "F5"):
        subj, obj = _extract_pair(item)
        return f"{subj} is identical to {obj}; they are the same thing."
    # boolean families: confidently say yes regardless (a careless asserter).
    return "Yes, that is correct."


def disciplined_policy(item: dict) -> str:
    """Draws the distinction on F1/F5, abstains on ill-posed, answers gold on the
    boolean families. The 'concept-disciplined' stand-in: ~zero violations."""
    fam = item.get("family")
    if fam in ("F1", "F5"):
        subj, obj = _extract_pair(item)
        return (f"{subj} is not identical to {obj}; they differ and belong to distinct "
                f"traditions, so the identification is unscoped.")
    if not item.get("answerable", True):
        return "This question is ill-posed, so I abstain rather than assert a determinate answer."
    gold = item.get("gold")
    if gold is True:
        return "Yes — that follows."
    if gold is False:
        return "No — that does not follow."
    return "This question is underdetermined; I abstain."


def reference_repair(item: dict, _text: str) -> str:
    """A deterministic 'successful re-generation' for the offline harness: when the
    raw answer trips the gate, the guarded loop re-answers with the disciplined
    response. For a real model the repair is a re-prompt with the repair hint; here
    it is the disciplined reference, so the mock A/B exercises the FULL loop
    (detect -> repair -> draw the distinction), not just abstain-on-fail."""
    return disciplined_policy(item)


__all__ = [
    "GUARDED_ABSTENTION", "to_record", "guard", "run_arm", "run_ab",
    "naive_policy", "disciplined_policy", "reference_repair",
]
