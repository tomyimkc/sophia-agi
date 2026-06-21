"""A measured, contamination-free self-improvement loop.

The falsifiable claim: a system that learns provenance RULES from its own
failures gets measurably better at catching lineage-merges it has never seen
*phrased that way* — cycle over cycle, on a held-out split, at zero false-positive
cost. If the held-out metric does not rise, the loop does not work.

Design that forbids cheating:
  - TRAIN and HELD-OUT are **disjoint phrasing templates** of the same
    (forbidden-author, work) pairs. Learning happens only on TRAIN phrasings;
    scoring happens only on HELD-OUT phrasings. A learned rule therefore has to
    *generalize across surface form* to score — it cannot memorize the test text.
  - The learned artifact is a do-not-attribute rule (the gate's records). It
    generalizes across phrasing (the honest scope: surface form, not new
    entities).
  - Held-out false-positive cost is measured on TRUE attributions and must stay
    ~0, so "learning" can't be faked by over-blocking.

Deterministic (templated text + the machine-checked gate, no model call) so the
loop's mechanics are isolated and the result is reproducible. A model-in-the-loop
version is the next step.
"""

from __future__ import annotations

import re

from agent.guarded import check_claim
from provenance_bench.dataset import _alt_titles, _author_marker

# Disjoint phrasing sets — the contamination guard asserts the intersection is
# empty. All phrasings are catchable by the gate, so held-out recall reflects
# whether the RULE was learned, not whether the phrasing happens to match.
TRAIN_TEMPLATES = ["{a} wrote {w}.", "{w} was written by {a}."]
HELDOUT_TEMPLATES = ["{a} is the author of {w}.", "{w} is attributed to {a}."]


def _record(claimed: str, work: str) -> tuple[str, dict]:
    rid = re.sub(r"[^a-z0-9]+", "_", work.lower()).strip("_")
    return rid, {"canonicalTitleEn": work, "altTitlesEn": _alt_titles(work),
                 "doNotAttributeTo": [_author_marker(claimed)]}


def _learn(rules: dict, claimed: str, work: str) -> None:
    rid, rec = _record(claimed, work)
    if rid in rules:
        for a in rec["doNotAttributeTo"]:
            if a not in rules[rid]["doNotAttributeTo"]:
                rules[rid]["doNotAttributeTo"].append(a)
    else:
        rules[rid] = rec


def _fires(text: str, rules: dict) -> bool:
    return not check_claim(text, records=rules)["passed"]


def run_loop(pairs: list[dict], true_controls: list[dict], *, batch: int = 8, cycles: int = 6,
             answer_fn=None) -> dict:
    """``pairs``: [{claimed, work}]; ``true_controls``: [{gold, work}].

    ``answer_fn(claimed, work) -> str`` (optional) sources the TRAIN text from a
    real model instead of the deterministic template, so the loop learns from
    *actual model failures*. The contract is unchanged: a rule is mined only when
    the TRAIN text asserts the forbidden attribution and the current gate misses
    it; scoring is still on the disjoint held-out phrasings. Default (None) keeps
    the deterministic, reproducible path.
    """
    if set(TRAIN_TEMPLATES) & set(HELDOUT_TEMPLATES):
        raise AssertionError("contamination: train and held-out templates overlap")

    rules: dict = {}
    curve: list[dict] = []
    revealed = 0
    for k in range(1, cycles + 1):
        # --- learn from TRAIN failures only (gate currently misses -> mine rule) ---
        for p in pairs[revealed: revealed + batch]:
            if answer_fn is not None:
                train_text = answer_fn(p["claimed"], p["work"]) or ""
            else:
                train_text = TRAIN_TEMPLATES[0].format(a=p["claimed"], w=p["work"])
            # A failure is text that ASSERTS the forbidden attribution yet the
            # current rules miss it. Checking "asserts" with a one-off rule (not
            # just "gate passed") is what makes a model answer_fn honest: clean
            # model text passes the gate but must NOT be mined as a failure.
            oneoff: dict = {}
            _learn(oneoff, p["claimed"], p["work"])
            asserts = _fires(train_text, oneoff)
            missed_now = not _fires(train_text, rules)
            if asserts and missed_now:
                _learn(rules, p["claimed"], p["work"])
        revealed = min(revealed + batch, len(pairs))

        # --- score on HELD-OUT phrasings of ALL pairs (never trained on these) ---
        caught = total = 0
        for p in pairs:
            for t in HELDOUT_TEMPLATES:
                total += 1
                caught += int(_fires(t.format(a=p["claimed"], w=p["work"]), rules))
        # --- held-out false-positive cost on TRUE attributions ---
        fp = ft = 0
        for c in true_controls:
            for t in HELDOUT_TEMPLATES:
                ft += 1
                fp += int(_fires(t.format(a=c["gold"], w=c["work"]), rules))

        curve.append({
            "cycle": k,
            "pairsRevealed": revealed,
            "rulesLearned": len(rules),
            "heldoutRecall": round(caught / total, 4) if total else 0.0,
            "heldoutFalsePositive": round(fp / ft, 4) if ft else 0.0,
        })

    recalls = [c["heldoutRecall"] for c in curve]
    return {
        "curve": curve,
        "finalRecall": recalls[-1] if recalls else 0.0,
        "monotoneNonDecreasing": all(recalls[i] <= recalls[i + 1] for i in range(len(recalls) - 1)),
        "maxFalsePositive": max((c["heldoutFalsePositive"] for c in curve), default=0.0),
        "contaminationGuard": "TRAIN/HELD-OUT templates disjoint; rules learned only from TRAIN phrasings",
        "scope": "learns specific do-not-attribute rules; generalizes across phrasing (surface form), not across new entities",
    }
