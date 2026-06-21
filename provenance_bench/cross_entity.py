"""Cross-entity generalization — the honest limit of memorized provenance rules.

The self-improvement loop (:mod:`provenance_bench.improvement`) generalizes across
PHRASING: learn a do-not-attribute rule for one ``(author, work)`` and catch it in
unseen wordings. It does NOT generalize across ENTITIES — a rule learned for one
pair says nothing about a *different* author/work it has never seen. This module
makes that gap falsifiable, and contrasts it with a content-free STRUCTURAL
detector to expose the trade-off:

  - memorized rules     — precise (≈0 false positives) but DO NOT transfer to
                          unseen entities (cross-entity recall ≈ 0);
  - structural detector — transfers perfectly across entities (it flags any
                          asserted attribution) but cannot tell a TRUE attribution
                          from a false one (false-positive rate ≈ 1).

Neither suffices. Closing cross-entity generalization at LOW false-positive cost
requires EXTERNAL GROUNDING (retrieval / a knowledge base), not pattern
memorization — which is exactly why Sophia's answer is the retrieval-grounded
verifier-gated loop, not a learned per-pair classifier.

Entities are split so that NO author and NO work appears in both train and test
(a connected-component split), so cross-entity recall genuinely measures transfer
to unseen entities, not leakage.
"""

from __future__ import annotations

import re

from provenance_bench.improvement import HELDOUT_TEMPLATES, _fires, _learn

# A content-free, entity-agnostic detector: does the text assert *an* attribution
# at all? It knows nothing about who wrote what — that is the point.
_ATTR_RX = re.compile(r"\b(?:wrote|authored|penned|composed|is the author of|attributed to)\b", re.I)


def _asserts_attribution(text: str) -> bool:
    return bool(_ATTR_RX.search(text or ""))


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:           # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def entity_disjoint_split(pairs: list, *, seed: int = 0) -> tuple:
    """Split ``pairs`` so no author and no work crosses train/test.

    Authors and works are nodes; each pair links its author to its work. Whole
    connected components are assigned to one side, greedily balancing sizes, so
    the two sides share no entity. ``seed`` rotates component order for variety.
    """
    uf = _UnionFind()
    for p in pairs:
        uf.union(f"a:{p['claimed']}", f"w:{p['work']}")
    comps: dict = {}
    for i, p in enumerate(pairs):
        comps.setdefault(uf.find(f"a:{p['claimed']}"), []).append(i)

    order = sorted(comps.values(), key=lambda idxs: (-len(idxs), idxs[0]))
    if order:
        shift = seed % len(order)
        order = order[shift:] + order[:shift]

    train_idx: list = []
    test_idx: list = []
    for idxs in order:                          # greedy balance into two bins
        (train_idx if len(train_idx) <= len(test_idx) else test_idx).extend(idxs)
    return [pairs[i] for i in train_idx], [pairs[i] for i in test_idx]


def _recall(pairs: list, rules: dict) -> float:
    caught = total = 0
    for p in pairs:
        for t in HELDOUT_TEMPLATES:
            total += 1
            caught += int(_fires(t.format(a=p["claimed"], w=p["work"]), rules))
    return round(caught / total, 4) if total else 0.0


def _structural_recall(pairs: list) -> float:
    caught = total = 0
    for p in pairs:
        for t in HELDOUT_TEMPLATES:
            total += 1
            caught += int(_asserts_attribution(t.format(a=p["claimed"], w=p["work"])))
    return round(caught / total, 4) if total else 0.0


def _structural_fp(controls: list) -> float:
    fp = total = 0
    for c in controls:
        for t in HELDOUT_TEMPLATES:
            total += 1
            fp += int(_asserts_attribution(t.format(a=c["gold"], w=c["work"])))
    return round(fp / total, 4) if total else 0.0


def _memorized_fp(controls: list, rules: dict) -> float:
    fp = total = 0
    for c in controls:
        for t in HELDOUT_TEMPLATES:
            total += 1
            fp += int(_fires(t.format(a=c["gold"], w=c["work"]), rules))
    return round(fp / total, 4) if total else 0.0


def run_cross_entity(pairs: list, true_controls: list, *, seed: int = 0) -> dict:
    """Learn rules on an entity-disjoint TRAIN split and measure transfer to TEST.

    Returns within-entity recall (seen entities), cross-entity recall for both the
    memorized rules and the structural detector, and the false-positive cost of
    each — the falsifiable evidence that pattern memorization does not transfer
    across entities and structure cannot tell true from false.
    """
    train, test = entity_disjoint_split(pairs, seed=seed)
    disjoint = (
        not ({p["claimed"] for p in train} & {p["claimed"] for p in test})
        and not ({p["work"] for p in train} & {p["work"] for p in test})
    )

    rules: dict = {}
    for p in train:
        _learn(rules, p["claimed"], p["work"])

    return {
        "seed": seed,
        "entityDisjoint": disjoint,
        "nTrain": len(train),
        "nTest": len(test),
        "withinEntityRecall": _recall(train, rules),
        "crossEntityRecall_memorized": _recall(test, rules),
        "crossEntityRecall_structural": _structural_recall(test),
        "structuralFalsePositive": _structural_fp(true_controls),
        "memorizedFalsePositive": _memorized_fp(true_controls, rules),
        "interpretation": (
            "Memorized rules are precise but do not transfer across entities; a "
            "structural detector transfers but cannot distinguish true from false "
            "attributions. Low-false-positive cross-entity generalization requires "
            "external grounding (retrieval / knowledge base), not pattern memorization."
        ),
    }
