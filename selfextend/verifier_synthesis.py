# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Synthesize a verifier from labeled examples, then validate it on held-out data.

The honest core of self-extension: faced with a domain it cannot check, the system
*writes a checker* (here, a transparent decision stump over substring features —
program synthesis kept interpretable and deterministic) and only trusts it if it
clears a held-out accuracy bar. If it cannot, it stays abstained (fail-closed: a
verifier you can't validate is worse than none).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Rule:
    """A learned verifier: predict ``label`` iff ``feature`` is present (or absent)."""
    feature: str          # a substring (or "re:<pattern>")
    present: bool         # True: feature present -> positive; False: absent -> positive
    accuracy_train: float

    def predict(self, text: str) -> bool:
        if self.feature.startswith("re:"):
            hit = bool(re.search(self.feature[3:], text or "", re.IGNORECASE))
        else:
            hit = self.feature.lower() in (text or "").lower()
        return hit if self.present else (not hit)


def candidate_features(examples: "list[tuple[str, bool]]", max_feats: int = 200) -> "list[str]":
    """Tokens appearing in the examples — the hypothesis space for the stump.

    Public so cross-module consumers (e.g. ``selfextend.evolve``'s top-k proposer)
    depend on a stable API rather than an underscore-prefixed private helper.
    """
    feats: dict = {}
    for text, _ in examples:
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
            if len(tok) >= 2:
                feats[tok] = feats.get(tok, 0) + 1
    return [f for f, _ in sorted(feats.items(), key=lambda kv: -kv[1])][:max_feats]


# Backward-compat alias for any in-module/private callers.
_candidate_features = candidate_features


def synthesize_verifier(train: "list[tuple[str, bool]]",
                        candidate_features: "list[str] | None" = None) -> "Rule | None":
    """Best decision stump: the (feature, polarity) that best separates labels on
    train. Returns None if train is empty/degenerate."""
    if not train:
        return None
    feats = candidate_features or _candidate_features(train)
    best: "Rule | None" = None
    n = len(train)
    for feat in feats:
        for present in (True, False):
            correct = 0
            for text, label in train:
                hit = (feat.lower() in (text or "").lower())
                pred = hit if present else (not hit)
                correct += int(pred == label)
            acc = correct / n
            if best is None or acc > best.accuracy_train:
                best = Rule(feature=feat, present=present, accuracy_train=round(acc, 4))
    return best


def stratified_split(examples: "list[tuple[str, bool]]", frac: float = 0.5) -> "tuple[list, list]":
    """Split keeping both classes in each side (deterministic), so a verifier is
    learned and validated on label-balanced data rather than one class."""
    pos = [e for e in examples if e[1]]
    neg = [e for e in examples if not e[1]]
    train, heldout = [], []
    for group in (pos, neg):
        k = max(1, int(len(group) * frac)) if len(group) > 1 else len(group)
        train += group[:k]
        heldout += group[k:] or group[:k]  # if a class can't be split, reuse for both
    return train, heldout


def validate(rule: "Rule", heldout: "list[tuple[str, bool]]") -> float:
    if not heldout:
        return 0.0
    correct = sum(int(rule.predict(t) == lab) for t, lab in heldout)
    return round(correct / len(heldout), 4)


def propose_and_validate(train: "list[tuple[str, bool]]", heldout: "list[tuple[str, bool]]",
                         *, threshold: float = 0.8) -> dict:
    """Synthesize a verifier and validate it. ``promoted`` only when held-out accuracy
    clears the bar — otherwise the system stays abstained (fail-closed)."""
    rule = synthesize_verifier(train)
    if rule is None:
        return {"promoted": False, "reason": "no rule synthesizable", "heldoutAccuracy": 0.0}
    acc = validate(rule, heldout)
    return {
        "promoted": acc >= threshold,
        "heldoutAccuracy": acc,
        "trainAccuracy": rule.accuracy_train,
        "rule": {"feature": rule.feature, "present": rule.present},
        "reason": "validated" if acc >= threshold else "below threshold -> abstain",
    }
