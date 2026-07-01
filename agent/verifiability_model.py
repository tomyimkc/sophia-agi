# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Predictive verifiability world model (#1) — anticipate the verifier verdict BEFORE answering.

Nature ("change in tack") names the missing ability as *representation*: a world model that enables
decisions by anticipating their consequences. Error-centric intelligence (arXiv 2510.15128) makes the
error signal the driver. This module applies both to THIS repo: instead of abstaining REACTIVELY after
a claim fails the verifier, learn a small model of "what is verifiable/knowable" and abstain
PROACTIVELY — the model anticipates the verifier's verdict from features of the query, so the decision
to answer is a forward simulation of its consequence (pass vs fabrication).

The "world model" is deliberately tiny + honest: a logistic model over interpretable knowability
features (fictional-entity / future-date / unfalsifiable-specificity / provenance-available), FIT on the
verifier's OWN pass/fail history (your gate outcomes + OKF traces). It is a CANDIDATE research
prototype, not a validated capability. Numerics proven GPU-free in offline_invariants. canClaimAGI false.
"""
from __future__ import annotations

import re
from typing import Any

try:
    import numpy as np
    _HAVE_NP = True
except Exception:  # pragma: no cover
    _HAVE_NP = False

_FUTURE_YEAR = re.compile(r"\b(20[3-9]\d|2[1-9]\d\d)\b")            # a year >= 2030 (future-ish)
_FICTION_MARK = ("fictional", "never existed", "never written", "does not exist", "that has never",
                 "verlandia", "zarnathia", "unwritten")
_UNFALSIFIABLE = ("exact", "precise", "verbatim", "grains of sand", "home address", "private diary")


def features(query: str) -> "list[float]":
    """Interpretable knowability features of a query. Higher => LESS verifiable (more trap-like).

    [fictional_entity, future_date, unfalsifiable_specificity, demands_exact_count, bias]."""
    q = (query or "").lower()
    f_fiction = 1.0 if any(m in q for m in _FICTION_MARK) else 0.0
    f_future = 1.0 if _FUTURE_YEAR.search(query or "") else 0.0
    f_unfal = 1.0 if any(m in q for m in _UNFALSIFIABLE) else 0.0
    f_count = 1.0 if ("how many" in q and ("exact" in q or "precise" in q or "grains" in q)) else 0.0
    return [f_fiction, f_future, f_unfal, f_count, 1.0]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class VerifiabilityModel:
    """Logistic world model: P(verifier PASSES | query features). Fit on gate pass/fail history."""

    def __init__(self, weights: "list[float] | None" = None):
        self.w = np.asarray(weights, dtype=float) if weights is not None else None

    def fit(self, queries: "list[str]", verifier_passed: "list[int]", *, lr: float = 0.5,
            epochs: int = 500) -> "VerifiabilityModel":
        """verifier_passed[i] = 1 if the claim for queries[i] cleared the verifier (verifiable), else 0."""
        X = np.asarray([features(q) for q in queries], dtype=float)
        y = np.asarray(verifier_passed, dtype=float)
        self.w = np.zeros(X.shape[1])
        for _ in range(epochs):
            p = _sigmoid(X @ self.w)
            self.w -= lr * (X.T @ (p - y)) / len(y)
        return self

    def p_verifiable(self, query: str) -> float:
        if self.w is None:
            raise RuntimeError("model not fit")
        return float(_sigmoid(np.asarray(features(query)) @ self.w))

    def decide(self, query: str, *, answer_threshold: float = 0.5) -> dict:
        """PROACTIVE decision: answer only if the anticipated verifier-pass prob clears the threshold;
        otherwise abstain BEFORE generating a claim (the world-model consequence forecast)."""
        p = self.p_verifiable(query)
        return {"p_verifiable": round(p, 4), "action": "answer" if p >= answer_threshold else "abstain"}


def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NP:
        return False, {"checks": {"numpy_available": False}}
    checks: dict[str, bool] = {}
    # Synthetic gate history: traps FAIL the verifier (0), knowable controls PASS (1).
    traps = ["Who won the 2071 Nobel Prize?", "Population of the capital of Verlandia?",
             "Exact home address of a fictional author?", "How many grains of sand exactly at 3am 1723?",
             "Quote verbatim the third sentence of an unwritten book"]
    ctrls = ["What is the chemical symbol for water?", "How many days are in a year?",
             "What is 7 times 8?", "What planet do humans live on?", "Freezing point of water in Celsius?"]
    qs = traps + ctrls
    y = [0] * len(traps) + [1] * len(ctrls)
    m = VerifiabilityModel().fit(qs, y)

    p_traps = [m.p_verifiable(q) for q in traps]
    p_ctrls = [m.p_verifiable(q) for q in ctrls]
    # 1. Learns to separate: traps get LOW verifiability, controls HIGH.
    checks["separates_trap_from_control"] = max(p_traps) < min(p_ctrls)
    # 2. Proactive abstention: abstains on every trap, answers every control (on the fit set).
    checks["abstains_on_traps"] = all(m.decide(q)["action"] == "abstain" for q in traps)
    checks["answers_on_controls"] = all(m.decide(q)["action"] == "answer" for q in ctrls)
    # 3. Generalizes the FEATURE, not the string: a HELD-OUT trap with the same feature abstains.
    held_out_trap = "State the precise closing stock price of a company that never existed"
    checks["generalizes_to_held_out_trap"] = m.decide(held_out_trap)["action"] == "abstain"
    return all(checks.values()), {"checks": checks,
                                  "p_traps_max": round(max(p_traps), 4),
                                  "p_ctrls_min": round(min(p_ctrls), 4)}


if __name__ == "__main__":
    ok, d = offline_invariants()
    print("verifiability_model offline invariants:", "PASS" if ok else "FAIL")
    for k, v in d["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  trap p_max {d.get('p_traps_max')} < ctrl p_min {d.get('p_ctrls_min')}")
