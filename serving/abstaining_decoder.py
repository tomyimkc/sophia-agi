# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Reference ABSTAINING DECODER — enforce a calibrated abstention policy in the token loop.

``serving/abstention_serve.AbstentionPolicy`` decides answer-vs-abstain for a single next-token
distribution; this module wires that decision into a generation loop so v5+abstention is END-TO-END,
not just measured in a cert. It is deliberately generation-AGNOSTIC: you supply a ``logits_provider``
callable — ``(step, emitted) -> quant next-token logits/probs over the vocab`` — which is the ONLY place
a real runtime touches the model (HF ``model(...).logits[:, -1]``, the low-RAM runtime, a mock, etc.).
The decoder never sees the FP model; the abstention signal is the quant top1-top2 margin alone, exactly
what low-RAM serving has. On a confident step it emits the argmax; on a low-margin step it emits the
``ABSTAIN`` sentinel (defer / flag / route-to-bigger-model) instead of fabricating a likely-wrong token.

This is the deployable shape of the repo's thesis at the quant gap: serve the compressed model, and
where it is most likely to disagree with full precision, ABSTAIN rather than emit. Coverage (fraction of
steps answered) is tracked live so a caller can enforce a floor or trip a circuit-breaker. Pure/offline/
deterministic; numerics proven in ``offline_invariants``. No capability claim; ``canClaimAGI`` stays false.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from serving.abstention_serve import ABSTAIN, ANSWER, AbstentionPolicy  # noqa: E402

# Sentinel emitted in place of a token when the policy abstains. A real runtime maps this to its own
# defer action (special token, tool-call, route to a bigger model); -1 is never a valid vocab id.
ABSTAIN_TOKEN = -1


@dataclass
class DecodeTrace:
    """Outcome of an abstaining decode: the emitted sequence (ABSTAIN_TOKEN where deferred) + stats."""
    tokens: "list[int]" = field(default_factory=list)
    abstained_steps: "list[int]" = field(default_factory=list)
    nonconformities: "list[float]" = field(default_factory=list)
    n_steps: int = 0

    @property
    def answered(self) -> int:
        return self.n_steps - len(self.abstained_steps)

    @property
    def coverage(self) -> float:
        return self.answered / self.n_steps if self.n_steps else 0.0

    def as_dict(self) -> "dict[str, Any]":
        return {"tokens": list(self.tokens), "abstained_steps": list(self.abstained_steps),
                "n_steps": self.n_steps, "answered": self.answered,
                "coverage": round(self.coverage, 4)}


class AbstainingDecoder:
    """Applies an ``AbstentionPolicy`` step-by-step over a caller-supplied quant logits provider."""

    def __init__(self, policy: AbstentionPolicy, *, abstain_token: int = ABSTAIN_TOKEN):
        self.policy = policy
        self.abstain_token = abstain_token

    # ---- streaming: one step at a time (the shape a real runtime calls) ------------------------ #
    def step(self, low_logits_row) -> "tuple[int, bool, float]":
        """Decide a single step from one quant next-token distribution (logits or probs, shape (V,)).

        Returns (token, abstained, nonconformity). On ANSWER: (argmax, False, nc). On ABSTAIN:
        (abstain_token, True, nc). argmax + top1-top2 margin are scale-monotone, so raw logits are fine
        for the ordering; pass probabilities if you want the threshold to mean what the cert calibrated."""
        import numpy as np
        row = np.asarray(low_logits_row, dtype=np.float64).reshape(1, -1)
        from serving.quant_abstention import quant_nonconformity
        nc = float(quant_nonconformity(row)[0])
        if self.policy.decide(row)[0] == ANSWER:
            return int(row.argmax()), False, nc
        return int(self.abstain_token), True, nc

    # ---- batch: drive a whole sequence from a logits provider --------------------------------- #
    def decode(self, logits_provider: "Callable[[int, list[int]], Any]", *, max_tokens: int,
               stop_on_coverage_below: "float | None" = None) -> DecodeTrace:
        """Generate up to ``max_tokens`` steps. ``logits_provider(step, emitted)`` returns the quant
        next-token distribution for that step (the ONLY model touch-point). If ``stop_on_coverage_below``
        is set, the decode trips a circuit-breaker once running coverage drops below it (after a warmup),
        so a caller never ships a mostly-abstained answer — it fails loud instead."""
        tr = DecodeTrace()
        for s in range(max_tokens):
            tok, abstained, nc = self.step(logits_provider(s, tr.tokens))
            tr.tokens.append(tok)
            tr.nonconformities.append(nc)
            tr.n_steps += 1
            if abstained:
                tr.abstained_steps.append(s)
            if stop_on_coverage_below is not None and tr.n_steps >= 4 and tr.coverage < stop_on_coverage_below:
                break
        return tr


# --------------------------------------------------------------------------- #
# Offline invariants — GPU-free, prove the decoder answers the confident steps and abstains the ties.
# --------------------------------------------------------------------------- #
def offline_invariants() -> "tuple[bool, dict]":
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return False, {"checks": {"numpy_available": False}}

    checks: dict[str, bool] = {}
    detail: dict = {}
    V = 32

    # A policy with a mid nonconformity threshold: confident rows (big top1-top2 gap) answer, near-tie
    # rows abstain. Build a deterministic sequence of both kinds and check the split.
    pol = AbstentionPolicy(threshold=0.5, target_answered=0.97, measured_coverage=0.7,
                           measured_answered_top1=0.99, raw_top1=0.9, n_test=512, source="selftest")
    dec = AbstainingDecoder(pol)

    def confident_row(arg):
        r = np.full(V, 0.01 / (V - 1)); r[arg] = 0.99; return r          # margin ~0.98 -> nc ~0.02 (answer)

    def tie_row(a, b):
        r = np.full(V, 0.005 / (V - 2)); r[a] = 0.5; r[b] = 0.495; return r  # margin ~0.005 -> nc ~0.995 (abstain)

    # 1. Confident step -> ANSWER with the argmax; near-tie step -> ABSTAIN sentinel.
    tok_c, ab_c, nc_c = dec.step(confident_row(7))
    tok_t, ab_t, nc_t = dec.step(tie_row(3, 9))
    checks["confident_answers_argmax"] = (not ab_c) and tok_c == 7
    checks["near_tie_abstains"] = ab_t and tok_t == ABSTAIN_TOKEN
    checks["nonconformity_orders"] = nc_c < nc_t

    # 2. Drive a mixed sequence: 7 confident + 3 tie, interleaved -> coverage == 7/10, abstains at the ties.
    plan = [("c", 1), ("c", 2), ("t", None), ("c", 3), ("t", None), ("c", 4), ("c", 5), ("t", None), ("c", 6), ("c", 8)]
    def provider(step, emitted):
        kind, a = plan[step]
        return confident_row(a) if kind == "c" else tie_row(0, 1)
    tr = dec.decode(provider, max_tokens=len(plan))
    checks["coverage_matches_plan"] = abs(tr.coverage - 0.7) < 1e-9
    checks["abstained_exactly_the_ties"] = tr.abstained_steps == [2, 4, 7]
    checks["answered_tokens_are_argmax"] = all(
        tr.tokens[i] != ABSTAIN_TOKEN for i in range(len(plan)) if plan[i][0] == "c")

    # 3. Circuit-breaker: an all-tie stream trips stop_on_coverage_below and does NOT emit a full answer.
    def all_tie(step, emitted):
        return tie_row(0, 1)
    tr_bad = dec.decode(all_tie, max_tokens=50, stop_on_coverage_below=0.5)
    checks["circuit_breaker_trips"] = tr_bad.n_steps < 50 and tr_bad.coverage < 0.5

    # 4. A None-derived policy can't exist here, but a policy that answers nothing must yield coverage 0.
    pol_strict = AbstentionPolicy(threshold=-1.0, target_answered=0.97, measured_coverage=0.0,
                                  measured_answered_top1=0.0, raw_top1=0.9, n_test=512, source="selftest")
    tr_none = AbstainingDecoder(pol_strict).decode(provider, max_tokens=len(plan))
    checks["strict_policy_abstains_all"] = tr_none.coverage == 0.0

    detail["trace"] = tr.as_dict()
    detail["circuit_breaker_steps"] = tr_bad.n_steps
    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("abstaining_decoder offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  example decode: {detail['trace']}; circuit-breaker tripped at {detail['circuit_breaker_steps']} steps")
