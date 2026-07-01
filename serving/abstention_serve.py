# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Serve-time ABSTENTION gate — enforce the cert's measured operating point on the live quant model.

``serving/quant_abstention.py`` MEASURES (offline, with the FP reference) whether a raw-failing quant
model becomes honestly shippable by abstaining on its low-margin tokens, and traces the
coverage/answered-top1 FRONTIER to pick the max-coverage operating point that clears the cert bar.
That is a diagnostic: it needs the FP model, which low-RAM deployment does NOT have.

This module is the DEPLOYABLE half. It reads the chosen operating point (a single nonconformity
THRESHOLD) from a cert artifact, then at serve time decides answer-vs-abstain for each next token using
the QUANT MODEL ALONE — the top1-vs-top2 probability margin, exactly the signal the frontier calibrated
on. No FP model, no labels, no network: one comparison per token. This closes the loop from "the cert
measured v5+abstention is shippable at coverage ~0.86 / answered-top1 ~0.98" to "the server actually
abstains on those tokens." Pure/offline/deterministic; numerics proven in ``offline_invariants``.

The threshold is a CALIBRATED constant, not a live guarantee: it was picked out-of-sample on the cert's
test split. Answered accuracy at serve time will track the cert's measured number only insofar as the
serve distribution matches the calibration distribution — we do NOT claim a per-token guarantee, and
``canClaimAGI`` stays false. Ship this as a hedge that refuses to fabricate on the tokens most likely to
be wrong, never as a capability claim.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from serving.quant_abstention import quant_nonconformity  # noqa: E402

ANSWER = "answer"
ABSTAIN = "abstain"


@dataclass(frozen=True)
class AbstentionPolicy:
    """A calibrated serve-time abstention policy: abstain when quant nonconformity EXCEEDS ``threshold``.

    ``threshold`` is the top1-top2-margin nonconformity cut chosen on the cert's frontier (the
    max-coverage point whose out-of-sample answered_top1 >= ``target_answered``). ``measured_coverage``
    and ``measured_answered_top1`` are the cert's out-of-sample numbers AT that point — provenance, not
    a guarantee. ``n_test`` records the sample size behind them so a thin read is visible, not hidden."""
    threshold: float
    target_answered: float
    measured_coverage: float
    measured_answered_top1: float
    raw_top1: float
    n_test: int
    source: str = "unknown"

    def decide(self, low_probs) -> "list[str]":
        """ANSWER / ABSTAIN per row of a quant next-token distribution (probabilities or logits;
        argmax + top1-top2 margin are scale-monotone so unnormalized logits are fine for the ordering,
        but pass probabilities if you want the threshold to mean what the cert calibrated)."""
        nc = quant_nonconformity(low_probs)
        return [ANSWER if float(v) <= self.threshold else ABSTAIN for v in nc]

    def decide_one(self, low_prob_row) -> str:
        import numpy as np
        return self.decide(np.asarray(low_prob_row, dtype=float).reshape(1, -1))[0]

    def coverage_on(self, low_probs) -> float:
        d = self.decide(low_probs)
        return sum(x == ANSWER for x in d) / len(d) if d else 0.0

    def as_dict(self) -> "dict[str, Any]":
        return asdict(self)


def policy_from_cert(cert_path: "str | Path") -> "AbstentionPolicy":
    """Load the shippable operating point from a certify_lowram.py artifact (``out['abstention_frontier']``).

    Raises if the cert found NO shippable point (``shippable_operating_point is None``) — that means
    abstention cannot rescue the model at the target and there is no honest policy to serve; the caller
    must NOT silently fall back to answering everything (that would re-introduce the fabrication risk the
    whole hedge exists to remove)."""
    data = json.loads(Path(cert_path).read_text())
    fr = data.get("abstention_frontier")
    if not fr:
        raise ValueError(f"{cert_path}: no abstention_frontier block (old cert or frontier errored)")
    best = fr.get("shippable_operating_point")
    if not best:
        raise ValueError(
            f"{cert_path}: shippable_operating_point is None — abstention cannot reach "
            f"target_answered={fr.get('target_answered')} on this model; do NOT ship an answer-everything "
            f"fallback. The recipe fix (v6 QAT) is the path.")
    return AbstentionPolicy(
        threshold=float(best["threshold"]),
        target_answered=float(fr.get("target_answered", 0.97)),
        measured_coverage=float(best["coverage"]),
        measured_answered_top1=float(best["answered_top1"]),
        raw_top1=float(fr.get("raw_top1", 0.0)),
        n_test=int(fr.get("n_test", 0)),
        source=str(cert_path),
    )


# --------------------------------------------------------------------------- #
# Offline invariants — GPU-free, prove the serve gate matches the cert's frontier decision.
# --------------------------------------------------------------------------- #
def offline_invariants() -> "tuple[bool, dict]":
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return False, {"checks": {"numpy_available": False}}

    from serving.quant_abstention import quant_abstention_frontier

    checks: dict[str, bool] = {}
    detail: dict = {}
    rng = np.random.default_rng(0)
    N, V = 400, 20

    # Same separable regime as quant_abstention: confident tokens keep the FP argmax, near-ties flip.
    fp = rng.standard_normal((N, V))
    fp_soft = np.exp(fp) / np.exp(fp).sum(1, keepdims=True)
    fp_arg = fp_soft.argmax(1)
    low = fp.copy()
    for i in range(N):
        if rng.random() < 0.30:
            j = (fp_arg[i] + 1) % V
            low[i, fp_arg[i]] = fp[i, j] + 0.01
            low[i, j] = fp[i, fp_arg[i]]
        else:
            low[i, fp_arg[i]] += 3.0
    low_soft = np.exp(low) / np.exp(low).sum(1, keepdims=True)

    fr = quant_abstention_frontier(fp_soft, low_soft, target_answered=0.97)
    best = fr["shippable_operating_point"]
    pol = AbstentionPolicy(threshold=best["threshold"], target_answered=0.97,
                           measured_coverage=best["coverage"], measured_answered_top1=best["answered_top1"],
                           raw_top1=fr["raw_top1"], n_test=fr["n_test"], source="selftest")

    # 1. The serve gate reproduces the frontier's coverage at the SAME threshold on the SAME test split
    #    (the frontier chose the point on test_idx = second half; check the gate matches there).
    ncal = fr["n_calib"]
    test = low_soft[ncal:]
    cov = pol.coverage_on(test)
    checks["gate_coverage_matches_frontier"] = abs(cov - best["coverage"]) < 1e-9

    # 2. Answered accuracy of the gate on the test split clears the target (this is the whole point).
    agree = (fp_soft.argmax(1) == low_soft.argmax(1))[ncal:]
    d = np.array(pol.decide(test))
    answered = d == ANSWER
    ans_top1 = float(agree[answered].mean()) if answered.any() else 0.0
    checks["gate_answered_clears_target"] = ans_top1 >= 0.97 - 1e-9
    checks["gate_answered_matches_measured"] = abs(ans_top1 - best["answered_top1"]) < 1e-9

    # 3. Abstaining strictly raises answered accuracy over answering everything (the hedge earns its cost).
    raw_all = float(agree.mean())
    checks["hedge_beats_answer_all"] = ans_top1 > raw_all

    # 4. Confident row -> ANSWER, near-tie row -> ABSTAIN (per-token behaviour).
    conf_row = np.zeros(V); conf_row[0] = 10.0
    conf_row = np.exp(conf_row) / np.exp(conf_row).sum()
    tie_row = np.zeros(V); tie_row[0] = 0.51; tie_row[1] = 0.49  # not softmax but valid prob-ish margin
    tie_row = tie_row / tie_row.sum()
    checks["confident_answers"] = pol.decide_one(conf_row) == ANSWER
    checks["near_tie_abstains"] = pol.decide_one(tie_row) == ABSTAIN

    # 5. A None operating point must RAISE, never silently answer-everything.
    raised = False
    try:
        bad = {"abstention_frontier": {"shippable_operating_point": None, "target_answered": 0.99,
                                       "raw_top1": 0.5, "n_test": 10}}
        import tempfile, os
        fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd)
        Path(p).write_text(json.dumps(bad))
        try:
            policy_from_cert(p)
        finally:
            os.unlink(p)
    except ValueError:
        raised = True
    checks["none_operating_point_raises"] = raised

    detail["gate_coverage"] = round(cov, 4)
    detail["gate_answered_top1"] = round(ans_top1, 4)
    detail["answer_all_top1"] = round(raw_all, 4)
    detail["threshold"] = best["threshold"]
    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("abstention_serve offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  gate: answered_top1={detail['gate_answered_top1']} @ coverage={detail['gate_coverage']} "
          f"(answer-all would be {detail['answer_all_top1']}); threshold={detail['threshold']}")
