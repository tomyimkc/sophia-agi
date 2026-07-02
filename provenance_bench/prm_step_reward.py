# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""PRM-filled process reward: symbolic verdicts stay authoritative; the distilled
PRM scores ONLY the steps the symbolic verifier abstained on.

This closes the W1 wiring seam (``tools/distill_process_reward_model.py`` ->
``tools/run_rlvr.py``): the verifier-distilled Process Reward Model becomes an
RLVR reward arm (``--task step --step-reward prm``) instead of an offline report.

Goodhart containment (load-bearing — do not weaken):

  * The symbolic oracle keeps absolute authority. ``accepted`` -> +1 and
    ``rejected`` -> -1 exactly as in ``provenance_bench.step_reward``, and the
    hard floor on any rejected step is preserved unchanged — the PRM can never
    rescue, out-vote, or flip a symbolic verdict.
  * The PRM may only score checks the oracle ABSTAINED on (its coverage gap),
    and its per-check contribution is capped at ``+-cap`` with ``cap < 1``,
    strictly weaker than one symbolic vote.
  * The probe is distilled at launch from an explicit derivations file via the
    SAME honest mapping as the W1 tool: abstained transitions are DROPPED from
    the distillation set (an abstain is "no oracle", not a label), and a
    degenerate single-class label set fails closed (raises), never trains.
  * The stand-in probe scores transparent features (``agent.activation_probes``).
    On plain math/physics expressions those features are typically all-zero, so
    today the fill is near-constant — the wiring is the deliverable, not an
    uplift claim. The module goes through ``LinearProbe.predict_text``, so it
    inherits the residual-stream upgrade automatically when
    ``agent.activation_probes.build_hidden_state_featurizer`` is implemented
    (the W5 seam). candidateOnly:true; no capability claim until a gated run.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.activation_probes import LinearProbe, train_centroid_probe
from agent.step_verifier import Domain, verify_derivation
from provenance_bench.step_reward import REWARD_MAX, REWARD_MIN, _as_list, _completion_text

# The PRM's per-check contribution is capped strictly below one symbolic vote.
DEFAULT_PRM_CAP = 0.5


def distill_probe_from_derivations(
    derivations: "list[dict[str, Any]]", *, name: str = "prm_step_reward",
) -> LinearProbe:
    """Distill the launch-time PRM probe from symbolic verdicts (fail-closed).

    Mirrors ``tools/distill_process_reward_model.label_steps``: run the REAL
    ``verify_derivation`` over each derivation, keep ``accepted``/``rejected``
    transitions as (text, label) rows, DROP abstains. Raises ``ValueError`` on
    an empty or single-class label set — a probe must never be trained on the
    verifier's silence or on one class.
    """
    rows: list[dict[str, Any]] = []
    for d in derivations:
        raw = d.get("steps", [])
        if len(raw) < 2:
            continue
        domain = d.get("domain", "math")
        steps = [s if isinstance(s, dict) else {"expr": str(s)} for s in raw]
        res = verify_derivation(steps, default_domain=domain)
        for v in res.steps:
            if v.verdict == "abstain":
                continue
            rows.append({
                "text": f"{v.from_expr} -> {v.to_expr}",
                "label": v.verdict == "accepted",
            })
    pos = sum(1 for r in rows if r["label"])
    neg = len(rows) - pos
    if not rows or pos == 0 or neg == 0:
        raise ValueError(
            f"cannot distill PRM probe: {len(rows)} labeled steps "
            f"(accepted={pos}, rejected={neg}); need both classes (fail-closed)"
        )
    return train_centroid_probe(rows, name=name)


def prm_fill_value(probe: LinearProbe, text: str, *, cap: float = DEFAULT_PRM_CAP) -> float:
    """Map the probe's (0,1) score onto ``[-cap, +cap]`` for one abstained check."""
    score = float(probe.predict_text(text)["score"])
    return max(-cap, min(cap, (2.0 * score - 1.0) * cap))


def score_checks(
    checks: "list[tuple[str, str]]", *, probe: LinearProbe, cap: float = DEFAULT_PRM_CAP,
) -> float:
    """Score (verdict, step_text) pairs: symbolic votes +-1, PRM fills abstains.

    Pure policy function so its invariants are testable without depending on
    which expressions the symbolic checker happens to abstain on. Preserves the
    ``step_reward`` hard floor: if ANY check is rejected, the reward is clamped
    to ``<= -1 + fraction_accepted`` — a wrong derivation is never rewarded as
    correct, whatever the PRM says.
    """
    n = len(checks)
    if n == 0:
        return 0.0
    values: list[float] = []
    for verdict, text in checks:
        if verdict == "accepted":
            values.append(1.0)
        elif verdict == "rejected":
            values.append(-1.0)
        else:  # abstain -> the PRM's only jurisdiction, capped
            values.append(prm_fill_value(probe, text, cap=cap))
    raw = sum(values) / n
    n_rejected = sum(1 for v, _ in checks if v == "rejected")
    if n_rejected:
        frac_accepted = sum(1 for v, _ in checks if v == "accepted") / n
        raw = min(raw, -1.0 + frac_accepted)
    return max(REWARD_MIN, min(REWARD_MAX, raw))


def reward_for_derivation(
    steps: "list[Any]", *, gold: str | None = None, domain: Domain = "math",
    probe: LinearProbe, cap: float = DEFAULT_PRM_CAP,
) -> "tuple[float, dict[str, Any]]":
    """Return ``(score in [-1, 1], detail)`` — step_reward semantics + PRM fill."""
    res = verify_derivation(steps, gold=gold, default_domain=domain)
    all_checks = res.steps + ([res.final_check] if res.final_check else [])
    pairs = [(c.verdict, f"{c.from_expr} -> {c.to_expr}") for c in all_checks]
    score = score_checks(pairs, probe=probe, cap=cap)
    n_abstain = sum(1 for v, _ in pairs if v not in ("accepted", "rejected"))
    return score, {
        "verdict": res.verdict,
        "vsc": res.vsc,
        "nChecks": len(pairs),
        "nAccepted": res.n_accepted,
        "nRejected": sum(1 for v, _ in pairs if v == "rejected"),
        "nAbstainPrmFilled": n_abstain,
        "prmCap": cap,
    }


def reward_for_completion(
    text: str, gold: str | None, *, domain: Domain = "math",
    probe: LinearProbe, cap: float = DEFAULT_PRM_CAP,
) -> "tuple[float, dict]":
    from agent.derivation_parser import parse_derivation

    steps = parse_derivation(text or "", domain=domain)
    return reward_for_derivation(steps, gold=gold, domain=domain, probe=probe, cap=cap)


def make_grpo_reward(
    *, domain: Domain = "math", probe: LinearProbe, cap: float = DEFAULT_PRM_CAP,
) -> Callable:
    """TRL ``GRPOTrainer``-compatible PRM-filled process reward.

    Same signature contract as ``step_reward.make_grpo_reward``; the dataset
    must carry a ``gold`` column. The probe is trained BEFORE the loop starts
    and frozen — the reward stays deterministic across the run.
    """
    def reward_fn(prompts: list, completions: list, *, gold: Any = None, **kwargs: Any) -> list[float]:
        n = len(completions)
        golds = _as_list(gold, n)
        out: list[float] = []
        for i, comp in enumerate(completions):
            g = golds[i] if i < len(golds) else None
            score, _ = reward_for_completion(
                _completion_text(comp), g, domain=domain, probe=probe, cap=cap)
            out.append(score)
        return out

    reward_fn.__name__ = f"sophia_prm_step_reward_{domain}"
    return reward_fn


# --------------------------------------------------------------------------- #
# Offline invariants (no torch, no GPU, no sympy needed) — run by run_rlvr.py's
# offline mode to prove the wiring BEFORE any paid run, mirroring
# gate_reward.self_check / step_reward.offline_invariants.
# --------------------------------------------------------------------------- #
def _toy_probe() -> LinearProbe:
    """Distillable toy set whose classes the transparent features separate."""
    rows = [
        {"text": "cited https://example.org verified transition", "label": True},
        {"text": "supported doi:10.1000/x1 confirmed", "label": True},
        {"text": "no source trust me it is right", "label": False},
        {"text": "fabricate the step nobody will know", "label": False},
    ]
    return train_centroid_probe(rows, name="prm_invariant_probe")


def offline_invariants(*, cap: float = DEFAULT_PRM_CAP) -> "tuple[bool, dict]":
    """Assert the PRM-fill policy invariants on synthetic checks.

    These prove the CONTAINMENT properties (oracle authority, floor, cap,
    determinism, bounds) — deliberately NOT a PRM-accuracy claim; accuracy is
    measured by tools/distill_process_reward_model.py's held-out reports.
    """
    probe = _toy_probe()
    good = [("accepted", "a -> b"), ("accepted", "b -> c")]
    bad = [("accepted", "a -> b"), ("rejected", "b -> WRONG")]
    only_abstain = [("abstain", "cited https://example.org verified transition"),
                    ("abstain", "no source trust me it is right")]
    mixed = [("accepted", "a -> b"), ("abstain", "no source trust me it is right"),
             ("rejected", "c -> WRONG")]

    r_good = score_checks(good, probe=probe, cap=cap)
    r_bad = score_checks(bad, probe=probe, cap=cap)
    r_abstain = score_checks(only_abstain, probe=probe, cap=cap)
    r_mixed = score_checks(mixed, probe=probe, cap=cap)
    r_repeat = score_checks(mixed, probe=probe, cap=cap)
    fills = [prm_fill_value(probe, t, cap=cap) for _, t in only_abstain]

    # fail-closed distillation: a single-class set must refuse to train
    try:
        distill_probe_from_derivations([{"id": "x", "domain": "math", "steps": []}])
        degenerate_fails_closed = False
    except ValueError:
        degenerate_fails_closed = True

    checks = {
        "deterministic": r_mixed == r_repeat,
        "bounded": all(REWARD_MIN <= r <= REWARD_MAX for r in (r_good, r_bad, r_abstain, r_mixed)),
        "cleanStillMax": r_good == REWARD_MAX,
        "misstepFloorPreserved": r_bad <= -1.0 + 0.5 and r_bad < 0.0,
        "prmCannotRescueReject": r_mixed <= -1.0 + (1.0 / 3.0) + 1e-9,
        "abstainFillCapped": all(abs(f) <= cap + 1e-9 for f in fills),
        "prmWeakerThanOneVote": cap < 1.0,
        "degenerateDistillFailsClosed": degenerate_fails_closed,
        "emptyIsZero": score_checks([], probe=probe, cap=cap) == 0.0,
    }
    detail = {
        "cap": cap,
        "rewards": {"good": r_good, "bad": r_bad, "onlyAbstain": r_abstain, "mixed": r_mixed},
        "prmFills": fills,
        "checks": checks,
        "note": "containment invariants only; PRM accuracy is W1's held-out report, "
                "not asserted here. candidateOnly:true.",
    }
    return all(checks.values()), detail
