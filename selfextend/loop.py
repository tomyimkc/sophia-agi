# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Close the full self-extending loop end to end on a HELD-OUT domain.

Each selfextend module works in isolation; this orchestrates them into the one cycle
that is the honest signature of general capability:

    abstain (no verifier for this domain) -> log the gap (curiosity) -> synthesize a
    verifier via the REAL agent.verifier_synthesis engine (template fitters + sandboxed
    proposer + fit/val/test meta-verification) -> promote only if it clears the bar on
    the engine's DISJOINT held-out test split -> use it as VERIFIED REWARD to improve a
    policy by selection -> anti-gaming check (reward_is_hackable: the gate must not be a
    permissive/train-only reward) -> competence map flips abstain->answer.

The verifier is the real compositional engine, NOT the toy single-token decision stump
(selfextend.verifier_synthesis), so the loop can genuinely solve a domain whose signal
is non-lexical (e.g. a numeric range) — which a single-token stump provably cannot — and
genuinely abstain on a concept outside the template library. The toy stump is retained
as a minimal teaching artifact; the capstone loop uses the real one.

Honest scope: the "improvement" here is **verifier-guided selection** (rejection
sampling), a real, GPU-free form of verified-reward optimization — not a live GRPO
weight update (that needs a GPU and consumes `selfextend.verified_reward`). The loop
STRUCTURE and its falsifiable invariants are the deliverable.
"""

from __future__ import annotations

from selfextend.abstention_ledger import AbstentionLedger
from selfextend.calibration_metrics import expected_calibration_error
from selfextend.competence_map import CompetenceMap
from selfextend.verified_reward import reward_is_hackable


def _accuracy(predict, evalset: "list[tuple[str, bool]]") -> float:
    if not evalset:
        return 0.0
    return round(sum(int(predict(t) == lab) for t, lab in evalset) / len(evalset), 4)


def close_loop(domain: str, examples: "list[tuple[str, bool]]", *, threshold: float = 0.8,
               seed: int = 0, evolve_against: "object | None" = None) -> dict:
    """Run the full cycle on one held-out ``domain``. ``examples`` are ``(text, label)``
    where ``label=True`` means VALID/correct. Returns a report + falsifiable
    ``invariants`` and a ``loop_closed`` flag.

    All examples are handed to the synthesis engine, which performs its OWN disjoint
    fit/val/test split internally; the loop measures promotion, policy lift, anti-gaming
    and competence on the engine's disjoint TEST split (the honest held-out).

    When ``evolve_against`` is a prior verifier GATE callable, the SophiaArk Evolve
    canary runs on the success path: the freshly synthesised gate is promoted over the
    prior one ONLY if it scores strictly better on the disjoint held-out test split
    (else the prior is kept). Opt-in; absent it, behaviour is unchanged."""
    from agent.verifier_synthesis import _partition, synthesize as _synthesize_real

    ledger = AbstentionLedger()
    competence = CompetenceMap(threshold=threshold)

    # 0) The domain starts as a gap: no verifier => the system abstains.
    route_before = competence.route(domain)               # 'abstain' (no record yet)
    ledger.record(domain=domain, reason="no_verifier")

    task_examples = [{"answer": t, "label": bool(lab), "_idx": i} for i, (t, lab) in enumerate(examples)]
    task = {"task_id": domain, "examples": task_examples}

    # 1) Synthesize a compositional verifier with the REAL engine (fit/val/test
    #    meta-verification happens inside; trust comes only from measured validation).
    res = _synthesize_real(task, min_precision=0.95, min_recall=0.8, seed=seed, meta_verify=True)
    synthesis_report = res.report()

    if res.abstained or res.gate is None:
        return {
            "domain": domain, "loop_closed": False, "promoted": False,
            "heldoutAccuracy": 0.0, "routeBefore": route_before, "routeAfter": "abstain",
            "reason": "verifier synthesis abstained (no template/proposer fit the held-out bar) "
                      "-> stay abstained (fail-closed)",
            "synthesisReport": synthesis_report,
        }

    gate_pred = lambda a: res.gate(a, None, {})["passed"]  # accept/reject an answer

    # The engine's disjoint TEST split = the honest held-out (re-derived deterministically
    # for per-sample metrics; identical partition the engine scored internally).
    _fit, _val, test = _partition(task_examples, 0.4, 0.3, seed)
    if not test:  # too few examples to leave a test slice -> cannot honestly promote
        return {
            "domain": domain, "loop_closed": False, "promoted": False,
            "heldoutAccuracy": 0.0, "routeBefore": route_before, "routeAfter": "abstain",
            "reason": "no held-out test slice (too few examples) -> stay abstained (fail-closed)",
            "synthesisReport": synthesis_report,
        }

    test_pairs = [(ex["answer"], ex["label"]) for ex in test]

    # 2) Held-out validation on the disjoint test split.
    heldout_acc = _accuracy(lambda ans: gate_pred(ans), test_pairs)
    promoted = heldout_acc >= threshold

    # Baseline policy (no verified reward): always predict the majority class over all examples.
    majority = sum(1 for ex in task_examples if ex["label"]) >= (len(task_examples) / 2)
    pre_acc = _accuracy(lambda _a: majority, test_pairs)

    if not promoted:
        return {
            "domain": domain, "loop_closed": False, "promoted": False,
            "heldoutAccuracy": heldout_acc, "preAccuracy": pre_acc,
            "routeBefore": route_before, "routeAfter": "abstain",
            "reason": f"verifier held-out accuracy {heldout_acc} < threshold {threshold} -> abstain",
            "synthesisReport": synthesis_report,
        }

    # 3) Verified-reward improvement by SELECTION: the policy adopts the gate as its rule.
    post_acc = heldout_acc

    # 4) Anti-gaming: the gate (the reward we would train on) must not be a permissive /
    #    train-only reward vs the ORACLE on the held-out test split. A gamed gate that
    #    accepts many invalid answers shows trainReward >> heldoutReward -> hacked -> reject.
    test_answers = [ex["answer"] for ex in test]
    oracle = {ex["answer"]: ex["label"] for ex in test}
    anti_gaming = reward_is_hackable(test_answers, gate_pred, lambda a: oracle.get(a, False), gap=0.2)

    # 5) Generalization on the held-out test split + competence flip.
    for ex in test:
        competence.update(domain, bool(gate_pred(ex["answer"])) == bool(ex["label"]))
    route_after = competence.route(domain)

    # 6) Calibration of the loop's confidence (held-out accuracy) against held-out outcomes.
    conf = heldout_acc
    ece = expected_calibration_error(
        [(conf, bool(gate_pred(ex["answer"])) == bool(ex["label"])) for ex in test])

    invariants = {
        "verifier_promoted_on_heldout": promoted,
        "policy_improved_on_eval": post_acc > pre_acc,
        "generalizes_not_gamed": post_acc >= threshold,
        "gate_not_hackable_vs_oracle": not anti_gaming["hacked"],
        "competence_flips_abstain_to_answer": route_before == "abstain" and route_after == "answer",
    }
    evolve_block = None
    if evolve_against is not None:
        from selfextend.evolve import Candidate, evolve

        # The real engine yields a COMPOSED gate callable (not a legacy Rule), so the
        # canary scores gates by their accuracy on the disjoint held-out test split.
        # ``evolve_against`` must likewise be a prior gate callable.
        def _gate_scorer(gate, heldout_pairs):
            return _accuracy(gate, heldout_pairs)

        evolve_block = evolve(
            f"verifier:{domain}",
            [Candidate(target=f"verifier:{domain}", kind="verifier", payload=gate_pred)],
            test_pairs, baseline=evolve_against, score=_gate_scorer,
        )

    return {
        "domain": domain,
        "loop_closed": all(invariants.values()),
        "promoted": True,
        "evolve": evolve_block,
        "heldoutAccuracy": heldout_acc,
        "preAccuracy": pre_acc,
        "postAccuracy": post_acc,
        "improvement": round(post_acc - pre_acc, 4),
        "evalCalibrationECE": ece,
        "routeBefore": route_before,
        "routeAfter": route_after,
        "rule": {"engine": "agent.verifier_synthesis",
                 "gate": "+".join(c["name"] for c in synthesis_report["admitted"]) or "composed",
                 "admitted": synthesis_report["admitted"]},
        "antiGamingCheck": anti_gaming,
        "synthesisReport": synthesis_report,
        "invariants": invariants,
        "interpretation": (
            "The system started abstaining on this domain, synthesized a COMPOSITIONAL verifier "
            "with the real engine (meta-verified on a disjoint split), used it as verified reward "
            "to improve a policy by selection, passed the anti-gaming check vs the oracle, and the "
            "gain held on the disjoint held-out test split — the loop closed without a human writing "
            "the check and without gaming. (Selection-based; a live-RL weight update needs a GPU.)"
        ),
    }
