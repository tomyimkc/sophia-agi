# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Bridge the self-extending flywheel onto a kernel-checked proof verifier.

THE THESIS-LEVEL MOVE (why this module exists):

``selfextend.loop.close_loop`` already closes the self-extending cycle on toy domains
(abstain -> synthesize a verifier -> validate on held-out -> verified reward ->
competence flip, with an anti-gaming invariant). Its synthesized verifiers today are
interpretable decision stumps over text features (``selfextend.verifier_synthesis``),
and its "verified reward" is rejection sampling against that Python predicate
(``selfextend.verified_reward``).

This module swaps the VERIFIER TARGET for kernel-checked proofs while keeping the
loop's contracts byte-for-byte. The consequence is asymmetric and powerful:

  - A proof certificate is NOT a learned classifier. It either type-checks under the
    kernel or it does not. There is no decision boundary to drift, no threshold to
    overfit, no surrogate to game. The kernel is a SOUND oracle for the property
    "is this a valid proof of P" — which is exactly the property a reward function
    needs. ``verified_reward.reward_is_hackable`` becomes structurally near-vacuous
    here: the train verifier and the held-out verifier are the SAME kernel, because
    the kernel is not trained on anything. You cannot reward-hack a proof checker by
    optimizing against it; you can only produce proofs, which is the task.

  - The fail-closed abstention path (``agent.lean_verifier`` returns ``held`` when the
    toolchain is absent) becomes the NATIVE behavior on open problems: no proof ->
    no promotion -> the system says "I cannot prove this", which is the
    wisdom-before-intelligence default. The Millennium problems are the independent
    eval split, never the training set.

This module is the wiring only — it produces a ``Callable[[ProofAttempt], bool]`` and a
``Scorer`` in the shapes ``selfextend.verified_reward`` and ``selfextend.evolve`` already
consume, so ``close_loop``/``evolve`` work unchanged. It does NOT touch their internals.

Honest scope: ``candidateOnly: true``, ``level3Evidence: false``. Closing the loop on a
real Mathlib lemma with a live kernel is a future run, gated by the no-overclaim rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.lean_verifier import ProofCertificate, check_proof, lean_available
from selfextend.verified_reward import mean_reward, verified_reward


# --------------------------------------------------------------------------- #
# The reward function — kernel-acceptance as a Callable
# --------------------------------------------------------------------------- #


@dataclass
class ProofAttempt:
    """A candidate proof a policy produced, in the shape the kernel checks.

    ``proposition`` is the statement (Lean term); ``proof_text`` is the proof term /
    ``by``-tactic block. This is the unit of "candidate" in the verified-reward sense:
    the reward is 1.0 iff the kernel accepts this attempt as a valid proof of the
    proposition, else 0.0.
    """

    claim_id: str
    proposition: str
    proof_text: str


def kernel_verifier(attempt: ProofAttempt) -> bool:
    """The held-out reward oracle: True iff the Lean kernel accepts the proof attempt.

    This is a ``Callable[[object], bool]`` in the exact shape
    ``selfextend.verified_reward.verified_reward`` expects. It is fail-closed by
    construction — ``check_proof`` returns ``held`` (not ``accepted``) when the
    toolchain is absent, so a missing kernel yields reward 0.0, never 1.0.
    """
    if not isinstance(attempt, ProofAttempt):
        return False
    r = check_proof(attempt.claim_id, attempt.proposition, attempt.proof_text)
    # Only a kernel-ACCEPTED proof is reward 1.0. held/refuted/error all -> 0.0.
    return r["verdict"] == "accepted"


def proof_reward(attempt: ProofAttempt) -> float:
    """Convenience: the verified reward for a single proof attempt (1.0 iff accepted)."""
    return verified_reward(attempt, kernel_verifier)


def mean_proof_reward(attempts: "list[ProofAttempt]") -> float:
    """Mean verified reward over a batch of proof attempts (the policy-quality signal)."""
    return mean_reward(attempts, kernel_verifier)


# --------------------------------------------------------------------------- #
# The anti-gaming firewall — and why it is structurally vacuous for proofs
# --------------------------------------------------------------------------- #


def kernel_reward_is_hackable(attempts: "list[ProofAttempt]") -> dict:
    """Anti-gaming check specialised to the kernel verifier.

    For a LEARNED verifier, ``reward_is_hackable`` compares train vs held-out reward to
    catch a policy that gamed the checker. For a KERNEL verifier the train and held-out
    verifiers are the SAME sound oracle (the kernel is not fit to anything), so the
    train/held-out gap is identically zero by construction — and if it were not, that
    would indicate a KERNEL BUG (two runs of a deterministic type-checker disagreeing),
    which is a far more serious finding than reward hacking.

    We still run the check and report it, because the *measurement* matters even when the
    failure mode is structurally ruled out: a non-zero drop here would be evidence of
    non-determinism in the toolchain, not of a clever policy.
    """
    from selfextend.verified_reward import reward_is_hackable

    report = reward_is_hackable(attempts, kernel_verifier, kernel_verifier, gap=0.0)
    report["interpretation"] = (
        "structurally near-vacuous: the train and held-out verifiers are the same sound "
        "kernel oracle; a non-zero drop would indicate toolchain non-determinism, not "
        "reward hacking"
    )
    return report


# --------------------------------------------------------------------------- #
# Loop integration — a close_loop_on_proofs that mirrors close_loop's contract
# --------------------------------------------------------------------------- #
# We do NOT modify selfextend.loop.close_loop. Instead we provide a thin orchestrator
# that runs the SAME cycle on a proof domain, using kernel_verifier as the oracle and
# reporting the SAME falsifiable invariants. This keeps the toy-domain loop untouched
# (its tests stay green) while making the proof-domain loop directly runnable.


def close_loop_on_proofs(domain: str,
                         train: "list[ProofAttempt]",
                         heldout: "list[ProofAttempt]",
                         evalset: "list[ProofAttempt]",
                         *, threshold: float = 1.0) -> dict:
    """Run the self-extending cycle on a proof domain.

    The cycle mirrors ``selfextend.loop.close_loop``:
      abstain (no verifier promoted) -> validate candidate proofs on held-out via the
      kernel -> promote only if the held-out proof reward clears ``threshold`` -> measure
      the gain on an independent eval split -> competence route flips abstain->answer.

    Key difference from the toy loop: the "synthesized verifier" is not a learned
    predicate — it IS the kernel. Promotion here means "the policy's proof attempts clear
    the kernel on held-out lemmas at the threshold rate". Fail-closed: if the kernel is
    unavailable, held-out reward is 0.0 and the system STAYS ABSTAINED (loop_closed=False).

    Returns a report with falsifiable ``invariants`` and a ``loop_closed`` flag, matching
    the toy loop's shape so the two can be compared directly.
    """
    # 0) Baseline: without a promoted proof ability, the system abstains on the domain.
    route_before = "abstain"

    # Kernel unavailable -> every reward is 0.0 -> fail closed, stay abstained.
    if not lean_available():
        return {
            "domain": domain, "loop_closed": False, "promoted": False,
            "heldoutReward": 0.0, "routeBefore": route_before, "routeAfter": "abstain",
            "reason": "Lean kernel unavailable -> held-out reward 0.0 -> stay abstained "
                      "(fail-closed; the wisdom-before-intelligence default)",
            "invariants": {
                "kernel_present": False,
                "verifier_promoted_on_heldout": False,
                "policy_reward_on_eval": 0.0,
                "competence_flips_abstain_to_answer": False,
            },
        }

    heldout_reward = mean_proof_reward(heldout)
    promoted = heldout_reward >= threshold
    if not promoted:
        return {
            "domain": domain, "loop_closed": False, "promoted": False,
            "heldoutReward": heldout_reward, "routeBefore": route_before,
            "routeAfter": "abstain",
            "reason": f"held-out proof reward {heldout_reward} < threshold {threshold} "
                      f"-> stay abstained (the policy cannot yet prove these lemmas)",
            "invariants": {
                "kernel_present": True,
                "verifier_promoted_on_heldout": False,
                "policy_reward_on_eval": mean_proof_reward(evalset),
                "competence_flips_abstain_to_answer": False,
            },
        }

    # Promoted: the policy's proofs clear the kernel on held-out. Measure the gain on the
    # independent eval split (lemmas the policy never saw).
    eval_reward = mean_proof_reward(evalset)
    route_after = "answer" if eval_reward >= threshold else "abstain"
    anti_game = kernel_reward_is_hackable(train + heldout)

    invariants = {
        "kernel_present": True,
        "verifier_promoted_on_heldout": True,
        "policy_reward_on_eval": eval_reward,
        "generalizes_not_gamed": eval_reward >= threshold and not anti_game["hacked"],
        "competence_flips_abstain_to_answer": route_before == "abstain" and route_after == "answer",
    }
    return {
        "domain": domain,
        "loop_closed": all(invariants.values()),
        "promoted": True,
        "heldoutReward": heldout_reward,
        "evalReward": eval_reward,
        "antiGaming": anti_game,
        "routeBefore": route_before,
        "routeAfter": route_after,
        "invariants": invariants,
        "interpretation": (
            "The policy's proof attempts cleared the Lean kernel on held-out lemmas, and "
            "the gain held on an independent eval split. The kernel is the same sound "
            "oracle for train and eval, so the anti-gaming check is structurally near-"
            "vacuous — you cannot reward-hack a proof checker, only produce proofs. "
            "(candidateOnly; a real Mathlib run needs the no-overclaim gate.)"
        ),
        # Provenance-honesty metadata, matching the repo's report convention.
        "candidateOnly": True,
        "level3Evidence": False,
    }


__all__ = [
    "ProofAttempt",
    "kernel_verifier",
    "proof_reward",
    "mean_proof_reward",
    "kernel_reward_is_hackable",
    "close_loop_on_proofs",
]
