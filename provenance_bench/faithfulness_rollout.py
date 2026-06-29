# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Retrieve-then-reason rollout harness with the counterfactual citation-drop test.

Produces the machine-checkable *trajectory* that
``provenance_bench.retrieval_faithfulness.reward_for_trajectory`` scores. The
faithfulness term needs a MODEL — it regenerates the answer with a claim's
supporting chunk removed and checks whether the claim survives — so the rollout
is where that extra inference happens; the reward itself stays a deterministic
function of the resulting trajectory.

Injectable seams (so the SAME harness runs offline-mock and live-GPU):

  retrieve(query) -> list[Chunk]               the wiki/RAG retriever (agent.ai_search, live)
  generate(prompt, context_chunks) -> str      the policy under training (ABSTAIN to abstain)
  extract_claims(answer, context_chunks) -> list   atomic-claim decomposition + citation
  verify_claim(claim, context_chunks) -> verdict   entailment vs in-context chunks (source_verifier, live)
  check_correct(answer, gold) -> bool          optional task-success verifier (execution_verifiers, live)

A Chunk is a dict ``{chunk_id, text, author_confidence}`` — the same provenance
``agent.retrieval`` already carries forward from OKF frontmatter. The offline
path (``offline_invariants``) wires deterministic mock seams so the rollout ->
reward invariants are CI-checkable with no torch / GPU / corpus, exactly like
``tools/run_rlvr.py --model mock``. The live rollout-driven GRPO loop (sampling =
this rollout, advantage over a group of rollouts) is OPEN in the failure ledger;
this module ships the harness + its offline invariants, not a trained model.
"""

from __future__ import annotations

from typing import Any, Callable

from provenance_bench.retrieval_faithfulness import (
    REWARD_MAX,
    REWARD_MIN,
    reward_for_trajectory,
)

# Sentinel a policy returns to abstain (fail-closed) instead of answering.
ABSTAIN = "<ABSTAIN>"


def _claim_key(claim: dict) -> str:
    """Stable identity for matching a claim across the ablation regeneration. A
    live ``extract_claims`` should emit a ``key`` (e.g. a (subject,predicate,object)
    tuple rendered to text); we fall back to normalized text."""
    return str(claim.get("key") or claim.get("text", "")).strip().lower()


def _claim_present(claim: dict, re_claims: list) -> bool:
    """True if an equivalent claim reappears after its support was dropped — i.e.
    the model emitted it WITHOUT the retrieved evidence (a weights leak)."""
    target = _claim_key(claim)
    return any(_claim_key(rc) == target for rc in re_claims)


def rollout(
    case: dict,
    *,
    retrieve: Callable[[str], list],
    generate: Callable[[str, list], str],
    extract_claims: Callable[[str, list], list],
    verify_claim: Callable[[dict, list], str],
    check_correct: Callable[[str, Any], bool] | None = None,
    do_ablation: bool = True,
) -> dict:
    """Run one retrieve-then-reason rollout and return the trajectory dict that
    ``reward_for_trajectory`` consumes.

    ``case`` carries the prompt and the gold decision labels:
      prompt          : str
      should_retrieve : bool   gold — was retrieval needed (vs settled-answerable)?
      answerable      : bool   gold — is the case answerable from the wiki?
      gold            : Any    optional — task-success key for ``check_correct``
    """
    query = case["prompt"]
    n_retrievals = {"n": 0}

    def _retrieve(q: str) -> list:
        n_retrievals["n"] += 1
        return retrieve(q) or []

    chunks = _retrieve(query)
    context = {c["chunk_id"]: c for c in chunks}
    did_retrieve = bool(chunks)

    answer = generate(query, list(context.values()))

    if answer == ABSTAIN:
        return {
            "prompt": query,
            "abstained": True,
            "answerable": bool(case.get("answerable", True)),
            "did_retrieve": did_retrieve,
            "n_retrievals": n_retrievals["n"],
            "should_retrieve": case.get("should_retrieve"),
            "claims": [],
            "answer_text": "",
        }

    raw_claims = extract_claims(answer, list(context.values())) or []
    claims: list[dict] = []
    for rc in raw_claims:
        support_ids = list(rc.get("support_chunk_ids", []) or [])
        support_conf = [context[i]["author_confidence"] for i in support_ids if i in context]
        claim = {
            "text": rc.get("text", ""),
            "key": rc.get("key"),
            "kind": rc.get("kind", "knowledge"),
            "verdict": verify_claim(rc, list(context.values())),
            "support_chunk_ids": support_ids,
            "support_confidences": support_conf,
            "asserted_confidence": rc.get("asserted_confidence"),
            "survives_ablation": False,
        }
        claims.append(claim)

    # --- The counterfactual citation-drop test (the faithfulness signal). ---
    # For each SUPPORTED knowledge claim, drop its supporting chunk(s), regenerate,
    # and see whether the claim reappears. Reappearing == it came from the weights.
    if do_ablation:
        for claim in claims:
            if (claim["kind"] == "knowledge" and claim["verdict"] == "supported"
                    and claim["support_chunk_ids"]):
                drop = set(claim["support_chunk_ids"])
                ablated = [c for cid, c in context.items() if cid not in drop]
                re_answer = generate(query, ablated)
                re_claims = [] if re_answer == ABSTAIN else (extract_claims(re_answer, ablated) or [])
                claim["survives_ablation"] = _claim_present(claim, re_claims)

    task_correct = None
    if check_correct is not None and "gold" in case:
        task_correct = bool(check_correct(answer, case["gold"]))

    return {
        "prompt": query,
        "abstained": False,
        "answerable": bool(case.get("answerable", True)),
        "task_correct": task_correct,
        "claims": claims,
        "retrieved_ids": list(context.keys()),
        "context_ids": list(context.keys()),
        "should_retrieve": case.get("should_retrieve"),
        "did_retrieve": did_retrieve,
        "n_retrievals": n_retrievals["n"],
        "answer_text": answer,
    }


# --------------------------------------------------------------------------- #
# Offline mock world — deterministic seams so the rollout -> reward invariants
# are CI-checkable with no torch / GPU / corpus (mirrors run_rlvr._RECORDS).
# --------------------------------------------------------------------------- #

_WIKI = {
    "c1": {"chunk_id": "c1", "author_confidence": "attributed",
           "text": "The Project Phoenix Charter was written by the founding committee."},
    "c2": {"chunk_id": "c2", "author_confidence": "legendary",
           "text": "Some legends credit Alice with drafting the Charter."},
}


def _mock_retrieve(query: str) -> list:
    # Returns the whole tiny wiki; the policy decides what to use.
    return [_WIKI["c1"], _WIKI["c2"]]


def _mock_extract(answer: str, context: list) -> list:
    """One knowledge claim per canned answer, keyed on the asserted author."""
    low = answer.lower()
    if "founding committee" in low:
        return [{"text": "The Charter was written by the founding committee.",
                 "key": "author=committee", "kind": "knowledge",
                 "support_chunk_ids": ["c1"], "asserted_confidence": "attributed"}]
    if "alice" in low:
        return [{"text": "The Charter was written by Alice.",
                 "key": "author=alice", "kind": "knowledge",
                 "support_chunk_ids": ["c2"], "asserted_confidence": "consensus"}]
    return []


def _mock_verify(claim: dict, context: list) -> str:
    """Supported iff a context chunk entails the claim; the Alice claim is the
    legend the wiki does not assert (treated as contradicted by c1's attribution)."""
    texts = " ".join(c["text"].lower() for c in context)
    if claim.get("key") == "author=committee":
        return "supported" if "founding committee" in texts else "unsupported"
    if claim.get("key") == "author=alice":
        return "contradicted" if "founding committee" in texts else "unsupported"
    return "unsupported"


def _faithful_policy(query: str, context: list) -> str:
    """Uses retrieval: asserts the committee only when c1 is in context, else abstains."""
    if any(c["chunk_id"] == "c1" for c in context):
        return "The Project Phoenix Charter was written by the founding committee."
    return ABSTAIN


def _leaky_policy(query: str, context: list) -> str:
    """Ignores retrieval: always asserts the committee (so the claim survives the drop)."""
    return "The Project Phoenix Charter was written by the founding committee."


def _check_correct(answer: str, gold: Any) -> bool:
    return str(gold).lower() in answer.lower()


def offline_invariants() -> tuple[bool, dict]:
    """Assert the rollout -> reward invariants (no torch, no GPU, no corpus).

    Proves: a retrieval-USING policy is scored above a weights-LEAKING one on an
    identical answer (the faithfulness signal works), correct fail-closed
    abstention is rewarded, a wiki-contradicted claim hits the hard floor, the
    rollout is deterministic, and every reward is bounded.
    """
    answerable_case = {"prompt": "Who wrote the Project Phoenix Charter?",
                       "should_retrieve": True, "answerable": True, "gold": "founding committee"}
    unanswerable_case = {"prompt": "What did the Charter's authors eat for lunch?",
                         "should_retrieve": True, "answerable": False}

    common = dict(retrieve=_mock_retrieve, extract_claims=_mock_extract,
                  verify_claim=_mock_verify, check_correct=_check_correct)

    traj_faithful = rollout(answerable_case, generate=_faithful_policy, **common)
    traj_leaky = rollout(answerable_case, generate=_leaky_policy, **common)
    traj_faithful_2 = rollout(answerable_case, generate=_faithful_policy, **common)
    traj_abstain = rollout(unanswerable_case, generate=lambda q, c: ABSTAIN, **common)
    # An Alice-asserting policy makes a claim the wiki contradicts -> hard floor.
    traj_contra = rollout(
        answerable_case,
        generate=lambda q, c: "The Charter was written by Alice.", **common,
    )

    r_faithful, d_faithful = reward_for_trajectory(traj_faithful)
    r_leaky, d_leaky = reward_for_trajectory(traj_leaky)
    r_faithful_2, _ = reward_for_trajectory(traj_faithful_2)
    r_abstain, _ = reward_for_trajectory(traj_abstain)
    r_contra, d_contra = reward_for_trajectory(traj_contra)

    # The committee claim must FLIP under ablation for the faithful policy (it
    # abstains when c1 is dropped) and SURVIVE for the leaky one.
    faithful_claim = traj_faithful["claims"][0]
    leaky_claim = traj_leaky["claims"][0]

    checks = {
        "faithfulBeatsLeaky": r_faithful > r_leaky,
        "faithfulClaimFlips": faithful_claim["survives_ablation"] is False,
        "leakyClaimSurvives": leaky_claim["survives_ablation"] is True,
        "deterministic": r_faithful == r_faithful_2,
        "correctAbstentionPositive": r_abstain > 0.0,
        "contradictedIsFloor": r_contra == REWARD_MIN,
        "bounded": all(REWARD_MIN <= r <= REWARD_MAX
                       for r in (r_faithful, r_leaky, r_abstain, r_contra)),
    }
    detail = {
        "checks": checks,
        "rewards": {
            "faithful": d_faithful, "leaky": d_leaky, "contradicted": d_contra,
            "abstain": round(r_abstain, 4),
        },
        "faithfulSurvivesAblation": faithful_claim["survives_ablation"],
        "leakySurvivesAblation": leaky_claim["survives_ablation"],
    }
    return all(checks.values()), detail


__all__ = ["rollout", "offline_invariants", "ABSTAIN"]
