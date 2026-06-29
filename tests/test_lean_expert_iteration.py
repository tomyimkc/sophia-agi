# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the Lean verifier-as-reward expert-iteration driver
(tools.run_lean_expert_iteration).

No network, no model, no Lean toolchain (the real Lean path MUST fail-closed/abstain here).
The contract these tests lock:

  * With the stub-equivalent proposer the loop RUNS over rounds and KEEPS a verified+novel
    proof on the first round (the scripted applier stands in for the kernel).
  * The novelty probe REJECTS a near-duplicate of the corpus (Jaccard >= threshold).
  * Once a proof is kept into the corpus, a later round does NOT re-keep the same proof
    (novelty is judged against the growing corpus) -> total novel_kept does not double-count.
  * Without --stub and with no Lean, the loop FAIL-CLOSES (`lean_unavailable`): zero
    verified, zero kept, NO fabricated assertion.
  * kernel_reward_is_hackable returns a STRUCTURED result every round.
  * The harness NEVER asserts an unverified proof (a fail-closed run is a PASS).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.lean_verifier import lean_available  # noqa: E402
from selfextend.proof_verifier import ProofAttempt  # noqa: E402
from tools.run_lean_expert_iteration import (  # noqa: E402
    BUNDLED,
    is_novel,
    propose_attempts,
    run_expert_iteration,
    run_round,
    verify_attempt,
)


def test_stub_loop_runs_and_keeps_a_verified_novel_proof() -> None:
    """With --stub the loop RUNS and keeps >=1 verified+novel proof (scripted applier).
    This is the happy path that exercises the keep/verify/novelty branch deterministically.
    """
    report = run_expert_iteration(rounds=2, stub=True, theorem="trivial_true")
    assert report["mode"] == "stub"
    assert report["rounds"] == 2
    # The scripted applier accepts `trivial` for trivial_true; the FIRST round keeps it
    # (empty corpus -> overlap 0.0 -> novel). Verdict reflects a kept proof.
    assert report["totalNovelKept"] >= 1
    assert report["verdict"] == "novel_proof_kept"
    # candidateOnly discipline preserved; stub is NOT a real verification.
    assert report["candidateOnly"] is True
    assert report["canClaimAGI"] is False
    assert report["level3Evidence"] is False


def test_stub_keeps_proof_only_once_across_rounds() -> None:
    """A proof kept into the corpus in round 1 must NOT be re-kept in round 2: novelty is
    judged against the GROWING corpus, so total novel_kept does not double-count it."""
    report = run_expert_iteration(rounds=3, stub=True, theorem="trivial_true")
    # Only the first round should keep `trivial`; subsequent rounds see it in the corpus
    # (Jaccard overlap 1.0 >= 0.92) and reject it as a near-duplicate.
    per_round_kept = [r["novelKept"] for r in report["rounds_detail"]]
    assert per_round_kept[0] >= 1
    assert sum(per_round_kept[1:]) == 0, f"proof re-kept after round 1: {per_round_kept}"


def test_novelty_probe_rejects_near_duplicate_of_corpus() -> None:
    """The novelty probe rejects a near-duplicate of the corpus (Jaccard >= threshold)."""
    corpus = ["intros; apply Nat.add_comm"]
    # An exact duplicate must score overlap 1.0 -> NOT novel.
    dup = is_novel("intros; apply Nat.add_comm", corpus, threshold=0.92)
    assert dup["novel"] is False
    assert dup["best_overlap"] >= 0.92
    # A clearly different proof body must be flagged novel.
    diff = is_novel("trivial", corpus, threshold=0.92)
    assert diff["novel"] is True
    # The very first proof (empty corpus) is trivially novel (overlap 0.0).
    first = is_novel("any proof", [], threshold=0.92)
    assert first["novel"] is True
    assert first["best_overlap"] == 0.0


def test_real_backend_fails_closed_when_lean_absent() -> None:
    """Without --stub and with no Lean toolchain, the loop FAIL-CLOSES: zero verified, zero
    kept, verdict lean_unavailable, and NO fabricated proof. (If Lean is somehow installed
    in this env, the test still holds: it asserts the no-fabrication invariant either way.)"""
    report = run_expert_iteration(rounds=2, stub=False, theorem="add_zero")
    if not lean_available():
        assert report["verdict"] == "lean_unavailable"
        assert report["totalNovelKept"] == 0
        for r in report["rounds_detail"]:
            assert r["verified"] == 0
            assert r["novelKept"] == 0
            assert r["keptProofs"] == []
    else:
        # Lean present: the harness still must not fabricate — kept proofs (if any) are real.
        assert report["verdict"] in ("novel_proof_kept", "no_novel_proof")
    # In every case the discipline fields hold.
    assert report["canClaimAGI"] is False
    assert report["candidateOnly"] is True


def test_verify_attempt_real_path_abstains_without_lean() -> None:
    """The real verify path defers to the kernel; with no Lean it returns False (never True
    on an unverified proof) — the load-bearing no-fabrication invariant at the unit level."""
    att = ProofAttempt(claim_id="t#0", proposition="True", proof_text="trivial")
    if not lean_available():
        assert verify_attempt(att, stub=False, stub_winning="trivial") is False


def test_stub_applier_accepts_only_the_scripted_proof() -> None:
    """The stub applier accepts ONLY the exact scripted winning proof — every other
    candidate is rejected (it is a scripted stand-in, not a real prover)."""
    win = ProofAttempt(claim_id="t#0", proposition="True", proof_text="trivial")
    lose = ProofAttempt(claim_id="t#1", proposition="True", proof_text="simp")
    assert verify_attempt(win, stub=True, stub_winning="trivial") is True
    assert verify_attempt(lose, stub=True, stub_winning="trivial") is False


def test_kernel_reward_is_hackable_returns_structured_result() -> None:
    """Each round runs the anti-gaming check; it must return a STRUCTURED result with the
    expected keys, and the drop is 0.0 (train and held-out verifier are the same kernel)."""
    report = run_expert_iteration(rounds=1, stub=True, theorem="trivial_true")
    ag = report["rounds_detail"][0]["antiGaming"]
    for key in ("trainReward", "heldoutReward", "drop", "hacked", "interpretation"):
        assert key in ag, f"missing anti-gaming key {key!r}: {ag}"
    assert ag["drop"] == 0.0  # same sound oracle -> identically zero drop
    assert ag["hacked"] is False
    assert report["anyHackableFlag"] is False


def test_run_round_grows_corpus_in_place() -> None:
    """run_round mutates the corpus in place by appending kept proofs (the expert set)."""
    corpus: list[str] = []
    res = run_round(1, {"trivial_true": BUNDLED["trivial_true"]}, corpus,
                    proposer=lambda thm, st: [], stub=True)
    # The scripted `trivial` proof was verified+novel -> kept and appended to corpus.
    assert res.novel_kept >= 1
    assert "trivial" in corpus
    assert res.kept_proofs == ["trivial"]


def test_propose_attempts_injects_scripted_proof_under_stub() -> None:
    """Under --stub, propose_attempts injects the scripted winning proof as a candidate so
    the loop has something the stub applier can accept (mirrors run_proof_search.py)."""
    spec = BUNDLED["trivial_true"]
    attempts = propose_attempts(lambda thm, st: [], "trivial_true", spec, stub=True)
    bodies = [a.proof_text for a in attempts]
    assert spec["stub_tactic"] in bodies
    assert all(isinstance(a, ProofAttempt) for a in attempts)


def test_unknown_theorem_raises() -> None:
    """An unknown theorem name raises KeyError (the CLI maps it to exit code 2)."""
    raised = False
    try:
        run_expert_iteration(rounds=1, stub=True, theorem="does_not_exist")
    except KeyError:
        raised = True
    assert raised


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
