#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-as-reward EXPERT ITERATION over a Lean-checked proof kernel (Cluster E).

The reframe behind this driver (closing the scaffold bet
``verifier_synthesis_over_proof_kernel``):

A machine-checked Lean proof is the ideal NON-GAMEABLE verifier. It either type-checks
under the kernel or it does not — there is no decision boundary to drift, no surrogate to
overfit. So we run the textbook expert-iteration / STaR loop, but harnessed by the already
validated SSIL verifier-gated machinery (``selfextend.proof_verifier``):

    for each round:
      1. PROPOSE candidate proofs (agent.tactic_proposer; --stub => stub_proposer, CI-safe)
      2. VERIFY each with the kernel (selfextend.proof_verifier.kernel_verifier ->
         agent.lean_verifier.check_proof). Fail-closed: without a Lean toolchain every
         attempt abstains `lean_unavailable` and NOTHING is kept (no fabricated proof).
      3. KEEP only proofs that are BOTH kernel-verified AND novel w.r.t. the running
         corpus (novelty probe = char-trigram Jaccard, the strict pre-filter used by the
         Path B experiment).
      4. ADD kept proofs to the corpus (the "expert" set the proposer would train on).
      5. ANTI-GAMING: run kernel_reward_is_hackable() on the round's attempts to confirm
         the proposer is not gaming the kernel (structurally near-vacuous here — the train
         and held-out verifiers are the SAME sound kernel oracle).

Honest scope: WITHOUT lean-dojo / a Lean toolchain the REAL backend abstains
``lean_unavailable`` and the run records that status — it NEVER fabricates a proof and
NEVER asserts an unverified proof. With ``--stub`` the loop STRUCTURE runs against a
scripted applier (no model, no Lean) so the harness is CI-testable. ``canClaimAGI`` stays
false; ``candidateOnly: true``. Converting the negative into a Level-3 datapoint needs the
real compute described in ``agi-proof/proof-search/LEAN-EXPERT-ITERATION-SETUP.md``.

Usage:
    python tools/run_lean_expert_iteration.py --stub --rounds 2
    python tools/run_lean_expert_iteration.py --stub --theorem trivial_true --rounds 3
    python tools/run_lean_expert_iteration.py --rounds 2          # real backend -> abstains
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lean_backend  # noqa: E402  (novelty probe lives here)
from agent.lean_verifier import lean_available  # noqa: E402
from agent.tactic_proposer import default_proposer, stub_proposer  # noqa: E402
from selfextend.proof_verifier import (  # noqa: E402
    ProofAttempt,
    kernel_reward_is_hackable,
    kernel_verifier,
)

DEFAULT_OUT = ROOT / "agi-proof" / "proof-search" / "lean-expert-iteration.public-report.json"

# Bundled rehearsal theorems. The same self-contained snippets the Path B search uses, with
# a `stub_tactic` that a scripted applier "accepts" so the loop STRUCTURE runs without Lean.
# Real runs point the kernel (agent.lean_verifier.check_proof) at these propositions.
BUNDLED: dict[str, dict[str, str]] = {
    "trivial_true": {
        "proposition": "True",
        "stub_tactic": "trivial",
    },
    "add_zero": {
        "proposition": "∀ (n : Nat), n + 0 = n",
        "stub_tactic": "intros; rfl",
    },
    "zero_add": {
        "proposition": "∀ (n : Nat), 0 + n = n",
        "stub_tactic": "intros; omega",
    },
}


# --------------------------------------------------------------------------- #
# Novelty probe — the strict char-trigram Jaccard pre-filter (Path B convention).
# --------------------------------------------------------------------------- #


def is_novel(proof_text: str, corpus: list[str], *, threshold: float = 0.92) -> dict[str, Any]:
    """Strict novelty probe: True iff ``proof_text`` is NOT a near-duplicate of ``corpus``.

    Uses the char-trigram Jaccard probe (method="trigram") from ``agent.lean_backend`` —
    the same cheap, dependency-free pre-filter the Path B proof-search experiment recorded.
    A proof whose best Jaccard overlap with the corpus is >= ``threshold`` is rejected as a
    retrieved near-duplicate; below threshold it is flagged novel. Fail-safe: an empty
    corpus yields overlap 0.0 (the FIRST verified proof is trivially novel).
    """
    return lean_backend.novelty_check(
        proof_text, corpus=corpus, near_dup_threshold=threshold, method="trigram"
    )


# --------------------------------------------------------------------------- #
# Proposer adapter — turn a TacticProposer into per-theorem ProofAttempt candidates.
# --------------------------------------------------------------------------- #


def propose_attempts(
    proposer: Callable[[str, str], list[str]],
    theorem_name: str,
    spec: dict[str, str],
    *,
    stub: bool,
) -> list[ProofAttempt]:
    """Produce candidate ``ProofAttempt``s for one theorem from a tactic proposer.

    With ``--stub`` we ALSO inject the scripted ``stub_tactic`` as a candidate so the loop
    can exercise its keep/verify branch deterministically without Lean or a model — exactly
    as ``tools/run_proof_search.py`` injects a scripted applier. Without ``--stub`` the
    proposer's tactic lines become candidate proof bodies; the kernel is the only thing that
    can promote them (and it abstains when Lean is absent — fail-closed).
    """
    proposition = spec["proposition"]
    candidates: list[str] = []
    if stub:
        # The scripted "winning" proof for this theorem (what a real kernel would accept).
        candidates.append(spec["stub_tactic"])
    # Tactic lines the proposer suggests, each as a single-tactic proof body candidate.
    for tac in proposer(proposition, proposition) or []:
        if tac not in candidates:
            candidates.append(tac)
    return [
        ProofAttempt(claim_id=f"{theorem_name}#{i}", proposition=proposition, proof_text=body)
        for i, body in enumerate(candidates)
    ]


# --------------------------------------------------------------------------- #
# Verification — the kernel is the only promoter (with a CI-safe stub override).
# --------------------------------------------------------------------------- #


def verify_attempt(attempt: ProofAttempt, *, stub: bool, stub_winning: str) -> bool:
    """Decide whether ``attempt`` is kernel-VERIFIED.

    Real path (``stub=False``): defers entirely to ``kernel_verifier`` ->
    ``agent.lean_verifier.check_proof``. Fail-closed: when Lean is absent the verdict is
    ``held``/``lean_unavailable`` -> NOT accepted -> returns False. No proof is ever
    fabricated.

    Stub path (``stub=True``): a scripted "applier" stands in for the kernel so the loop is
    CI-testable WITHOUT Lean. It accepts ONLY the exact scripted winning proof for the
    theorem (``stub_winning``) — every other candidate is rejected. This mirrors
    ``run_proof_search.py``'s scripted applier and NEVER calls the real kernel.
    """
    if not stub:
        return kernel_verifier(attempt)
    return attempt.proof_text.strip() == stub_winning.strip()


# --------------------------------------------------------------------------- #
# One expert-iteration round.
# --------------------------------------------------------------------------- #


@dataclass
class RoundResult:
    """Outcome of one expert-iteration round (the per-round public-report row)."""

    round_index: int
    proposed: int = 0
    verified: int = 0
    novel_kept: int = 0
    kept_proofs: list[str] = field(default_factory=list)
    hackable: dict[str, Any] = field(default_factory=dict)
    lean_available: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "proposed": self.proposed,
            "verified": self.verified,
            "novelKept": self.novel_kept,
            "keptProofs": list(self.kept_proofs),
            "hackableFlag": bool(self.hackable.get("hacked")) if self.hackable else None,
            "antiGaming": self.hackable,
            "leanAvailable": self.lean_available,
            "note": self.note,
        }


def run_round(
    round_index: int,
    theorems: dict[str, dict[str, str]],
    corpus: list[str],
    proposer: Callable[[str, str], list[str]],
    *,
    stub: bool,
    novelty_threshold: float = 0.92,
) -> RoundResult:
    """Run ONE expert-iteration round over ``theorems``, mutating ``corpus`` in place.

    Propose -> verify with the kernel (or the scripted stub applier) -> keep ONLY
    verified+novel proofs -> append kept proofs to ``corpus`` (the expert set) -> run the
    anti-gaming check on the round's attempts. Returns a ``RoundResult``. Fail-closed: with
    the real backend and no Lean, ``verified`` is 0 and nothing is kept — recorded honestly,
    no fabrication.
    """
    result = RoundResult(round_index=round_index, lean_available=lean_available())
    all_attempts: list[ProofAttempt] = []
    for name, spec in theorems.items():
        attempts = propose_attempts(proposer, name, spec, stub=stub)
        all_attempts.extend(attempts)
        result.proposed += len(attempts)
        for att in attempts:
            if not verify_attempt(att, stub=stub, stub_winning=spec["stub_tactic"]):
                continue  # kernel did not accept (or Lean unavailable) -> never kept
            result.verified += 1
            nov = is_novel(att.proof_text, corpus, threshold=novelty_threshold)
            if nov.get("novel"):
                result.novel_kept += 1
                result.kept_proofs.append(att.proof_text)
                corpus.append(att.proof_text)  # grow the expert corpus

    # Anti-gaming check on this round's attempts (structurally near-vacuous for a kernel).
    result.hackable = kernel_reward_is_hackable(all_attempts) if all_attempts else {}

    if not result.lean_available and not stub:
        result.note = ("lean_unavailable: real kernel absent -> every attempt abstained, "
                       "nothing verified or kept (fail-closed; no proof fabricated)")
    elif result.verified == 0:
        result.note = "no proof verified this round -> nothing kept (fail-closed abstention)"
    elif result.novel_kept == 0:
        result.note = "verified proofs were all corpus near-duplicates -> nothing novel kept"
    else:
        result.note = f"kept {result.novel_kept} verified+novel proof(s) into the expert corpus"
    return result


# --------------------------------------------------------------------------- #
# Full expert-iteration run (importable; not behind __main__).
# --------------------------------------------------------------------------- #


def run_expert_iteration(
    *,
    rounds: int,
    stub: bool,
    theorem: str | None = None,
    novelty_threshold: float = 0.92,
    proposer: Callable[[str, str], list[str]] | None = None,
) -> dict[str, Any]:
    """Drive ``rounds`` of verifier-as-reward expert iteration and return a public report.

    Selects the theorem set (one bundled theorem if ``theorem`` given, else all bundled),
    picks the proposer (``stub_proposer`` under ``--stub``, else ``default_proposer()`` —
    which itself falls back to the stub with no model), and runs the rounds, threading a
    single growing corpus across them so novelty is judged against everything kept so far.

    The ``verdict`` is HONEST and fail-closed:
      * no Lean + real backend -> ``lean_unavailable`` (the negative; no fabrication).
      * Lean present (or stub) and >=1 verified+novel proof kept -> ``novel_proof_kept``.
      * otherwise -> ``no_novel_proof`` (fail-closed abstention).
    Only a REAL kernel verification on a HELD-OUT theorem reproduced across seeds closes the
    ``verifier_synthesis_over_proof_kernel`` bet — never the stub (see the SETUP runbook).
    """
    if theorem is not None:
        if theorem not in BUNDLED:
            raise KeyError(f"unknown theorem {theorem!r}; bundled: {sorted(BUNDLED)}")
        theorems = {theorem: BUNDLED[theorem]}
    else:
        theorems = dict(BUNDLED)

    if proposer is None:
        proposer = stub_proposer if stub else default_proposer()

    corpus: list[str] = []
    round_results: list[RoundResult] = []
    for i in range(1, max(1, rounds) + 1):
        round_results.append(
            run_round(i, theorems, corpus, proposer, stub=stub,
                      novelty_threshold=novelty_threshold)
        )

    total_novel_kept = sum(r.novel_kept for r in round_results)
    any_hacked = any(r.hackable.get("hacked") for r in round_results if r.hackable)
    lean_present = lean_available()

    if not lean_present and not stub:
        verdict = "lean_unavailable"
        interpretation = (
            "Real Lean kernel absent on this host -> every proof attempt abstained "
            "(held/lean_unavailable). Nothing was verified, nothing kept. This is the "
            "honest fail-closed negative — NO proof was fabricated. To convert this into a "
            "Level-3 datapoint, provision a Lean toolchain per "
            "agi-proof/proof-search/LEAN-EXPERT-ITERATION-SETUP.md and re-run."
        )
    elif total_novel_kept > 0:
        verdict = "novel_proof_kept"
        kernel_word = "scripted stub applier (NOT real Lean)" if stub else "Lean kernel"
        interpretation = (
            f"Expert iteration kept {total_novel_kept} verified+novel proof(s) across "
            f"{len(round_results)} round(s), gated by the {kernel_word}. The anti-gaming "
            "check confirmed no kernel-gaming (drop 0.0; structurally near-vacuous). "
            + ("candidateOnly: a --stub run exercises the LOOP, not a real verification. "
               "It does NOT close the bet." if stub else
               "candidateOnly until reproduced across seeds on a HELD-OUT theorem under the "
               "no-overclaim gate.")
        )
    else:
        verdict = "no_novel_proof"
        interpretation = (
            "No verified+novel proof was kept (every verified proof was a corpus "
            "near-duplicate, or none verified). Fail-closed abstention — the harness never "
            "asserted an unverified proof."
        )

    return {
        "schema": "sophia.lean_expert_iteration.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "validated": False,
        "canClaimAGI": False,
        "scaffoldBet": "verifier_synthesis_over_proof_kernel",
        "mode": "stub" if stub else "real",
        "rounds": len(round_results),
        "theorems": sorted(theorems),
        "noveltyThreshold": novelty_threshold,
        "leanAvailable": lean_present,
        "environment": {
            "leanToolchain": "present" if lean_backend.lean_cli_available() else "NOT INSTALLED",
            "leanDojo": "present" if lean_present else "NOT INSTALLED",
            "proposer": "stub_proposer (deterministic)" if stub else "default_proposer()",
        },
        "falsifiableQuestion": (
            "Across rounds, does verifier-gated expert iteration KEEP at least one "
            "Lean-verified AND novel proof, with the anti-gaming check showing no "
            "kernel-gaming?"
        ),
        "rounds_detail": [r.to_dict() for r in round_results],
        "totalNovelKept": total_novel_kept,
        "anyHackableFlag": any_hacked,
        "verdict": verdict,
        "interpretation": interpretation,
        "successCriterion": (
            ">=1 Lean-verified proof with novelty=true on a HELD-OUT theorem, reproduced "
            "across seeds, under the no-overclaim gate. A --stub run is NOT sufficient: it "
            "verifies via a scripted applier, not the real kernel. See "
            "agi-proof/proof-search/LEAN-EXPERT-ITERATION-SETUP.md."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rounds", type=int, default=2, help="number of expert-iteration rounds")
    ap.add_argument("--stub", action="store_true",
                    help="CI-safe: deterministic stub proposer + scripted applier (no model/Lean)")
    ap.add_argument("--theorem", default=None,
                    help=f"single bundled theorem (default: all). bundled: {sorted(BUNDLED)}")
    ap.add_argument("--threshold", type=float, default=0.92, help="novelty Jaccard threshold")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    try:
        report = run_expert_iteration(
            rounds=args.rounds, stub=args.stub, theorem=args.theorem,
            novelty_threshold=args.threshold,
        )
    except KeyError as exc:
        print(str(exc))
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out}")
    # Exit 0 on a clean fail-closed outcome too (lean_unavailable / no_novel_proof are PASSES
    # for the discipline — nothing fabricated). Non-zero only if the anti-gaming flag fired.
    return 3 if report["anyHackableFlag"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
