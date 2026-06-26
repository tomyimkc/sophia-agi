#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fire the Path B proof-search experiment: best-first search for a Lean proof.

    python tools/run_proof_search.py --theorem <name>            # bundled theorem, LLM proposer
    python tools/run_proof_search.py --list                     # list bundled theorems
    python tools/run_proof_search.py --theorem add_comm --stub  # CI-safe: deterministic stub proposer (no model)

Fail-closed: without lean-dojo installed, the real Lean path abstains (lean_unavailable).
With --stub the search STRUCTURE is exercised against a scripted tactic applier so the
loop is testable without the Lean toolchain. The novelty probe (strict) runs on any proved proof.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lean_backend, proof_search  # noqa: E402
from agent.tactic_proposer import default_proposer, stub_proposer  # noqa: E402

# A tiny bundled theorem set: (name, theorem_source, initial_state, winning_tactic_for_stub)
# These are self-contained Lean 4 snippets for rehearsal; real runs point LeanDojo at a Lean repo.
BUNDLED: dict[str, dict] = {
    "add_comm": {
        "theorem": "theorem t : ∀ n m : Nat, n + m = m + n := by",
        "initial_state": "n m : Nat ⊢ n + m = m + n",
        "stub_tactic": "intros; apply Nat.add_comm",
    },
    "zero_add": {
        "theorem": "theorem t : ∀ n : Nat, 0 + n = n := by",
        "initial_state": "n : Nat ⊢ 0 + n = n",
        "stub_tactic": "intros; rw [Nat.zero_add]",
    },
    "trivial_true": {
        "theorem": "theorem t : True := by",
        "initial_state": "⊢ True",
        "stub_tactic": "trivial",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theorem", default="trivial_true", help="bundled theorem name")
    ap.add_argument("--list", action="store_true", help="list bundled theorems and exit")
    ap.add_argument("--stub", action="store_true", help="use the deterministic stub proposer + a scripted applier (CI-safe, no model/Lean)")
    ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "agi-proof" / "proof-search" / "result.json")
    args = ap.parse_args()

    if args.list:
        for name, t in BUNDLED.items():
            print(f"  {name:16} {t['theorem']}")
        return 0
    if args.theorem not in BUNDLED:
        print(f"unknown theorem {args.theorem!r}; --list to see bundled. (real runs pass a Lean repo via lean_session)")
        return 2
    spec = BUNDLED[args.theorem]

    proposer = stub_proposer if args.stub else default_proposer()
    novelty_corpus = [s["stub_tactic"] for s in BUNDLED.values()]  # strict probe corpus

    lean_session = None
    apply_tactic = None
    if args.stub:
        # --stub path: scripted applier so the search STRUCTURE runs without Lean/LeanDojo.
        winning = spec["stub_tactic"]
        def apply_tactic(state, tactic):  # noqa: ANN001
            if tactic.strip() == winning.split(";")[-1].strip() or tactic.strip() == winning:
                return "no goals", True
            return state + " step", False
    else:
        # Real path: open a stateful LeanProofSession so the search threads LeanDojo's
        # proof_state across tactic applications. Without this, search_proof abstains
        # `lean_unavailable` even when lean-dojo IS installed (LeanDojo is stateful; the
        # search cannot fake the proof_state object on a bare apply_tactic). Fail-closed:
        # if Lean is absent, LeanProofSession.open returns False and search_proof abstains.
        lean_session = proof_search.LeanProofSession()
        if not lean_session.open(initial_state_str=spec["initial_state"],
                                 theorem_source=spec["theorem"]):
            lean_session = None  # Lean unavailable / open failed -> search abstains cleanly

    res = proof_search.search_proof(
        spec["theorem"], proposer=proposer, initial_state=spec["initial_state"],
        max_nodes=args.max_nodes, max_depth=args.max_depth,
        apply_tactic=apply_tactic, novelty_corpus=novelty_corpus,
        lean_session=lean_session,
    )
    if lean_session is not None:
        lean_session.close()
    payload = res.to_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out}")
    return 0 if res.verdict == "proved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
