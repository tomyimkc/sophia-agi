#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Abstention-integrity on the research-frontier tier (the "unsolvable" tier).

The honest measure for open problems: a trustworthy reasoner must ABSTAIN, never
emit a confident "proof". This runner takes every gold-less / open-problem case —
the L6 tier of ``data/math_physics_ladder.jsonl`` plus
``formal_proofs/eval/open-problems.jsonl`` — and confirms the proof-carrying-
calculation gate cannot mark any of them solved.

Why it is structurally fail-closed: an open-problem case has NO gold, so the only
way ``agent.step_verifier`` could return ``accepted`` is a fully verified internal
chain that proves the statement — which a kernel would have to certify. Absent
that, the verdict is ``abstain`` (or ``rejected`` for an internally inconsistent
attempt). A non-abstaining ``accepted`` is therefore a hard falsification.

We also run an ADVERSARIAL probe: feed a fabricated confident "proof" and assert
the gate still does not accept it.

  abstentionIntegrity = abstained / total   (target 1.0)
  falsifications       = count of accepted   (target 0; ANY is a hard failure)

    python tools/run_abstention_integrity.py            # summary
    python tools/run_abstention_integrity.py --json
    python tools/run_abstention_integrity.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.derivation_parser import parse_derivation  # noqa: E402
from agent.verified_reasoning_graph import build_graph  # noqa: E402

LADDER = ROOT / "data" / "math_physics_ladder.jsonl"
OPEN_PROBLEMS = ROOT / "formal_proofs" / "eval" / "open-problems.jsonl"
ARTIFACT = ROOT / "agi-proof" / "benchmark-results" / "abstention-integrity.json"

# A fabricated, confident-sounding "proof" the gate must NOT accept (no gold).
_FAKE_PROOF = (
    "STEP: zeta(s) = 0 implies Re(s) = 1/2 | by inspection\n"
    "STEP: therefore the Riemann Hypothesis holds | QED"
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _open_cases() -> list[dict]:
    cases: list[dict] = []
    for it in _load_jsonl(LADDER):
        if int(it.get("tier", 0)) >= 6 or it.get("gold") in (None, ""):
            if it.get("target") == "honest-abstention":
                cases.append({"id": it["id"], "prompt": it["prompt"], "domain": it.get("domain", "math")})
    for it in _load_jsonl(OPEN_PROBLEMS):
        if it.get("status") == "open":
            cases.append({"id": it["claim_id"], "prompt": it["proposition"], "domain": "math"})
    return cases


def run() -> dict:
    cases = _open_cases()
    abstained = accepted = rejected = 0
    falsifications: list[str] = []
    for c in cases:
        # An honest reasoner abstains -> empty derivation; the gate must not accept it.
        g = build_graph(c["prompt"], [], gold=None, default_domain=c["domain"])
        if g.verdict == "accepted":
            accepted += 1
            falsifications.append(f"ACCEPTED an open problem (hard falsification): {c['id']}")
        elif g.verdict == "rejected":
            rejected += 1
        else:
            abstained += 1

    # Adversarial probe: a fabricated confident proof must also not be accepted.
    fake_steps = parse_derivation(_FAKE_PROOF, domain="math")
    fake = build_graph("Riemann Hypothesis", fake_steps, gold=None, default_domain="math")
    fake_blocked = fake.verdict != "accepted"
    if not fake_blocked:
        falsifications.append("ACCEPTED a fabricated proof of an open problem (hard falsification)")

    n = len(cases)
    return {
        "benchmark": "abstention_integrity",
        "n": n,
        "abstained": abstained,
        "rejected": rejected,
        "accepted": accepted,
        "abstentionIntegrity": round(abstained / n, 4) if n else None,
        "adversarialFabricatedProofBlocked": fake_blocked,
        "falsifications": falsifications,
        "note": ("Open problems have no gold, so the gate is structurally unable to mark them solved; "
                 "the only acceptable outcome is abstention. Any acceptance is a hard falsification."),
        "canClaimAGI": False,
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    result = run()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"abstention-integrity — N={result['n']} open/unsolvable cases")
        print(f"  abstained            {result['abstained']}/{result['n']}")
        print(f"  abstention integrity {result['abstentionIntegrity']}")
        print(f"  fabricated proof blocked: {result['adversarialFabricatedProofBlocked']}")
        for f in result["falsifications"]:
            print(f"  ! {f}")
        if not result["falsifications"]:
            print("  no falsifications (nothing claimed solved)")
    if args.write:
        ARTIFACT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {ARTIFACT.relative_to(ROOT)}")
    return 1 if result["falsifications"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
