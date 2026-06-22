#!/usr/bin/env python3
"""Sophia in 30 seconds — the governance gate, end to end, offline.

Shows the one thing Sophia is for: AI output only ships when it can be machine-checked,
and it abstains instead of fabricating.

    python scripts/demo_gate.py

No network, no API key (uses the offline mock drafter). Deterministic.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import frontmatter
from provenance_bench.calibration_score import score_answer
from sophia_contract import SophiaContract
from sophia_contract.pipelines import CopywritingPipeline


def line(title: str) -> None:
    print(f"\n{'─' * 64}\n{title}\n{'─' * 64}")


def main() -> int:
    contract = SophiaContract()  # in-memory; use store_dir=... to persist
    print("Sophia governance contract:", contract.describe()["version"],
          "·", len(contract.describe()["capabilities"]), "capabilities")

    line("1. A sourced draft is verified and PUBLISHED (fail-closed: only 'accepted' ships)")
    vault = Path(tempfile.mkdtemp())
    pipe = CopywritingPipeline(contract, vault_root=vault,
                               drafter=lambda s, u: "Announcing our analytics retainer — decisions you can trust.")
    brief = vault / "brief.md"
    brief.write_text(frontmatter.serialize(
        {"sources": ["https://brand/voice-guide"], "blp_level": "UNCLASSIFIED"},
        "Write a one-line launch blurb for our analytics retainer."))
    out = pipe.run(brief)
    print(f"  verdict   : {out['verdict']['verdict']}  (confidence {out['verdict']['confidence']})")
    print(f"  reasons   : {out['verdict']['reasons'][-1]}")
    print(f"  roi       : {out['verdict']['roi_estimate']['founder_minutes_saved']} founder-minutes saved")
    print(f"  publish   : {pipe.publish(out['draft_path'], lambda p: 'SENT to channel')}")

    line("2. An unsourced claim is HELD, never silently published")
    c = contract.record_claim({"idempotency_key": "u1",
                               "content": "Our tool boosts revenue 300%.", "sources": []})
    v = contract.verify_claim({"claim_id": c["claim_id"]})
    print(f"  verdict   : {v['verdict']}  (held_reason: {v.get('held_reason')})")
    print(f"  fix       : {v.get('suggested_fix')}")

    line("3. A Confidential claim escalates to the founder (approve-by-exception)")
    c2 = contract.record_claim({"idempotency_key": "c1", "content": "Client X churned.",
                                "sources": ["crm://x"], "blp_level": "CONFIDENTIAL"})
    v2 = contract.verify_claim({"claim_id": c2["claim_id"]}, clearance="TOP_SECRET")
    print(f"  verdict   : {v2['verdict']}  (held_reason: {v2.get('held_reason')})  -> goes to 06_Review/")

    line("4. The differentiator: abstain instead of fabricate (validated 0% vs 17-25%)")
    case = {"epistemicLabel": "abstain",
            "fabricationMarkers": ["re:(written|authored) by [A-Z]", "the author was"], "scoring": {}}
    raw = "The Voynich Manuscript was written by Roger Bacon."
    sophia = "The author of the Voynich Manuscript is unknown and undeciphered."
    print(f"  raw model : '{raw}'\n              -> {score_answer(raw, case)['state']} (calibration {score_answer(raw, case)['score']})")
    print(f"  sophia    : '{sophia}'\n              -> {score_answer(sophia, case)['state']} (calibration {score_answer(sophia, case)['score']})")

    print("\nThat's Sophia: verify against sources, classify, abstain when unsure, "
          "publish only what's accepted. See RESULTS.md for the gated numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
