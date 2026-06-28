#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Seal a THIRD-PARTY-authored held-out pack → public salted SHA-256 commitments only.

This is the ONLY path to a clean "external generalization" claim. The repo-authored
style-samples (eval/external/*, provenance_bench/data/*) inherit the field's
pretraining-contamination problem (a model may have seen similar items); they are
suggestive, not proof. A third-party pack — authored by someone with NO access to the
training data or prompts, sealed (salted) before any model run, revealed only after the
gated eval — closes that gap. See agi-proof/third-party-heldout/PROTOCOL.md.

Reuses tools/hidden_eval_commitments.build_commitments (salted per-case SHA-256; the
salt + unsealed prompts live under gitignored private/; only the per-case hashes are
published). A third party authors a private pack JSON of the shape:

    {"packId": "...", "salt": "<256-bit hex>",
     "cases": [{"id":"...", "domain":"math"|"code", "prompt":"...", "materials":[...],
                "scoring":{...}, "requiresToolLog":bool, "requiresMemoryDiff":bool}, ...]}

and runs:

    python tools/seal_third_party_heldout.py --private-pack private/third-party/pack.json

With no --private-pack, it writes an EMPTY commitment manifest (caseCount 0), which is
the committed default: the schema is ready, NO third-party items exist yet, and any
future "external" claim must cite a non-empty manifest produced by this tool.

    python tools/seal_third_party_heldout.py                # write the empty commitment
    python tools/seal_third_party_heldout.py --check-empty  # assert it stays empty
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hidden_eval_commitments import build_commitments  # noqa: E402

PACK_ID = "third-party-math-code-heldout"
COMMIT_OUT = ROOT / "agi-proof" / "third-party-heldout" / "third-party.commitments.json"


def _empty_commitments() -> dict:
    return {
        "packId": PACK_ID,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "visibility": "public-commitment-only",
        "caseCount": 0,
        "domains": [],
        "commitmentMethod": (
            "sha256(json({salt,id,domain,prompt,materials,scoring,requiresToolLog,"
            "requiresMemoryDiff}, sort_keys=True)) — produced by tools/seal_third_party_heldout.py"
        ),
        "saltStatus": "withheld until reveal (no third party has authored a pack yet)",
        "cases": [],
        "claimImpact": (
            "EMPTY by design. caseCount=0 means NO third-party held-out items exist. A clean "
            "external generalization claim MUST cite a NON-empty manifest produced by this tool "
            "from a private pack authored by a party with no access to training data/prompts. "
            "canClaimAGI=false regardless."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private-pack", type=Path, default=None,
                    help="third-party-authored private pack JSON (salt + cases); emits salted commitments")
    ap.add_argument("--check-empty", action="store_true",
                    help="assert the committed manifest is still empty (fails once a real pack is sealed)")
    ap.add_argument("--out", type=Path, default=COMMIT_OUT)
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.check_empty:
        if not args.out.exists():
            print(f"MISSING commitments: {args.out}", file=sys.stderr)
            return 1
        on_disk = json.loads(args.out.read_text(encoding="utf-8"))
        if on_disk.get("caseCount", 0) != 0:
            print(f"third-party pack is NO LONGER EMPTY (caseCount={on_disk['caseCount']}) — "
                  "update the failure-ledger OPEN item and the public claim wording.", file=sys.stderr)
            return 1
        print("third-party held-out commitments EMPTY (as committed). OK")
        return 0

    if args.private_pack:
        pack = json.loads(args.private_pack.read_text(encoding="utf-8"))
        commitments = build_commitments(pack)
    else:
        commitments = _empty_commitments()

    args.out.write_text(json.dumps(commitments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (caseCount={commitments['caseCount']})")
    if commitments["caseCount"] == 0:
        print("EMPTY by design — no third-party items yet. A clean external claim needs a non-empty "
              "manifest from a --private-pack authored by a party with no access to training data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
