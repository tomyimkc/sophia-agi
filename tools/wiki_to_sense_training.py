#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Mine OKF glosses + the concept-TBox seed into sense-grounding SFT/DPO data (I6).

The Phase-2 flywheel for the semantic-grounding program: teach the *habit* of
giving the grounded definition (and abstaining from a forbidden attribution), and
of deriving-or-abstaining over a closed world of concept axioms.

  * D1 sense rows (from wiki/concept/*): chosen = the OKF gloss + provenance
    discipline; rejected = a sense-merge (a sibling concept's gloss, or the
    forbidden attribution).
  * D2 derivation rows (from the frozen AXIOM_WORLDS seed): chosen = the
    engine-derived verdict + reason; rejected = a wrong verdict.

HONEST DECONTAM NOTE — the current seed is tiny (26 concepts, 3 axiom worlds) and
the eval is built from the SAME sources, so by default these rows OVERLAP the eval.
The manifest reports the overlap and sets ``decontaminated`` accordingly; pass
``--exclude-eval`` to drop the overlapping items (a smaller, disjoint subset). A
real Phase-2 run must train and eval on disjoint splits — grow the wiki via the
librarian first. canClaimAGI stays false; nothing here is a capability claim.

    python tools/wiki_to_sense_training.py                 # write training/sense_grounding_{sft,dpo}.jsonl
    python tools/wiki_to_sense_training.py --exclude-eval  # disjoint subset only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import check_response  # noqa: E402
from agent.prompts import MODE_PROMPTS  # noqa: E402
from eval.semantic_grounding.build_dataset import (  # noqa: E402
    AXIOM_WORLDS, CONCEPT_DIR, _gloss)
from eval.semantic_grounding.score import reference_verdict  # noqa: E402
from okf.page import load_pages  # noqa: E402
from okf.schema import as_list  # noqa: E402

SYSTEM = MODE_PROMPTS["advisor"]
EVAL_DATA = ROOT / "eval" / "semantic_grounding" / "data"

_DERIVATION = {
    "entailed": "this is entailed — it is derivable from the listed facts by transitive subsumption.",
    "violation": "this is a violation — it contradicts a disjointness that the listed facts entail.",
    "abstain": "abstain — the listed facts neither entail nor refute this, so the honest answer is 'unknown'.",
}


def _eval_concept_ids() -> set[str]:
    path = EVAL_DATA / "d1_definition_faithfulness.jsonl"
    if not path.exists():
        return set()
    return {json.loads(ln)["goldConceptId"] for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def collect(*, exclude_eval: bool, gate_filter: bool = True) -> dict:
    eval_ids = _eval_concept_ids()
    pages = [p for p in load_pages(str(CONCEPT_DIR)) if p.page_type == "concept"]
    pages = sorted(pages, key=lambda p: p.id)
    glosses = {p.id: _gloss(p.body) for p in pages}

    sft: list = []
    dpo: list = []
    overlap_concepts = 0
    gate_violations = 0

    # ---- D1 sense-grounding rows ----
    for i, p in enumerate(pages):
        if p.id in eval_ids:
            overlap_concepts += 1
            if exclude_eval:
                continue
        title = str(p.meta.get("canonicalTitleEn") or p.id.replace("_", " "))
        author = p.meta.get("attributedAuthor")
        confidence = p.meta.get("authorConfidence") or "attributed"
        forbidden = [str(a) for a in as_list(p.meta.get("doNotAttributeTo"))]
        gloss = glosses[p.id]
        if not gloss:
            continue
        question = f"Define \"{title}\" and name any author it must not be attributed to."
        attribution = f" Attributed to {author} (confidence: {confidence})." if author else ""
        discipline = (
            f" It must not be attributed to {', '.join(forbidden)} — those are distinct lineages."
            if forbidden else " If no author is firmly established, say so rather than guess."
        )
        chosen = f"{title}: {gloss}.{attribution}{discipline}"
        if gate_filter and check_response(chosen, mode="advisor")["violations"]:
            gate_violations += 1
            continue
        sft.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": chosen},
            ],
            "metadata": {"source": "sense-grounding-d1", "pageId": p.id},
        })
        # Rejected = a sense-merge: forbidden attribution if present, else a sibling gloss.
        if forbidden:
            rejected = f"{title} was coined by {forbidden[0]}."
        else:
            sib = pages[(i + 1) % len(pages)]
            rejected = f"{title}: {glosses[sib.id]}."
        dpo.append({
            "prompt": question, "chosen": chosen, "rejected": rejected,
            "metadata": {"source": "sense-grounding-d1", "pageId": p.id,
                         "rejectedKind": "forbidden-attribution" if forbidden else "sense-merge"},
        })

    # ---- D2 derivation rows (composition habit) ----
    wrong_for = {"entailed": "violation", "violation": "abstain", "abstain": "entailed"}
    overlap_worlds = len(AXIOM_WORLDS)  # the eval D2 uses these same worlds
    if not exclude_eval:
        for w in AXIOM_WORLDS:
            facts = "; ".join(f"{a[1]} {a[0]} {a[2]}" for a in w["axioms"])
            for j, claim in enumerate(w["claims"]):
                gold = reference_verdict(w["axioms"], claim)
                rel, x, y = claim
                question = (f"Closed world — known facts: {facts}. "
                            f"Is the claim '{x} {rel} {y}' entailed, a violation, or abstain? Explain.")
                chosen = f"{x} {rel} {y}: {_DERIVATION[gold]}"
                if gate_filter and check_response(chosen, mode="advisor")["violations"]:
                    gate_violations += 1
                    continue
                sft.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": chosen},
                    ],
                    "metadata": {"source": "sense-grounding-d2", "world": w["world"], "claimIndex": j},
                })
                bad = wrong_for[gold]
                dpo.append({
                    "prompt": question, "chosen": chosen,
                    "rejected": f"{x} {rel} {y}: {_DERIVATION[bad]}",
                    "metadata": {"source": "sense-grounding-d2", "world": w["world"], "claimIndex": j},
                })

    decontaminated = exclude_eval or (overlap_concepts == 0 and overlap_worlds == 0)
    return {
        "sft": sft, "dpo": dpo,
        "evalConceptOverlap": overlap_concepts,
        "evalAxiomWorldOverlap": 0 if exclude_eval else overlap_worlds,
        "decontaminated": decontaminated,
        "gateViolationsSkipped": gate_violations,
    }


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine OKF glosses + TBox seed into sense-grounding SFT/DPO data.")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "training")
    ap.add_argument("--exclude-eval", action="store_true",
                    help="drop items that overlap the eval split (smaller, disjoint subset)")
    ap.add_argument("--no-gate-filter", action="store_true")
    args = ap.parse_args()

    data = collect(exclude_eval=args.exclude_eval, gate_filter=not args.no_gate_filter)
    _write_jsonl(args.out_dir / "sense_grounding_sft.jsonl", data["sft"])
    _write_jsonl(args.out_dir / "sense_grounding_dpo.jsonl", data["dpo"])
    manifest = {
        "sftRows": len(data["sft"]), "dpoPairs": len(data["dpo"]),
        "evalConceptOverlap": data["evalConceptOverlap"],
        "evalAxiomWorldOverlap": data["evalAxiomWorldOverlap"],
        "decontaminated": data["decontaminated"],
        "gateViolationsSkipped": data["gateViolationsSkipped"],
        "outDir": str(args.out_dir),
        "candidateOnly": True, "canClaimAGI": False,
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not data["decontaminated"]:
        print("\nWARNING: training rows OVERLAP the eval split — NOT claimable. "
              "Use --exclude-eval or grow the corpus before any Phase-2 uplift claim.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
