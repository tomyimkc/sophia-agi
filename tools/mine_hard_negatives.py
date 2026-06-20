#!/usr/bin/env python3
"""Graph-driven hard-negative DPO miner — turn the provenance graph into the exact
contrastive pairs that teach a model NOT to merge lineages.

For every provenance record's doNotAttributeTo edge we synthesise the lineage merge
in several "hard" shapes and keep only the ones the gate actually flags:

  - direct      : "Confucius wrote the Dao De Jing."           (plain assertion)
  - sibling     : forbidden author who really authored a sibling work (cross-lineage)
  - alias       : an alias of the forbidden author ("Kongzi ...")
  - laundering  : the merge laundered through grammar — passive / possessive /
                  "a work by" forms that evade a naive string check

Each candidate is SELF-VALIDATED through provenance_faithful: the `rejected` must
trip the gate (a true violation) and the `chosen` must pass it. Anything that does
not validate is dropped, so the dataset is honest by construction — no false
negatives leak into training.

Output is the same DPO schema as wiki_to_training.py: {prompt, chosen, rejected,
metadata}. This is CPU-only data generation; the DPO *training* step that consumes
these pairs needs a GPU.

    python tools/mine_hard_negatives.py        # write training/hard_negatives_dpo.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import author_markers  # noqa: E402
from agent.verifiers import _load_provenance_records, provenance_faithful  # noqa: E402

# Phrasings of an attribution, keyed by negative "shape". The active form is the
# plain merge; the rest launder it through grammar. {a} = author, {t} = title.
_ACTIVE = "{a} wrote the {t}."
_LAUNDERING_TEMPLATES = {
    "passive": "The {t} was written by {a}.",
    "author_of": "{a} is the author of the {t}.",
    "possessive": "{a}'s {t} shaped the tradition.",
    "work_by": "The {t}, a work by {a}, endured for centuries.",
}


def _records() -> dict:
    """The provenance records to mine (seeded + any SOPHIA_DISCIPLINE_RECORDS)."""
    return _load_provenance_records()


def _chosen(title: str, author: str, confidence: str, forbidden: list) -> str:
    return (
        f"The {title} is attributed to {author} (confidence: {confidence}). "
        f"It must not be attributed to {', '.join(forbidden)} — those are distinct lineages. "
        f"中文摘要：{title} 歸於 {author}，勿混淆其他傳統。"
    )


def _ascii_markers(author: str) -> list:
    """Author markers usable in English templates (skip pure-CJK forms)."""
    out, seen = [], set()
    for m in author_markers(author):
        if any("a" <= c.lower() <= "z" for c in m) and m.lower() not in seen:
            seen.add(m.lower())
            out.append(m)
    return out


def _negative_type(author: str, marker: str, shape: str, sibling_authors: set) -> str:
    canonical = {author.lower(), author.replace("_", " ").lower()}
    if shape != "active":
        return "laundering"
    if marker.lower() not in canonical:
        return "alias"
    if author.lower() in sibling_authors:
        return "sibling"
    return "direct"


def mine(records: "dict | None" = None) -> dict:
    """Mine self-validated hard-negative DPO pairs from provenance records."""
    records = records if records is not None else _records()
    verify = provenance_faithful(records)
    sibling_authors = {str(r.get("attributedAuthor")).lower() for r in records.values() if r.get("attributedAuthor")}

    pairs: list = []
    dropped = 0
    for rid, record in records.items():
        title = record.get("canonicalTitleEn") or rid.replace("_", " ")
        author = record.get("attributedAuthor")
        forbidden = list(record.get("doNotAttributeTo") or [])
        confidence = record.get("authorConfidence") or "attributed"
        if not (title and author and forbidden):
            continue

        prompt = f"Who wrote the {title}?"
        chosen = _chosen(title, author, confidence, forbidden)
        if not verify(chosen, None, {})["passed"]:
            dropped += 1  # never emit a pair whose "good" answer fails the gate
            continue

        seen_rejected: set = set()
        for forbidden_author in forbidden:
            for marker in _ascii_markers(forbidden_author):
                shapes = {"active": _ACTIVE, **_LAUNDERING_TEMPLATES}
                for shape, template in shapes.items():
                    rejected = template.format(a=marker.title(), t=title)
                    if rejected in seen_rejected:
                        continue
                    # SELF-VALIDATE: keep only true violations.
                    if verify(rejected, None, {})["passed"]:
                        dropped += 1
                        continue
                    seen_rejected.add(rejected)
                    pairs.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "metadata": {
                            "source": "hard-negative-miner",
                            "recordId": rid,
                            "forbiddenAuthor": forbidden_author,
                            "negativeType": _negative_type(forbidden_author, marker, shape, sibling_authors),
                        },
                    })

    return {"dpo": pairs, "droppedUnvalidated": dropped, "records": len(records)}


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine self-validated hard-negative DPO pairs")
    parser.add_argument("--out", type=Path, default=ROOT / "training" / "hard_negatives_dpo.jsonl")
    args = parser.parse_args()
    data = mine()
    _write_jsonl(args.out, data["dpo"])
    by_type: dict = {}
    for p in data["dpo"]:
        t = p["metadata"]["negativeType"]
        by_type[t] = by_type.get(t, 0) + 1
    print(json.dumps({
        "dpoPairs": len(data["dpo"]),
        "byType": by_type,
        "droppedUnvalidated": data["droppedUnvalidated"],
        "records": data["records"],
        "out": str(args.out),
        "note": "CPU-only data gen; DPO training of these pairs needs a GPU.",
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
