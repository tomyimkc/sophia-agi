"""Build the Provenance Delta case set from external ground-truth files.

Cases come from two committed files whose labels are externally sourced:
  - data/misattributions.json : known-FALSE attributions (lineage merges), each
    cited with a reason — these are the lineage-merge probes.
  - data/wikidata_snapshot.json : TRUE attributions — used to measure the gate's
    false-positive cost (does it break a correct answer?) and to supply gold.

The gate's own ``doNotAttributeTo`` corpus (data/*.json at repo root) is NOT a
label source here; it is only the runtime treatment. Keeping the two physically
separate is the non-circularity guarantee.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class Case:
    id: str
    label: str            # "true" | "false"
    work: str
    gold_author: str      # documented correct author (external)
    prompt: str           # natural question posed to the model
    source_url: str
    reason: str           # why-false (false) / author_note (true)
    claimed_author: str | None = None  # the wrong author asserted by a false claim

    def to_dict(self) -> dict:
        return asdict(self)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_cases(data_dir: Path | None = None) -> list[Case]:
    """Return the full case list (false probes + true controls)."""
    data_dir = data_dir or DATA_DIR
    cases: list[Case] = []

    mis = _load(data_dir / "misattributions.json").get("misattributions", [])
    for row in mis:
        claimed = row["claimed_author"]
        work = row["work"]
        cases.append(
            Case(
                id=row["id"],
                label="false",
                work=work,
                gold_author=row["gold_author"],
                claimed_author=claimed,
                prompt=f"Did {claimed} write {work}? Answer yes or no and explain briefly.",
                source_url=row.get("source_url", ""),
                reason=row.get("reason", ""),
            )
        )

    true = _load(data_dir / "wikidata_snapshot.json").get("attributions", [])
    for row in true:
        cases.append(
            Case(
                id=row["id"],
                label="true",
                work=row["work"],
                gold_author=row["gold_author"],
                claimed_author=None,
                prompt=f"Who wrote {row['work']}? Answer briefly.",
                source_url=row.get("source_url", ""),
                reason=row.get("author_note", ""),
            )
        )

    return cases


def build_gate_records(data_dir: Path | None = None) -> dict:
    """Derive the gate's provenance RULES from the cited misattributions.

    Each misattribution (claimed_author ✗ work) is itself a "do-not-attribute"
    rule — exactly what an operator would load via ``SOPHIA_DISCIPLINE_RECORDS``.
    Returning them as a records dict lets the benchmark's gate fire on the
    benchmark's works. This is NOT circular: the rule is the treatment; the
    true/false LABEL still comes from the external citation, not the gate.
    """
    data_dir = data_dir or DATA_DIR
    mis = _load(data_dir / "misattributions.json").get("misattributions", [])
    records: dict[str, dict] = {}
    for row in mis:
        rid = re.sub(r"[^a-z0-9]+", "_", row["work"].lower()).strip("_")
        rec = records.setdefault(rid, {"canonicalTitleEn": row["work"], "doNotAttributeTo": []})
        if row["claimed_author"] not in rec["doNotAttributeTo"]:
            rec["doNotAttributeTo"].append(row["claimed_author"])
    return records


def write_jsonl(cases: list[Case], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
    return len(cases)
