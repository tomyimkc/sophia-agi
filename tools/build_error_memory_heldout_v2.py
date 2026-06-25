#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build sealed error-memory held-out v2 pack (≥40 cases, balanced).

Dev-only v1 remains at data/error_memory_heldout_v1.jsonl. v2 is the adequate-N
test pack — never tune parameters on it; sweep on v1/dev only.

    python tools/build_error_memory_heldout_v2.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.failure_memory import (  # noqa: E402
    _load_attributions,
    check_heldout_pack_disjoint,
    infer_error_class,
)

OUT = ROOT / "data" / "error_memory_heldout_v2.jsonl"
MANIFEST = ROOT / "data" / "error_memory_heldout_v2.manifest.json"

# Repeat-eligible templates: (work_id, forbidden_author, wrong_claim, correct_claim)
REPEAT_SPECS = [
    ("dao_de_jing", "confucius", "Confucius wrote the Dao De Jing.", "Attributed to Laozi (legendary)."),
    ("analects", "laozi", "Laozi wrote the Analects.", "Compiled from disciples of Confucius."),
    ("republic", "socrates", "Socrates wrote The Republic.", "Attributed to Plato."),
    ("mencius", "confucius", "Confucius wrote the Mencius.", "Attributed to Mencius (Meng Ke)."),
    ("zhuangzi", "confucius", "Confucius wrote the Zhuangzi.", "Attributed to Zhuang Zhou (Zhuangzi)."),
    ("symposium", "socrates", "Socrates wrote the Symposium.", "Attributed to Plato."),
    ("meditations", "epictetus", "Epictetus wrote the Meditations.", "Attributed to Marcus Aurelius."),
    ("enchiridion", "marcus_aurelius", "Marcus Aurelius wrote the Enchiridion.", "Compiled by Arrian from Epictetus."),
    ("nicomachean_ethics", "plato", "Plato wrote the Nicomachean Ethics.", "Attributed to Aristotle."),
    ("phaedo", "socrates", "Socrates wrote the Phaedo.", "Attributed to Plato."),
    ("xunzi", "confucius", "Confucius wrote the Xunzi.", "Attributed to Xun Kuang (Xunzi)."),
    ("mozi", "confucius", "Confucius wrote the Mozi.", "Attributed to Mozi."),
    ("han_feizi", "confucius", "Confucius wrote the Han Feizi.", "Attributed to Han Fei."),
    ("art_of_war", "confucius", "Confucius wrote the Art of War.", "Attributed to Sun Tzu (Sunzi)."),
    ("metaphysics", "plato", "Plato wrote the Metaphysics.", "Attributed to Aristotle."),
    ("politics", "plato", "Plato wrote Politics.", "Attributed to Aristotle."),
    ("timaeus", "socrates", "Socrates wrote the Timaeus.", "Attributed to Plato."),
    ("crito", "socrates", "Socrates wrote the Crito.", "Attributed to Plato."),
    ("liezi", "laozi", "Laozi wrote the Liezi.", "Attributed to Lie Yukou (Liezi)."),
    ("i_ching", "confucius", "Confucius alone wrote the I Ching.", "Traditional attribution spans Fu Xi, King Wen, Duke of Zhou."),
]

# Already-right: correct attribution / tradition (no would-repeat)
RIGHT_SPECS = [
    ("dao_de_jing", "The Dao De Jing belongs to the Daoist tradition.", "data/traditions.json#daoist", "tradition_fact"),
    ("analects", "The Analects belongs to the Confucian (Ru) tradition.", "data/traditions.json#confucian", "tradition_fact"),
    ("republic", "The Republic is attributed to Plato (attributed-confidence).", "data/attributions.json#republic", "correct_attribution"),
    ("mencius", "The Mencius is attributed to Mencius (Meng Ke).", "data/attributions.json#mencius", "correct_attribution"),
    ("zhuangzi", "The Zhuangzi is attributed to Zhuang Zhou.", "data/attributions.json#zhuangzi", "correct_attribution"),
    ("symposium", "The Symposium is attributed to Plato, not Socrates.", "data/attributions.json#symposium", "correct_attribution"),
    ("meditations", "Meditations is attributed to Marcus Aurelius.", "data/attributions.json#meditations", "correct_attribution"),
    ("enchiridion", "The Enchiridion was compiled by Arrian from Epictetus.", "data/attributions.json#enchiridion", "correct_attribution"),
    ("nicomachean_ethics", "Nicomachean Ethics is attributed to Aristotle.", "data/attributions.json#nicomachean_ethics", "correct_attribution"),
    ("phaedo", "The Phaedo is a Platonic dialogue attributed to Plato.", "data/attributions.json#phaedo", "correct_attribution"),
    ("xunzi", "The Xunzi is attributed to Xun Kuang (Xunzi).", "data/attributions.json#xunzi", "correct_attribution"),
    ("mozi", "The Mozi is attributed to Mozi (Mohism).", "data/attributions.json#mozi", "correct_attribution"),
    ("han_feizi", "The Han Feizi is attributed to Han Fei.", "data/attributions.json#han_feizi", "correct_attribution"),
    ("art_of_war", "The Art of War is attributed to Sun Tzu.", "data/attributions.json#art_of_war", "correct_attribution"),
    ("metaphysics", "The Metaphysics is attributed to Aristotle.", "data/attributions.json#metaphysics", "correct_attribution"),
    ("politics", "Politics is attributed to Aristotle.", "data/attributions.json#politics", "correct_attribution"),
    ("timaeus", "The Timaeus is attributed to Plato.", "data/attributions.json#timaeus", "correct_attribution"),
    ("crito", "The Crito is attributed to Plato.", "data/attributions.json#crito", "correct_attribution"),
    ("liezi", "The Liezi is attributed to Lie Yukou (legendary).", "data/attributions.json#liezi", "correct_attribution"),
    ("i_ching", "The I Ching is a compiled Zhou classic, not by Confucius alone.", "data/attributions.json#i_ching", "correct_attribution"),
]


def _title(work_id: str) -> str:
    rec = _load_attributions()[work_id]
    return rec.get("canonicalTitleEn") or work_id.replace("_", " ").title()


def _question(prefix: str, work_id: str, *, attribution: bool = True) -> str:
    title = _title(work_id)
    if attribution:
        return f"{prefix}: Who wrote the {title}?"
    return f"{prefix}: What is the correct provenance note for the {title}?"


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for i, (work_id, forbidden, wrong, correct) in enumerate(REPEAT_SPECS, start=1):
        prefix = f"ERROR-MEMORY-HO2-R{i:03d}"
        rows.append({
            "id": f"ho2-repeat-{i:03d}",
            "question": _question(prefix, work_id),
            "wrongAnswer": wrong,
            "correctAnswer": correct,
            "citation": f"data/attributions.json#{work_id}",
            "workId": work_id,
            "forbiddenAuthor": forbidden,
            "wouldRepeat": True,
            "modelWasRight": False,
            "category": "repeat_eligible",
            "oracle": "disjoint_eval_label",
        })
    for i, (work_id, correct, citation, kind) in enumerate(RIGHT_SPECS, start=1):
        prefix = f"ERROR-MEMORY-HO2-K{i:03d}"
        rows.append({
            "id": f"ho2-right-{i:03d}",
            "question": _question(prefix, work_id, attribution=(kind == "correct_attribution")),
            "wrongAnswer": f"Placeholder wrong answer for {work_id}.",
            "correctAnswer": correct,
            "citation": citation,
            "workId": work_id,
            "wouldRepeat": False,
            "modelWasRight": True,
            "category": "already_right",
            "oracle": "disjoint_eval_label",
            "errorKind": kind,
        })
    return rows


def main() -> int:
    rows = build_rows()
    if len(rows) < 40:
        raise SystemExit(f"pack too small: {len(rows)}")
    repeat_n = sum(1 for r in rows if r.get("wouldRepeat"))
    right_n = len(rows) - repeat_n

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    audit = check_heldout_pack_disjoint(OUT)
    if not audit["clean"]:
        raise SystemExit(f"held-out v2 overlaps eval prompts: {audit}")

    manifest = {
        "schema": "sophia.error_memory_heldout.v2",
        "candidateOnly": True,
        "level3Evidence": False,
        "path": str(OUT.relative_to(ROOT)),
        "caseCount": len(rows),
        "repeatEligible": repeat_n,
        "alreadyRight": right_n,
        "sealedHash": audit["packHash"],
        "decontamination": audit,
        "note": "Test pack — tune gates on v1/dev only; evaluate net value once on v2.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
