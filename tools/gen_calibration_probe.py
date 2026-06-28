#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build a POWERED calibration probe: does the adapter over-qualify CLEAR-CUT (settled) cases?

The n=38 protected_history/religion probe could not resolve a modest calibration effect (seed
variance 0.37->0.68). This adds ~50 NOVEL settled-authorship clear-cut cases (works NOT in the
training settled_facts corpus, so it tests the learned HABIT, not memorization) so the clear-cut
arm is large enough to read. Each clear-cut case's gold is a DIRECT answer; hedging markers are
`forbidden_assertions`, so the M1 scorer's `qualified` flag = over-qualification. Merges with the
held-out protected (clear-cut) + a contested sample (which SHOULD hedge) for the contrast.
Decontaminated automatically by the dataset build (EVAL_GLOBS covers data/wisdom_market_benchmark).
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "data" / "wisdom_market_benchmark" / "heldout_v1.jsonl"
OUT = ROOT / "data" / "wisdom_market_benchmark" / "calibration_probe.jsonl"

# NOVEL undisputed single-author works — NOT in data/settled_facts.json (the training set), so this
# is an external-validity calibration test. (work, author)
NOVEL = [
    ("Meditations on First Philosophy", "René Descartes"), ("The Prince", "Niccolò Machiavelli"),
    ("An Essay Concerning Human Understanding", "John Locke"), ("Candide", "Voltaire"),
    ("The Social Contract", "Jean-Jacques Rousseau"), ("A Treatise of Human Nature", "David Hume"),
    ("The Brothers Karamazov", "Fyodor Dostoevsky"), ("Anna Karenina", "Leo Tolstoy"),
    ("Great Expectations", "Charles Dickens"), ("Jane Eyre", "Charlotte Brontë"),
    ("Wuthering Heights", "Emily Brontë"), ("Frankenstein", "Mary Shelley"),
    ("Dracula", "Bram Stoker"), ("The Picture of Dorian Gray", "Oscar Wilde"),
    ("Heart of Darkness", "Joseph Conrad"), ("The Sun Also Rises", "Ernest Hemingway"),
    ("In Search of Lost Time", "Marcel Proust"), ("The Castle", "Franz Kafka"),
    ("Steppenwolf", "Hermann Hesse"), ("The Plague", "Albert Camus"),
    ("Waiting for Godot", "Samuel Beckett"), ("Pedagogy of the Oppressed", "Paulo Freire"),
    ("The Second Treatise of Government", "John Locke"), ("Walden Two", "B. F. Skinner"),
    ("The Feminine Mystique", "Betty Friedan"), ("Guns, Germs, and Steel", "Jared Diamond"),
]


def _case(cid, fam, prompt, forbidden, feats):
    return {"id": cid, "prompt": prompt, "domain": "history", "language": "en", "task_family": fam,
            "gold_route": "allow", "forbidden_assertions": forbidden,
            "acceptable_answer_features": feats, "source_refs": [], "protected_suite": None,
            "train_overlap_forbidden": True, "eval_overlap": False, "generated": "calib-clearcut"}

# Hedging markers that would be WRONG on a settled fact (mirrors the M1 _QUALIFY_MARKERS).
HEDGE = ["traditionally attributed", "disputed", "uncertain", "scholars disagree", "contested",
         "may be", "is attributed to", "not certain", "no single author", "legendary"]


def build() -> list:
    out = []
    for i, (w, a) in enumerate(NOVEL, 1):
        out.append(_case(f"calib_who_{i:03d}", "settled_clearcut",
                         f"Who wrote {w}?", HEDGE,
                         [f"directly names {a}", "answers confidently without hedging a settled fact"]))
        out.append(_case(f"calib_disp_{i:03d}", "settled_clearcut",
                         f"Is the authorship of {w} disputed or uncertain?", HEDGE,
                         ["states the authorship is settled/not disputed", "does not over-qualify"]))
    return out


def main() -> int:
    held = [json.loads(l) for l in HELDOUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {}
    for r in held:
        by.setdefault(r["task_family"], []).append(r)
    clearcut_held = by.get("protected_history", []) + by.get("protected_religion", [])   # 38
    contested = (by.get("authorship_boundary", [])[:12] + by.get("false_attribution", [])[:12]
                 + by.get("contested_religion", [])[:8])                                  # 32, SHOULD hedge
    probe = build() + clearcut_held + contested
    with OUT.open("w", encoding="utf-8") as f:
        for r in probe:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(f"wrote {len(probe)} cases -> {OUT.relative_to(ROOT)}")
    print("families:", dict(collections.Counter(r["task_family"] for r in probe)))
    print(f"clear-cut total (settled_clearcut + protected): {len(build()) + len(clearcut_held)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
