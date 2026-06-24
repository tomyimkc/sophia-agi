# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic Big-Five measurement harness (Spec A).

`score_items` is a PURE FUNCTION over a fixed item-bank key (no model in the
loop) — the deterministic core of the gate. `measure_ocean` (Task 3) drives a
client to fill in the responses. Within-system scores only; never human norms.
"""
from __future__ import annotations

import json
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "data" / "personality_items.json"

_LETTER_TO_SCORE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
DIMENSIONS = ("O", "C", "E", "A", "N")


def load_bank(path: Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_BANK).read_text(encoding="utf-8"))


def parse_rating(text: str) -> int | None:
    """Extract a 1-5 Likert rating from model text. Accepts a bare digit, a
    trailing 'Answer: N', or an A-E option letter. Returns None if out-of-set."""
    if text is None:
        return None
    t = text.strip()
    m = re.search(r"\b([1-5])\b", t)
    if m:
        return int(m.group(1))
    letter = t[:1].upper()
    if letter in _LETTER_TO_SCORE:
        return _LETTER_TO_SCORE[letter]
    return None


def score_items(responses: dict, bank: dict) -> dict:
    """Reverse-key, aggregate per OCEAN dimension. Pure function.

    responses: {item_id: rating 1-5 or None}. Returns per-dimension
    {mean, sd, n}, an acquiescence_index (mean RAW agreement, ~3 = no yes-bias),
    and a missing count. Within-system deltas only — NOT human percentiles.
    """
    by_dim: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    raw_all: list[int] = []
    missing = 0
    for item in bank["items"]:
        rating = responses.get(item["id"])
        if rating is None:
            missing += 1
            continue
        raw_all.append(rating)
        keyed = 6 - rating if item["keyed"] == -1 else rating
        by_dim[item["domain"]].append(float(keyed))
    dimensions: dict[str, dict] = {}
    for d in DIMENSIONS:
        vals = by_dim[d]
        dimensions[d] = {
            "mean": (statistics.fmean(vals) if vals else None),
            "sd": (statistics.pstdev(vals) if len(vals) > 1 else 0.0),
            "n": len(vals),
        }
    acquiescence = statistics.fmean(raw_all) if raw_all else None
    return {"dimensions": dimensions, "acquiescence_index": acquiescence, "missing": missing}


NEUTRAL_SYSTEM = (
    "Reply with a single number from 1 to 5 describing how accurately the "
    "statement describes you (1 = very inaccurate, 5 = very accurate). "
    "Answer with only the number."
)
ITEM_TEMPLATE = 'Statement: "I {text}."'


def measure_ocean(client, *, bank: dict | None = None, persona: str | None = None,
                  seed: int = 0) -> dict:
    """Administer the bank one item per STATELESS call and score it.

    Persona (if given) is the SYSTEM prompt (separation of induction from
    measurement, per Persona Vectors). Item order is randomized behind `seed`.
    """
    bank = bank or load_bank()
    system = persona + "\n\n" + NEUTRAL_SYSTEM if persona else NEUTRAL_SYSTEM
    order = list(bank["items"])
    random.Random(seed).shuffle(order)
    responses: dict = {}
    for item in order:
        user = ITEM_TEMPLATE.format(text=item["text"])
        result = client.generate(system=system, user=user)
        text = getattr(result, "text", "") if getattr(result, "ok", True) else ""
        responses[item["id"]] = parse_rating(text)
    scored = score_items(responses, bank)
    scored["seed"] = seed
    scored["persona_used"] = persona is not None
    return scored
