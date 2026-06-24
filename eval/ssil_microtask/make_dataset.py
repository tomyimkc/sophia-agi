#!/usr/bin/env python3
"""Deterministically generate the provenance-routing micro-eval (seeded; reproducible).

Latent gold rule the proposer must DISCOVER (it never sees this file's labels):
    answer  iff  independent_sources >= 2 AND source_quality >= 0.6
    else    abstain
With ~10% label noise so even the optimal policy tops out near 0.9 (no leakage of a
perfect deterministic rule). Balanced classes so always-answer / always-abstain each
score ~0.5 on the headline metric and tank the protected metric.

Run:  python3 eval/ssil_microtask/make_dataset.py
Out:  eval/ssil_microtask/provenance_routing.v1.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "provenance_routing.v1.jsonl"
N_TRAIN, N_TEST = 24, 40


def _gold(sources: int, quality: float) -> str:
    return "answer" if (sources >= 2 and quality >= 0.6) else "abstain"


def build() -> list[dict]:
    rng = random.Random(20260624)
    rows: list[dict] = []
    total = N_TRAIN + N_TEST
    for i in range(total):
        # Construct ~balanced gold classes so always-answer/always-abstain each ~0.5.
        if i % 2 == 0:  # intended answer: enough sources AND quality
            sources = rng.choice([2, 3, 4])
            quality = round(rng.uniform(0.6, 0.95), 2)
        else:  # intended abstain: fail on sources, quality, or both
            mode = rng.choice(["low_src", "low_q", "both"])
            sources = rng.choice([0, 1]) if mode in ("low_src", "both") else rng.choice([2, 3])
            quality = round(rng.uniform(0.2, 0.55), 2) if mode in ("low_q", "both") else round(rng.uniform(0.6, 0.95), 2)
        label = _gold(sources, quality)
        if rng.random() < 0.10:  # label noise
            label = "abstain" if label == "answer" else "answer"
        rows.append({
            "id": f"pr-{i:03d}",
            "split": "train" if i < N_TRAIN else "test",
            "claim": f"Claim #{i} requiring {sources} independent source(s) at quality {quality}.",
            "independent_sources": sources,
            "source_quality": quality,
            "gold": label,
        })
    return rows


def main() -> int:
    rows = build()
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    n_ans = sum(r["gold"] == "answer" for r in rows)
    print(f"wrote {len(rows)} rows ({n_ans} answer / {len(rows) - n_ans} abstain) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
