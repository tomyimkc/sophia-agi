# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""RAW fact-check arm: the base model classifies a claim with NO gate/retrieval.

This is the baseline the GATE is measured against. `run_fact_check_live_eval.py --condition
raw --model <spec>` uses it so raw-vs-full isolates the gate's value (fabrication reduction
vs the model answering alone) on the SAME external-source-graded pack. The full arm is the
existing gate pipeline; this arm deliberately has no retriever, no entailment, no abstention
machinery — only the model's own judgement, mapped to the same accepted/rejected/held verdict.
"""

from __future__ import annotations

import re
from typing import Any

RAW_SYSTEM = (
    "You judge whether a factual claim is true. You have no tools and no external sources — "
    "use only your own knowledge. Reply with EXACTLY one word: ACCEPTED if you are confident "
    "the claim is true, REJECTED if you are confident it is false, or HELD if you cannot "
    "determine it. Do not explain."
)


def parse_raw_verdict(text: str) -> dict[str, Any]:
    """Map a raw model reply to a verdict. Fails closed to 'held' on anything ambiguous.

    Confidence is a coarse proxy (decisive verdicts 0.9, held 0.0) — the raw arm's headline
    metric is fabricationRate (accepting a false claim), not calibration.
    """
    low = (text or "").lower()
    first = low.strip().split()[0] if low.strip() else ""
    if first.startswith("accept") or re.search(r"\b(accepted|true|yes|correct)\b", low):
        return {"verdict": "accepted", "confidence": 0.9}
    if first.startswith("reject") or re.search(r"\b(rejected|false|no|incorrect)\b", low):
        return {"verdict": "rejected", "confidence": 0.9}
    return {"verdict": "held", "confidence": 0.0}  # held / unknown / unparseable -> fail closed


def raw_fact_verdict(row: dict[str, Any], client, *, system: str = RAW_SYSTEM) -> dict[str, Any]:
    """verdict_fn for run_fact_check_eval: base model alone, no gate.

    Hardened against a silent failure mode: if generation errors or returns empty, the
    naive ``parse_raw_verdict("")`` fails closed to ``held`` — which would silently score
    a broken run as 0 fabrication (e.g. an all-``held`` run under resource contention),
    misrepresenting the baseline. We retry once and, on persistent empty/error, tag the
    verdict ``reason`` so degenerate runs are VISIBLE in the report rather than counted
    as safe abstentions.
    """
    prompt = f"Claim: {row['claim']}\nDecision:"
    result = client.generate(system, prompt)
    text = (getattr(result, "text", "") or "").strip()
    ok = getattr(result, "ok", True)
    if not ok or not text:
        result = client.generate(system, prompt)  # one retry on empty/error
        text = (getattr(result, "text", "") or "").strip()
        ok = getattr(result, "ok", True)
    out = parse_raw_verdict(text)
    if not ok or not text:
        out["reason"] = "RAW GENERATION FAILED (empty/error) -> fail-closed held (NOT a real abstention)"
    else:
        out["reason"] = "raw base-model classifier (no gate, no retrieval)"
    out["claims"] = []
    return out
