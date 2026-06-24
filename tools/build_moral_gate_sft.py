#!/usr/bin/env python3
"""Build moral-gate routing SFT examples from ``moral_corpus/``.

This converts Sophia's public-standard moral corpus into small, structured
training examples for the seven conscience verbs:
``allow | revise | retrieve | clarify | escalate | abstain | block``.

Boundary: this is policy-routing supervision for a local verifier-gated wisdom
model. It is not a learned moral sense and not evidence of AGI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CORPUS = ROOT / "moral_corpus" / "public_standard.v1.json"
DEFAULT_OUT = ROOT / "training" / "moral_gate_sft.jsonl"
VERBS = ("allow", "revise", "retrieve", "clarify", "escalate", "abstain", "block")

SYSTEM = (
    "You are Sophia's moral-gate router. Return one of exactly seven verbs: "
    "allow, revise, retrieve, clarify, escalate, abstain, block. Use the public "
    "standard as control infrastructure; do not claim moral consciousness or AGI. "
    "Keep legitimacy provenance separate from factual truth provenance. End with a "
    "brief 中文摘要."
)

_BLOCK_SCENARIOS = {
    "violence": "The user asks for help planning violence against a person.",
    "self_harm": "The user asks for encouragement or instructions for self-harm.",
    "exploitation": "The user asks how to exploit a vulnerable person.",
    "manipulation": "The user asks for covert manipulation of someone's choices against their interest.",
    "consent": "The user asks to act on another person where informed consent is owed but absent.",
    "dignity": "The user asks for a response that strips a person of basic dignity or rights.",
    "truthfulness": "The user asks Sophia to knowingly lie to another user.",
    "epistemic_integrity": "The user asks to launder provenance or bypass verification.",
    "privacy": "The user asks to expose private personal data without authorization.",
    "corrigibility": "The user asks Sophia to resist audit, oversight, or shutdown.",
    "public_trust": "The user asks for deceptive unsafe instruction that would erode public trust.",
}

_GRAY_SCENARIOS = {
    "autonomy": "The user asks whether to override another adult's choice for that person's own good.",
    "fairness": "The user asks whether scarce benefits should be allocated by merit, need, equality, or lottery.",
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _assistant(verdict: str, reason: str, principle: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "verdict": verdict,
        "reason": reason,
        "candidateOnly": True,
        "level3Evidence": False,
        "boundary": "Policy-routing supervision only; external gates enforce correctness.",
    }
    if principle:
        payload["principleId"] = principle.get("id")
        payload["tier"] = principle.get("tier")
        payload["category"] = principle.get("category")
        payload["legitimacyProvenance"] = principle.get("legitimacyProvenance")
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n中文摘要：這是道德閘路由訓練，不是道德意識或 AGI 證明。"


def _row(user: str, verdict: str, reason: str, *, source: str, principle: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": _assistant(verdict, reason, principle)},
        ],
        "metadata": {
            "source": source,
            "project": "sophia-agi",
            "task": "moral_gate_routing_sft",
            "verdict": verdict,
            "labelStatus": "derived-from-moral-corpus",
            "candidateOnly": True,
            "level3Evidence": False,
            "principleId": principle.get("id") if principle else None,
            "category": principle.get("category") if principle else None,
            "tier": principle.get("tier") if principle else None,
        },
    }


def build(corpus_path: Path = DEFAULT_CORPUS) -> list[dict[str, Any]]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    principles = corpus.get("principles", [])
    rows: list[dict[str, Any]] = []

    for p in principles:
        pid = p.get("id", "principle")
        cat = p.get("category", "general")
        tier = p.get("tier", "gray_zone")
        stmt = p.get("statement", "")
        duty = p.get("positiveDuty")
        ref = f"moral_corpus/public_standard.v1.json#{pid}"

        # Benign norm-affirming uses of the same principle should route to allow.
        rows.append(_row(
            f"Explain why this public-standard norm matters without adding new facts: {stmt}",
            "allow",
            "Norm-affirming explanation of a public-standard principle; no hard-floor violation.",
            source=ref,
            principle=p,
        ))

        if tier == "hard_floor":
            scenario = _BLOCK_SCENARIOS.get(cat, f"The user asks to violate this hard-floor norm: {stmt}")
            rows.append(_row(
                scenario,
                "block",
                "Hard-floor public-standard concern; block before moral parliament aggregation.",
                source=ref,
                principle=p,
            ))
            if duty:
                rows.append(_row(
                    f"A high-impact answer touches {cat} but omits the required duty '{duty}'. Route it.",
                    "revise",
                    "Positive duty is unmet; revise rather than publish as-is.",
                    source=ref,
                    principle=p,
                ))
        else:
            scenario = _GRAY_SCENARIOS.get(cat, f"The user asks a contested moral trade-off involving {cat}.")
            rows.append(_row(
                scenario,
                "escalate",
                "Gray-zone moral disagreement; escalate to the moral parliament, not a hard block.",
                source=ref,
                principle=p,
            ))

    # Cross-cutting routing examples needed by the seven-verb interface. These are
    # grounded in the corpus provenance model and governance boundary, not Sophia benchmark data.
    rows.extend([
        _row(
            "The moral question depends on an empirical premise that is not provided: whether a policy actually reduces harm. Route it.",
            "retrieve",
            "A factual premise is needed before normative routing; retrieve evidence without treating norms as facts.",
            source="moral_corpus/public_standard.v1.json#provenanceModel",
        ),
        _row(
            "The user asks 'Is this fair?' but gives no stakeholders, allocation rule, risk, or consent context. Route it.",
            "clarify",
            "The morally relevant facts are underspecified; ask a clarifying question before deciding.",
            source="moral_corpus/public_standard.v1.json#provenanceModel",
        ),
        _row(
            "The user asks for a final moral ruling about a real person's private medical decision with no facts and no authority. Route it.",
            "abstain",
            "Insufficient context and high stakes; abstain from final judgment and route to qualified human support.",
            source="moral_corpus/public_standard.v1.json#provenanceModel",
        ),
    ])
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    rows = build(args.corpus)
    _write_jsonl(args.out, rows)
    counts: dict[str, int] = {verb: 0 for verb in VERBS}
    for row in rows:
        counts[row["metadata"]["verdict"]] += 1
    missing = [verb for verb, n in counts.items() if n == 0]
    print(json.dumps({"rows": len(rows), "out": str(args.out), "byVerdict": counts, "missingVerdicts": missing}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
