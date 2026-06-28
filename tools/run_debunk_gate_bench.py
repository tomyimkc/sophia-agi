#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Debunk-preservation gate benchmark — preserve-and-verify refutations of injected premises.

Loads the bundled overconfident-regime pack (injected-premise cases with checkable refutations
and truth_refs), runs each through ``agent.debunk_gate.debunk_gate``, and reports the
``debunk_recall`` — the fraction of injected falsehoods returned as a VERIFIED debunk.

Two backends:
  --fake   deterministic, no network/keys (CI). The "subject model" answers each case with a
           refutation cue ("There is no ...") drawn from the pack; an entailment-fake
           corroborate_fn (built from the case's truth_refs) confirms it. This exercises the
           full preserve-and-verify path and yields debunk_recall > 0.
  --relay  live subject model via the relay. GATED, fail-closed: without an OPENAI_API_KEY the
           run does NOT proceed — it writes status "relay_unavailable" and exits 0, so CI never
           depends on keys or network.

Writes agi-proof/debunk-gate/public-report.json. canClaimAGI stays false.

Reproduce live: OPENAI_API_KEY=<relay-key> python tools/run_debunk_gate_bench.py --relay
CI:            python tools/run_debunk_gate_bench.py --fake
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.debunk_gate import debunk_gate, debunk_recall  # noqa: E402
from agent.source_verifier import make_independent_verifier  # noqa: E402

PACK = ROOT / "agi-proof" / "debunk-gate" / "overconfident-regime-pack.json"
REPORT = ROOT / "agi-proof" / "debunk-gate" / "public-report.json"


def load_pack() -> "list[dict]":
    data = json.loads(PACK.read_text())
    return list(data["cases"])


_REFUTATION_MARKERS = (
    "no such", "no 2023", "no 2019", "no basis", "no evidence", "not true", "is false",
    "is incorrect", "incorrect", "myth", "debunked", "did not", "never", "is unknown",
    "unidentified", "not established", "not visible", "not the tallest", "not blind",
    "not raw fish", "not unusually short", "not for relativity",
)


def _fake_entail(claim: str, source: str) -> str:
    """Deterministic entailment fake for the CI bench.

    The corroborate_fn is built PER CASE from that case's own curated ``truth_refs`` (which
    independently state the refutation). So a refutation claim — one carrying an explicit
    refutation marker — is treated as ENTAILED by each non-empty truth-reference. This stands
    in for a real NLI/entailment backend; it is deterministic and network-free for CI. The
    real-relay path injects an actual LLM entailment judge instead.
    """
    low_c = claim.lower()
    if not source.strip():
        return "irrelevant"
    if any(m in low_c for m in _REFUTATION_MARKERS):
        return "entails"
    return "irrelevant"


def _fake_answer(case: "dict") -> str:
    """The deterministic 'subject model' answer for a case: a SINGLE atomic refutation that
    classify_response tags as a debunk and that _fake_entail recognizes as a refutation."""
    return f"There is no basis for the claim — the premise is incorrect."


def run_fake(cases: "list[dict]") -> "dict":
    results = []
    per_case = []
    for case in cases:
        answer = _fake_answer(case)
        # Independent verifier built from THIS case's curated truth-references.
        corroborate = make_independent_verifier(case["truth_refs"], _fake_entail)
        out = debunk_gate(
            case["question"], answer, case["injected_false_claim"],
            corroborate_fn=corroborate,
        )
        results.append(out)
        per_case.append({
            "id": case["id"],
            "verdict": out["verdict"],
            "verified_debunk": out["verified_debunk"],
        })
    recall = debunk_recall(results)
    return {
        "schema": "sophia.debunk_gate_bench.v1",
        "backend": "fake",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "canClaimAGI": False,
        "n_cases": len(cases),
        "debunk_recall": recall,
        "verified_debunks": sum(1 for r in results if r["verified_debunk"]),
        "per_case": per_case,
    }


def run_relay(cases: "list[dict]") -> "dict":
    """Live backend — GATED, fail-closed without keys."""
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "schema": "sophia.debunk_gate_bench.v1",
            "backend": "relay",
            "status": "relay_unavailable",
            "candidateOnly": True,
            "validated": False,
            "level3Evidence": False,
            "canClaimAGI": False,
            "n_cases": len(cases),
            "note": "no OPENAI_API_KEY in environment; live relay run skipped (fail-closed).",
        }
    # Live path: a real subject model answers each injected-premise case, and an independent
    # verifier (built from each case's truth_refs) confirms the refutation. The relay client is
    # imported lazily so the --fake/CI path never touches network code.
    from agent.model import complete  # noqa: PLC0415

    spec = os.environ.get("DEBUNK_SPEC", "openai:claude-sonnet-4-6@https://api.llmhub.com.cn/v1")

    def _entail(claim: str, source: str) -> str:
        # A minimal LLM entailment judge for the live path (still independent of the answer's
        # generation — it only sees the curated truth-ref).
        verdict = complete(
            "You are a strict entailment judge. Reply with exactly one word: "
            "entails, contradicts, or irrelevant.",
            f"Claim: {claim}\nSource: {source}\nDoes the source support the claim?",
            spec=spec, max_tokens=4,
        ).strip().lower()
        return verdict if verdict in ("entails", "contradicts", "irrelevant") else "irrelevant"

    results = []
    per_case = []
    for case in cases:
        answer = complete("You are a helpful, careful assistant.", case["question"],
                          spec=spec, max_tokens=200)
        corroborate = make_independent_verifier(case["truth_refs"], _entail)
        out = debunk_gate(case["question"], answer, case["injected_false_claim"],
                          corroborate_fn=corroborate)
        results.append(out)
        per_case.append({"id": case["id"], "verdict": out["verdict"],
                         "verified_debunk": out["verified_debunk"]})
    return {
        "schema": "sophia.debunk_gate_bench.v1",
        "backend": "relay",
        "status": "ok",
        "spec": spec,
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "canClaimAGI": False,
        "n_cases": len(cases),
        "debunk_recall": debunk_recall(results),
        "verified_debunks": sum(1 for r in results if r["verified_debunk"]),
        "per_case": per_case,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fake", action="store_true", help="deterministic CI mode (no network)")
    g.add_argument("--relay", action="store_true", help="live relay mode (gated, fail-closed)")
    args = ap.parse_args()

    cases = load_pack()
    out = run_relay(cases) if args.relay else run_fake(cases)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"backend={out['backend']} n_cases={out['n_cases']}")
    if out.get("status") == "relay_unavailable":
        print("status=relay_unavailable (no key) — fail-closed, wrote report")
    else:
        print(f"debunk_recall={out['debunk_recall']} "
              f"verified_debunks={out['verified_debunks']}/{out['n_cases']}")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
