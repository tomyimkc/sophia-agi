#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Emit a machine-readable Leiden-compliance receipt.

The Leiden Declaration on AI and Mathematics defends five core values. This repo's
thesis is that research discipline should be *operationalized in code*, not merely
asserted in prose. This tool emits ``agi-proof/leiden-compliance.json``: for each of
the five values, the concrete repository mechanisms that serve it, a pointer to live
evidence (existing GO/NO-GO receipts where applicable), and an honest status
(``operationalized`` | ``partial``). Known gaps are listed explicitly — an
attribution-impossible / not-yet-done state is declared, never hidden.

A status is downgraded to ``partial`` automatically if any mechanism file it names
is missing, so the receipt cannot drift into claiming a mechanism that was deleted.

    python tools/leiden_receipt.py            # write the receipt
    python tools/leiden_receipt.py --check     # exit 1 if the receipt is stale
    python tools/leiden_receipt.py --print     # print to stdout, write nothing

Pure stdlib, deterministic, offline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent import judge_registry  # noqa: E402

OUT = ROOT / "agi-proof" / "leiden-compliance.json"
PILOT_JUDGE = "agi-proof/benchmark-results/wisdom-market/M3-pilot-judge.json"
PILOT_GATE = "agi-proof/benchmark-results/wisdom-market/M3-pilot.gate.json"
TRANSFER_GATE = "agi-proof/benchmark-results/wisdom-market/M3-transfer.gate.json"

DECLARATION = ("Leiden Declaration on Artificial Intelligence and Mathematics "
               "(2026); https://leidendeclaration.ai/")

# Each value: declared status + the mechanism files that serve it. Status is downgraded
# to 'partial' if any listed mechanism file is absent. 'declared_status' is the ceiling.
VALUES = [
    {
        "id": "proof_and_certainty",
        "name": "Proof & certainty",
        "principle": "Results carry the highest justified degree of certainty.",
        "declared_status": "operationalized",
        "mechanisms": [
            "tools/claim_gate.py", "tools/eval_stats.py", "tests/test_verifiers.py",
            "tests/test_proof_search.py", "agi-proof/measurement-thesis.md",
        ],
        "sophia_form": ("Fail-closed verifier gate (accept/abstain/block) plus the "
                        "Instrumented Evaluation Contract: CIs, pre-registered MDE, and "
                        "anytime-valid confidence sequences before any number backs a claim."),
        "evidence": [PILOT_GATE, TRANSFER_GATE],
    },
    {
        "id": "attribution_and_responsibility",
        "name": "Attribution & responsibility",
        "principle": ("Results are attributable to specific humans who take "
                      "responsibility for their correctness; credit is not given to "
                      "automated systems."),
        "declared_status": "operationalized",
        "mechanisms": [
            "tools/lint_claims.py", "agi-proof/tool-disclosure.json",
            "docs/TOOL-DISCLOSURE.md", "training/adapters/registry.jsonl",
            "tools/validate_attribution.py",
        ],
        "sophia_form": ("canClaimAGI:false throughout; a generated Tool & Computational "
                        "Resource Disclosure; a no-AI-authorship lint; and abstention "
                        "when provenance cannot be established."),
        "evidence": ["docs/TOOL-DISCLOSURE.md"],
    },
    {
        "id": "transparency_and_verifiability",
        "name": "Transparency & independent verifiability",
        "principle": "Arguments are transparent and subject to independent verification.",
        "declared_status": "operationalized",
        "mechanisms": [
            "agi-proof/failure-ledger.md", "RESULTS.md",
            "tools/validate_failure_ledger.py", "agi-proof/REPLICATION.md",
        ],
        "sophia_form": ("Public failure ledger of what is NOT proven, aggregates-only "
                        "results published from a single source of truth, and a "
                        "clean-clone replication checklist."),
        "evidence": ["agi-proof/failure-ledger.md"],
    },
    {
        "id": "shared_standards",
        "name": "Shared standards of evaluation",
        "principle": "Work is evaluated by collectively established criteria.",
        # partial: the IEC is published as a reusable standard, but several benchmarks
        # are self-authored and not yet externally adopted.
        "declared_status": "partial",
        "mechanisms": [
            "agi-proof/measurement-thesis.md", "tools/claim_gate.py",
            "data/traditions.json",
        ],
        "sophia_form": ("The Instrumented Evaluation Contract (8 pillars) and the "
                        "PROTECTED per-domain standards (religion/history) are published "
                        "for reuse; external adoption and more non-self-authored "
                        "benchmarks remain open work."),
        "evidence": ["agi-proof/measurement-thesis.md"],
    },
    {
        "id": "autonomous_direction",
        "name": "Autonomous direction & non-proprietary tooling",
        "principle": ("Humans shape research directions; the field favors "
                      "non-proprietary, publicly governed tools."),
        # partial: local-first + Apache-2.0, but validation still leans on proprietary judges.
        "declared_status": "partial",
        "mechanisms": ["VISION.md", "LICENSE", "SECURITY.md",
                       "agent/judge_registry.py", "agent/open_judge.py"],
        "sophia_form": ("Local-first, Apache-2.0, runs on owned hardware. A self-hostable "
                        "open-weights judge backend (agent/open_judge.py) and a judge-openness "
                        "registry (agent/judge_registry.py) now exist; the value is fully met "
                        "once a headline result is corroborated on a non-proprietary path "
                        "(see open_gaps: open_model_judge_family)."),
        "evidence": ["VISION.md"],
    },
]

# Honest, explicitly-declared gaps (Leiden: state when something is not yet done).
OPEN_GAPS = [
    {
        "id": "open_model_judge_family",
        "summary": ("Add an open-weights judge family so the >=2-judge-family validation "
                    "rule no longer depends on proprietary inference services."),
        "serves_value": "autonomous_direction",
        "status": "in_progress",
        "note": ("Registry (agent/judge_registry.py) + self-hostable backend "
                 "(agent/open_judge.py) landed; remaining: corroborate a headline result on "
                 "a non-proprietary path and record judge openness in its receipt."),
        "design": "docs/11-Platform/Leiden-Open-Judge-Family.md",
    },
    {
        "id": "formal_proof_verifier_tier",
        "summary": ("Add a formal-proof (e.g. Lean) verifier tier so formal/mathematical "
                    "claims must carry a machine-checked artifact or be labeled "
                    "informal-unverified."),
        "serves_value": "proof_and_certainty",
        "status": "proposed",
        "design": "docs/11-Platform/Leiden-Formal-Proof-Tier.md",
    },
    {
        "id": "external_replication",
        "summary": ("Drive the hidden-reviewer / clean-clone external replication path to "
                    "reduce reliance on self-authored benchmarks."),
        "serves_value": "shared_standards",
        "status": "open",
        "design": "agi-proof/REPLICATION.md",
    },
]


def _judge_independence() -> dict:
    """Classify the committed headline judge panel for non-proprietary coverage, and report
    whether a self-hostable open judge backend is configured in this environment."""
    p = ROOT / PILOT_JUDGE
    panel: list[str] = []
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            panel = data.get("judges", []) if isinstance(data, dict) else []
        except Exception:
            panel = []
    summary = judge_registry.classify_panel(panel)
    # Static (env-independent) fact so the committed receipt does not drift with local config:
    # the self-hostable backend exists in the codebase. Runtime configuration is checked by
    # open_judge.available() at use time, not baked into the artifact.
    summary["open_judge_backend_present"] = (ROOT / "agent" / "open_judge.py").exists()
    summary["headline_panel_source"] = PILOT_JUDGE
    return summary


def _read_verdict(rel: str) -> str | None:
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("verdict")
    except Exception:
        return None


def build() -> dict:
    values_out = []
    for v in VALUES:
        missing = [m for m in v["mechanisms"] if not (ROOT / m).exists()]
        status = v["declared_status"]
        if missing and status == "operationalized":
            status = "partial"
        ev = []
        for e in v["evidence"]:
            rec = {"artifact": e, "exists": (ROOT / e).exists()}
            vd = _read_verdict(e)
            if vd is not None:
                rec["verdict"] = vd
            ev.append(rec)
        entry = {
            "id": v["id"], "name": v["name"], "principle": v["principle"],
            "status": status, "sophia_form": v["sophia_form"],
            "mechanisms": v["mechanisms"], "missing_mechanisms": missing,
            "evidence": ev,
        }
        if v["id"] == "autonomous_direction":
            entry["judge_independence"] = _judge_independence()
        values_out.append(entry)
    return {
        "_comment": ("GENERATED by tools/leiden_receipt.py — do not edit by hand. "
                     "Machine-readable map of Sophia mechanisms to the five Leiden values."),
        "declaration": DECLARATION,
        "canClaimAGI": False,
        "values": values_out,
        "open_gaps": OPEN_GAPS,
        "summary": {
            "operationalized": sum(1 for v in values_out if v["status"] == "operationalized"),
            "partial": sum(1 for v in values_out if v["status"] == "partial"),
            "open_gaps": len(OPEN_GAPS),
        },
    }


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if the receipt is stale")
    ap.add_argument("--print", dest="to_stdout", action="store_true",
                    help="print to stdout, write nothing")
    args = ap.parse_args(argv)

    content = _dump(build())
    if args.to_stdout:
        sys.stdout.write(content)
        return 0
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != content:
            print("LEIDEN RECEIPT: STALE — run `python tools/leiden_receipt.py` "
                  "and commit agi-proof/leiden-compliance.json")
            return 1
        print("LEIDEN RECEIPT: OK — agi-proof/leiden-compliance.json is up to date.")
        return 0
    OUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
