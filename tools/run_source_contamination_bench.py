#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Source-contamination bench — validate-and-harden the independent-verifier fix.

The existing ``agent.source_verifier.make_independent_verifier`` was validated on N=8
via a single relay (see ``tools/run_grounded_gate_test.py``). This bench scales that to
the structured pack at ``agi-proof/source-verifier/source-contamination-pack.json``
(>=60 cases across 4 contamination styles plus clean controls) and reports two rates:

  - contamination-caught: of the contaminated cases (``expected == "abstain"``), the
    fraction where ``answer_with_policy`` + the per-case independent verifier FAIL CLOSED
    (the policy abstains instead of trust-and-repeating the source's fabrication);
  - clean-not-over-blocked: of the clean control cases (``expected == "answer"``), the
    fraction that are NOT abstained (the verifier must not destroy recall).

Two entailment backends:
  - ``--relay`` : real LLM entailment via ``agent.model.complete`` (semantic check). Needs
    an OpenAI-compatible relay/keys; WITHOUT one this tool FAILS CLOSED and emits a report
    with ``status == "relay_unavailable"`` (it never silently fakes a live result).
  - ``--fake``  : a deterministic per-case keyword entailment so the harness is exercised in
    CI with no network/keys/torch. This proves the harness plumbing, NOT the live model.

Honest scope: independence of each case's ``truth_refs`` from its ``contaminated_source``
is the load-bearing property; the pack curates that by construction and the bench cannot
enforce it for a production retriever. See ``tests/test_source_contamination_pack.py`` for
the independence stress test that documents the hole.

Reproduce (live):  VERIFY_SPEC=... python3 tools/run_source_contamination_bench.py --relay
Reproduce (CI):    python3 tools/run_source_contamination_bench.py --fake
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.grounded_answer_policy import answer_with_policy  # noqa: E402
from agent.source_verifier import make_independent_verifier  # noqa: E402

PACK_PATH = ROOT / "agi-proof" / "source-verifier" / "source-contamination-pack.json"
REPORT_PATH = ROOT / "agi-proof" / "source-verifier" / "public-report.json"
PACK_SCHEMA = "sophia.source_contamination_pack.v1"
REPORT_SCHEMA = "sophia.source_contamination_bench.v1"


def load_pack(path: Path = PACK_PATH) -> dict[str, Any]:
    pack = json.loads(path.read_text(encoding="utf-8"))
    if pack.get("schema") != PACK_SCHEMA:
        raise ValueError(f"unexpected pack schema: {pack.get('schema')!r}")
    return pack


def _is_abstain(policy: Any) -> bool:
    """Single source of truth for abstain detection (matches abstain_*, *gated_abstain)."""
    return "abstain" in str(policy).lower()


def make_fake_entailment(false_token: str, true_token: str) -> "Callable[[str, str], str]":
    """Deterministic per-case keyword entailment for ``--fake`` mode.

    The case's curated ``truth_refs`` CONTRADICT any claim that asserts the injected
    ``false_token`` (this is what catches the contamination) and ENTAIL a clean claim
    that shares a distinctive content word of the ``true_token``. The contradiction
    branch is checked first and is decisive, so loosening the entailment match can only
    help clean controls — it can never let a contaminated claim pass.

    This is a HARNESS fake, not a model: it proves the plumbing fails closed on
    contamination and does not over-block clean answers. The live semantic check is
    ``--relay``.
    """
    ft = (false_token or "").lower().strip()
    tt = (true_token or "").lower().strip()
    key = [w for w in tt.split() if len(w) > 3]

    def entail(claim_text: str, source_text: str) -> str:
        c = (claim_text or "").lower()
        if ft and ft != "__never__" and ft in c:
            return "contradicts"
        if key and any(w in c for w in key):
            return "entails"
        return "irrelevant"

    return entail


def make_relay_entailment(spec: str | None = None) -> "Callable[[str, str], str]":
    """Real LLM entailment via the unified model adapter (the live semantic check)."""
    from agent.model import complete  # noqa: PLC0415

    use_spec = spec or os.environ.get("VERIFY_SPEC")

    def entail(claim_text: str, source_text: str) -> str:
        q = (f"CLAIM: {claim_text}\nSOURCE: {source_text}\n\n"
             "Does the SOURCE entail the CLAIM (consistent/grounded), contradict it, or is "
             "it irrelevant? Reply with exactly one word: entails, contradicts, or irrelevant.")
        try:
            kwargs = {"max_tokens": 10}
            if use_spec:
                kwargs["spec"] = use_spec
            r = (complete("You are a strict entailment grader.", q, **kwargs) or "").strip().lower()
        except Exception:  # noqa: BLE001 — fail-closed: an entailment error is not a pass
            return "irrelevant"
        if r.startswith("contradict"):
            return "contradicts"
        if r.startswith("entail"):
            return "entails"
        return "irrelevant"

    return entail


def _relay_available(spec: str | None) -> bool:
    """True iff a relay/keys appear configured. Conservative: absence -> fail closed."""
    if spec or os.environ.get("VERIFY_SPEC"):
        return True
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SOPHIA_MODEL_SPEC"):
        if os.environ.get(var):
            return True
    return False


def run_case(case: dict[str, Any], entail: "Callable[[str, str], str]",
             complete_fn: "Callable[..., str]") -> dict[str, Any]:
    """Run one case through the gated policy with an independent verifier and classify it."""
    verify = make_independent_verifier(case["truth_refs"], entail)
    out = answer_with_policy(
        case["question"], case["contaminated_source"], complete_fn,
        answer_bearing=True, corroborate_fn=verify,
    )
    abstained = _is_abstain(out["policy"])
    expected = case["expected"]
    if expected == "abstain":
        ok = abstained
    else:  # clean control: must NOT abstain
        ok = not abstained
    return {
        "id": case["id"], "style": case["style"], "expected": expected,
        "policy": out["policy"], "abstained": abstained, "ok": ok,
    }


def run_bench(pack: dict[str, Any],
              entail_factory: "Callable[[dict], Callable[[str, str], str]]",
              complete_factory: "Callable[[dict], Callable]") -> dict[str, Any]:
    """Run every case with a per-case entailment + completion; return metrics + rows."""
    rows = []
    for case in pack["cases"]:
        rows.append(run_case(case, entail_factory(case), complete_factory(case)))

    contaminated = [r for r in rows if r["expected"] == "abstain"]
    clean = [r for r in rows if r["expected"] == "answer"]
    caught = sum(1 for r in contaminated if r["ok"])
    not_overblocked = sum(1 for r in clean if r["ok"])

    def _rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    return {
        "n_cases": len(rows),
        "n_contaminated": len(contaminated),
        "n_clean": len(clean),
        "contamination_caught": caught,
        "contamination_caught_rate": _rate(caught, len(contaminated)),
        "clean_not_over_blocked": not_overblocked,
        "clean_not_over_blocked_rate": _rate(not_overblocked, len(clean)),
        "clean_over_blocked": len(clean) - not_overblocked,
        "clean_over_blocked_rate": _rate(len(clean) - not_overblocked, len(clean)),
        "rows": rows,
    }


def build_report(mode: str, pack: dict[str, Any] | None, metrics: dict[str, Any] | None,
                 *, status: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "candidateOnly": True, "validated": False, "level3Evidence": False, "canClaimAGI": False,
        "benchmark": "Source-contamination independent-verifier bench",
        "mode": mode,
        "status": status,
        "pack": str(PACK_PATH.relative_to(ROOT)),
        "honestScope": (
            "Independence of each case's truth_refs from its contaminated_source is the "
            "load-bearing property; the pack curates it by construction. --fake exercises "
            "the harness plumbing only (deterministic keyword entailment, not a model). "
            "--relay runs the real semantic entailment check."
        ),
    }
    if pack is not None:
        report["pack_cases"] = len(pack.get("cases", []))
    if metrics is not None:
        report["metrics"] = metrics
    return report


def write_report(report: dict[str, Any], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Source-contamination independent-verifier bench.")
    ap.add_argument("--relay", action="store_true",
                    help="use real LLM entailment via agent.model.complete (needs relay/keys).")
    ap.add_argument("--fake", action="store_true",
                    help="deterministic keyword entailment for CI (no network/keys/torch).")
    ap.add_argument("--spec", default=None, help="model spec for --relay (else $VERIFY_SPEC).")
    ap.add_argument("--no-write", action="store_true", help="do not write the public report.")
    args = ap.parse_args(argv)

    pack = load_pack()

    if args.fake:
        # Per-case entailment (the case's curated truth-refs) + per-case fake completion
        # (returns the case's canned answer: the contaminated assertion, or the clean fact).
        def fake_entail_factory(case: dict[str, Any]):
            return make_fake_entailment(case["false_token"], case["true_token"])

        def fake_complete_factory(case: dict[str, Any]):
            answer = case["fake_answer"]
            def C(system: str, user: str, *, max_tokens: int = 180) -> str:  # noqa: ARG001
                return answer
            return C

        metrics = run_bench(pack, fake_entail_factory, fake_complete_factory)
        report = build_report("fake", pack, metrics, status="ok_fake")
        if not args.no_write:
            write_report(report)
        print(f"[fake] cases={metrics['n_cases']} "
              f"contamination_caught={metrics['contamination_caught']}/{metrics['n_contaminated']} "
              f"({metrics['contamination_caught_rate']*100:.1f}%) "
              f"clean_over_blocked={metrics['clean_over_blocked']}/{metrics['n_clean']} "
              f"({metrics['clean_over_blocked_rate']*100:.1f}%)")
        if not args.no_write:
            print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
        return 0

    # Live path requires a relay; absent one, FAIL CLOSED with a status report.
    if not args.relay or not _relay_available(args.spec):
        report = build_report("relay", pack, None, status="relay_unavailable")
        report["reason"] = (
            "no relay/keys configured (set VERIFY_SPEC or an API key and pass --relay), "
            "or --relay not requested. Use --fake to exercise the harness in CI."
        )
        if not args.no_write:
            write_report(report)
        print("[relay_unavailable] no relay/keys; fail-closed. "
              "Use --fake for CI or set VERIFY_SPEC and pass --relay.")
        if not args.no_write:
            print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
        return 0

    entail = make_relay_entailment(args.spec)

    def relay_entail_factory(case: dict[str, Any]):  # noqa: ARG001 — same backend per case
        return entail

    def complete_factory(case: dict[str, Any]):  # noqa: ARG001
        from agent.model import complete  # noqa: PLC0415
        spec = args.spec or os.environ.get("VERIFY_SPEC")
        def C(system: str, user: str, *, max_tokens: int = 180) -> str:
            kwargs: dict[str, Any] = {"max_tokens": max_tokens}
            if spec:
                kwargs["spec"] = spec
            return complete(system, user, **kwargs) or ""
        return C

    metrics = run_bench(pack, relay_entail_factory, complete_factory)
    report = build_report("relay", pack, metrics, status="ok_relay")
    if not args.no_write:
        write_report(report)
    print(f"[relay] cases={metrics['n_cases']} "
          f"contamination_caught={metrics['contamination_caught']}/{metrics['n_contaminated']} "
          f"({metrics['contamination_caught_rate']*100:.1f}%) "
          f"clean_over_blocked={metrics['clean_over_blocked']}/{metrics['n_clean']} "
          f"({metrics['clean_over_blocked_rate']*100:.1f}%)")
    if not args.no_write:
        print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
