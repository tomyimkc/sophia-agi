#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Retrieval-grounded, claim-decomposed SUPPORT check (process-level, not existence).

THE GAP THIS CLOSES
-------------------
The deployment gate verifies that a cited source EXISTS (the Mata v. Avianca
guardrail in agent/verifiers.py: legal_citation_exists). Existence is necessary
but not sufficient: a real source can be cited for a proposition it does NOT
support — a present-but-non-supporting citation. The held-out reference set marks
exactly these (data/reference_holdout_traps.json, category
``non_supporting_citation``), and the existence gate is blind to them by design.

This tool adds the PROCESS-level check the literature calls for — decompose,
retrieve, then judge ENTAILMENT, not mere presence (the CiteAudit / Med-V1
"does the source actually support the claim" pattern — post-cutoff, verify):

  1. DECOMPOSE the answer into atomic claims (one checkable assertion each).
  2. RETRIEVE candidate evidence per claim. Hooks okf RAG lazily
     (okf/rag_pipeline.py or okf/retrieval.py if importable; else the repo
     retrieval in agent/retrieval.py); falls back to inline --evidence so it runs
     fully offline.
  3. JUDGE whether the evidence ENTAILS the claim: SUPPORTED / NOT_SUPPORTED /
     CONTRADICTED / NO_EVIDENCE. Deterministic lexical-overlap + numeric-equality
     + negation heuristic by default; an optional --model adds an LLM adjudicator
     (still fail-closed: a model that errs or hedges does NOT upgrade a verdict).

Fail-closed: NO_EVIDENCE and NOT_SUPPORTED both fail. An abstention
("insufficient verified basis") is recognised and scored as CORRECT, never a
violation — abstention is a valid answer, never below a fabrication.

    python tools/support_check.py --dry-run
    python tools/support_check.py --answer "Cap. 486 s.33 is in force." \
        --evidence "Section 33 of Cap. 486 has NOT been brought into operation." --model mock
    python tools/support_check.py --in answer.txt --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NO_EVIDENCE = "NO_EVIDENCE"

_NEG = ("not ", "no ", "never", "without", "fails to", "has not", "did not",
        "並未", "並非", "沒有", "未能", "未曾")
_STOP = {"the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "that",
         "this", "it", "for", "on", "by", "as", "be", "was", "were", "至", "的"}


def _tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z一-鿿]{2,}|\d[\d.,]*", (text or "").lower())
    return {t for t in toks if t not in _STOP}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d.,]*", text or "")}


def is_abstention(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in (
        "insufficient verified basis", "未能核實", "cannot attribute", "cannot verify",
    ))


# --------------------------------------------------------------------------- #
# 1) Decompose into atomic claims
# --------------------------------------------------------------------------- #


def decompose(answer: str) -> list[str]:
    """Split an answer into atomic claims. Deterministic: sentence/clause split,
    drop pure framing lines (中文 summary, disclaimers, source-discipline notes)."""
    if not answer:
        return []
    # Protect abbreviation periods (Cap. / s. / No. / v. / ss. / r.) so a legal
    # citation is not torn into two pseudo-claims at its internal full stops.
    protected = re.sub(
        r"\b(Cap|cap|s|ss|r|No|no|art|Art|v|cf|al|para|Pt|pt)\.",
        lambda m: m.group(1) + "\x00", answer,
    )
    # Normalise CJK and latin sentence terminators to a common delimiter.
    norm = re.sub(r"[。！？;]", ".", protected)
    raw = [p.replace("\x00", ".") for p in re.split(r"(?<=[.])\s+|\n+", norm)]
    claims: list[str] = []
    for piece in raw:
        s = piece.strip().rstrip(".").strip()
        if len(s) < 6:
            continue
        low = s.lower()
        if low.startswith(("中文", "來源", "source discipline", "not professional advice")):
            continue
        if is_abstention(s):
            continue  # an abstaining clause is not a checkable factual claim
        claims.append(s)
    return claims


# --------------------------------------------------------------------------- #
# 2) Retrieve candidate evidence
# --------------------------------------------------------------------------- #


def retrieve_evidence(claim: str, *, inline: list[str], top_k: int = 5) -> list[str]:
    """Inline evidence wins (offline, deterministic, reproducible). Otherwise hook
    okf RAG lazily, then fall back to the repo retrieval; any failure degrades to
    [] so the verdict becomes NO_EVIDENCE (fail-closed), never a crash."""
    if inline:
        return inline
    # okf RAG pipeline (preferred per spec).
    for modpath, fn in (("okf.rag_pipeline", "retrieve"), ("okf.retrieval", "retrieve")):
        try:
            mod = __import__(modpath, fromlist=[fn])
            hits = getattr(mod, fn)(claim, top_k=top_k)
            out = [_hit_text(h) for h in hits if _hit_text(h)]
            if out:
                return out
        except Exception:  # noqa: BLE001 - okf may be absent/incompatible; degrade
            pass
    # Repo retrieval fallback.
    try:
        from agent.retrieval import retrieve

        return [c.excerpt for c in retrieve(claim, top_k=top_k) if getattr(c, "excerpt", "")]
    except Exception:  # noqa: BLE001
        return []


def _hit_text(hit) -> str:
    for attr in ("excerpt", "text", "content", "body"):
        v = getattr(hit, attr, None)
        if v:
            return str(v)
    if isinstance(hit, dict):
        for k in ("excerpt", "text", "content", "body"):
            if hit.get(k):
                return str(hit[k])
    return str(hit) if hit else ""


# --------------------------------------------------------------------------- #
# 3) Judge entailment
# --------------------------------------------------------------------------- #


def judge_entailment(claim: str, evidence: list[str], *, client=None) -> dict:
    """Deterministic entailment heuristic (optionally adjudicated by a model).

    Heuristic: a claim is SUPPORTED only if the evidence covers its content tokens
    AND every NUMBER in the claim appears in the evidence (numeric fabrication is
    the classic non-supporting case) AND polarity is not flipped. A flipped
    polarity on shared content is CONTRADICTED. Thin coverage is NOT_SUPPORTED.
    Empty evidence is NO_EVIDENCE. Fail-closed throughout.
    """
    if not evidence:
        return {"verdict": NO_EVIDENCE, "overlap": 0.0, "method": "heuristic",
                "reason": "no candidate evidence retrieved"}

    ctoks = _tokens(claim)
    blob = " ".join(evidence)
    etoks = _tokens(blob)
    overlap = len(ctoks & etoks) / len(ctoks) if ctoks else 0.0

    cnums, enums = _numbers(claim), _numbers(blob)
    numbers_ok = cnums.issubset(enums)
    missing_numbers = sorted(cnums - enums)

    claim_neg = any(n in claim.lower() for n in _NEG)
    eved_neg = any(n in blob.lower() for n in _NEG)
    polarity_flip = (claim_neg != eved_neg) and overlap >= 0.5

    if overlap >= 0.55 and numbers_ok and not polarity_flip:
        verdict, reason = SUPPORTED, "evidence covers claim content and numerics"
    elif polarity_flip:
        verdict, reason = CONTRADICTED, "evidence shares content but flips polarity (negation)"
    elif not numbers_ok:
        verdict, reason = NOT_SUPPORTED, f"claim numbers absent from evidence: {missing_numbers}"
    else:
        verdict, reason = NOT_SUPPORTED, f"insufficient evidence coverage (overlap={overlap:.2f})"

    result = {"verdict": verdict, "overlap": round(overlap, 3), "numbersOk": numbers_ok,
              "missingNumbers": missing_numbers, "polarityFlip": polarity_flip,
              "method": "heuristic", "reason": reason}

    if client is not None:
        result = _model_adjudicate(claim, evidence, result, client)
    return result


def _model_adjudicate(claim: str, evidence: list[str], base: dict, client) -> dict:
    """Optional LLM adjudicator. Fail-closed: the model may only CONFIRM a fail or
    DOWNGRADE a heuristic SUPPORTED; it can never upgrade NOT_SUPPORTED/NO_EVIDENCE
    to SUPPORTED (a hallucinating judge must not rescue an unsupported claim)."""
    system = (
        "You are a strict entailment judge. Reply with exactly one token: "
        "SUPPORTED, NOT_SUPPORTED, CONTRADICTED, or NO_EVIDENCE. SUPPORTED only if "
        "the evidence directly entails the claim."
    )
    user = "CLAIM:\n" + claim + "\n\nEVIDENCE:\n" + "\n---\n".join(evidence)
    try:
        res = client.generate(system, user)
        verdict = (getattr(res, "text", "") or "").strip().upper()
    except Exception:  # noqa: BLE001
        verdict = ""
    valid = {SUPPORTED, NOT_SUPPORTED, CONTRADICTED, NO_EVIDENCE}
    model_verdict = next((v for v in valid if v in verdict), None)
    out = dict(base)
    out["modelVerdict"] = model_verdict
    out["method"] = "heuristic+model"
    if model_verdict and model_verdict != SUPPORTED:
        # Model is allowed to tighten (downgrade), never to loosen.
        out["verdict"] = model_verdict
        out["reason"] = base["reason"] + f"; model downgraded to {model_verdict}"
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def support_check(answer: str, *, inline_evidence: list[str], client=None, top_k: int = 5) -> dict:
    if is_abstention(answer):
        return {"abstained": True, "passed": True, "claims": [],
                "note": "abstention is a CORRECT output — fail-closed; no claims to support"}
    claims = decompose(answer)
    per_claim: list[dict] = []
    for claim in claims:
        evidence = retrieve_evidence(claim, inline=inline_evidence, top_k=top_k)
        verdict = judge_entailment(claim, evidence, client=client)
        per_claim.append({"claim": claim, "evidenceCount": len(evidence), **verdict})
    failing = [c for c in per_claim if c["verdict"] in (NOT_SUPPORTED, CONTRADICTED, NO_EVIDENCE)]
    return {
        "abstained": False,
        "passed": len(failing) == 0 and bool(per_claim),
        "claimCount": len(per_claim),
        "supported": sum(1 for c in per_claim if c["verdict"] == SUPPORTED),
        "failing": len(failing),
        "claims": per_claim,
        "note": ("SUPPORT (process-level) check: a present source that does not ENTAIL "
                 "the claim still fails — beyond the existence gate."),
    }


def render(report: dict) -> str:
    if report.get("abstained"):
        return "[support_check] answer ABSTAINED — correct output, fail-closed. No claims to check."
    lines = [
        "[support_check] claim-decomposed SUPPORT verdicts",
        f"  claims={report['claimCount']} supported={report['supported']} "
        f"failing={report['failing']} passed={report['passed']}",
    ]
    for c in report["claims"]:
        lines.append(f"    - [{c['verdict']}] (overlap={c.get('overlap')}, ev={c['evidenceCount']}) "
                     f"{c['claim'][:80]}")
        lines.append(f"        reason: {c.get('reason')}")
    lines.append("  " + report["note"])
    return "\n".join(lines)


def _read_answer(args) -> str:
    if args.answer:
        return args.answer
    if getattr(args, "in_path", None):
        return Path(args.in_path).read_text(encoding="utf-8")
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Claim-decomposed retrieval-grounded SUPPORT check.")
    ap.add_argument("--answer", help="answer text to check")
    ap.add_argument("--in", dest="in_path", help="file containing the answer text")
    ap.add_argument("--evidence", action="append", default=[],
                    help="inline evidence snippet (repeatable); offline, overrides retrieval")
    ap.add_argument("--model", default=None, help="optional model spec for the LLM adjudicator (e.g. mock)")
    ap.add_argument("--top-k", type=int, default=5, help="retrieval depth when no inline evidence")
    ap.add_argument("--json", action="store_true", help="emit JSON report")
    ap.add_argument("--dry-run", action="store_true", help="validate plumbing on a fixed example; no retrieval/model")
    args = ap.parse_args(argv)

    if args.dry_run:
        demo = ("Section 33 of Cap. 486 is currently in force and prohibits all transfers. "
                "中文摘要：條例。")
        evidence = ["Section 33 of Cap. 486 has NOT been brought into operation; no commencement date appointed."]
        report = support_check(demo, inline_evidence=evidence, client=None)
        report["dryRun"] = True
        print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else render(report), flush=True)
        return 0

    answer = _read_answer(args)
    if not answer.strip():
        print("[support_check] no answer provided (use --answer or --in)", file=sys.stderr, flush=True)
        return 2

    client = None
    if args.model:
        from agent.model import default_client  # lazy

        client = default_client(args.model)

    report = support_check(answer, inline_evidence=list(args.evidence), client=client, top_k=args.top_k)
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else render(report), flush=True)
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
