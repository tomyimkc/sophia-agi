#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Honest-closure ratchet — make gap-closure a MEASURED, un-farmable signal.

The failure ledger (agi-proof/failure-ledger.md) is this repo's honesty spine: closing a
failure row is supposed to mean the gap was really addressed. But "closure" is farmable — a
row can be flipped to "Closed" by writing a bare "honest N=0" / "honest NEGATIVE" note with
NO independently-checkable evidence, or by flooding the ledger with cheap negative closures so
the raw closed-count climbs while nothing was actually proven. This tool refuses to let the
closed-count be a headline on its own.

What it does (all MEASURED from the ledger text — it makes NO claim about whether the repo's
actual closure rate is good or bad; it only computes the number):

  1. Parse every REAL failure row (kebab-case id with a YYYY-MM-DD suffix; ignores the
     sub-tables embedded inside "Required response" cells, which share the pipe delimiter).
  2. Classify each row:
       * open              — Status starts Open / Partial / Superseded / ... (not closed)
       * closed-positive   — Status is Closed AND the row cites a POWERED receipt: an artifact
                             path (*.public-report.json / *.json / *.md / *.py) or a
                             PASS/GO/VALIDATED marker.
       * closed-negative   — Status is Closed AND carries an honest-NEGATIVE / honest-null /
                             decontam-exhausted / honest-N=0 marker (a gap declared unwinnable
                             from the current evidence, which is a legitimate honest outcome).
  3. ANTI-FARMING guard (a) — the critic's warning: every closed-NEGATIVE row MUST carry an
     INDEPENDENTLY-CHECKABLE reason token (references assert_decontam, or a findings.json /
     PRE-REGISTRATION artifact path, or an explicit reproduce command). A closed-negative with
     no such token is flagged 'unverifiable-negative' — a bare assertion, not a checkable one.
  4. ANTI-FARMING guard (b) — ALARM if the ratio of negative-closures to RECEIPTED
     positive-closures rises above a pre-registered threshold (default 1.0): honest negatives
     are cheap to write, so if they start to outnumber receipted positives the closed-count is
     being inflated by the cheap side of the ledger.
  5. honestClosureRate — a Robbins ANYTIME-VALID confidence sequence (eval_stats.
     confidence_sequence_mean) over a PER-RELEASE closure-quality series. Each release (a
     distinct date suffix) contributes one quality score in [0,1] = receipted-quality of its
     closed rows (verifiable closures / all closed rows in that release). The CS is peeking-
     robust: the ledger is appended to continuously, so a fixed-n CI would be invalid here.

Exit codes: 0 = pass (no alarms), 1 = alarm (unverifiable-negative rows and/or ratio breach),
2 = unreadable/missing ledger. Prints a JSON receipt to stdout; human prose to stderr.

NO-OVERCLAIM: this is a MEASUREMENT instrument. It reports counts, the CS interval, and any
alarms. It does NOT assert the closure rate is acceptable — a downstream gate + human decide
against the pre-registered thresholds echoed in the receipt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# eval_stats lives in tools/ next to this file; import it whether run as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_stats  # noqa: E402

# A real failure-ledger row id: kebab-case tokens ending in a YYYY-MM-DD date suffix. The
# sub-tables embedded inside "Required response" cells never match this, so this is the robust
# row filter (the ledger reuses the pipe delimiter for those nested tables).
_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-\d{4}-\d{2}-\d{2}$")

# Status that means the row is CLOSED (resolved one way or another), vs still open/partial.
_CLOSED_RE = re.compile(r"^\s*(?:closed|resolved|validated|cleared)\b", re.I)

# honest-NEGATIVE family markers (a legitimately-closed gap declared unwinnable / null).
_NEGATIVE_RE = re.compile(
    r"honest\s+(?:negative|null)|decontam[- ]?exhausted|"
    r"honest\s+n\s*=\s*0|largest\s+honest\s+n\s*=\s*0|source[- ]exhausted",
    re.I,
)

# An artifact path cited in the row (a receipt the reader can open and check).
_PATH_RE = re.compile(r"[\w][\w./-]*\.(?:json|md|py)\b")

# A POWERED-receipt / verdict marker for a POSITIVE closure.
_POSITIVE_RECEIPT_RE = re.compile(
    r"\bPASS\b|\bGO\b(?!-)|\bVALIDATED\b|\bCONFIRMED\b|\bM1 GO\b|\bRAN\b", re.I
)

# An INDEPENDENTLY-CHECKABLE reason token that makes a NEGATIVE closure verifiable: it names a
# decontam check, a findings/pre-registration artifact, or an explicit reproduce command.
_CHECKABLE_RE = re.compile(
    r"assert_decontam|findings\.json|pre[- ]?registration|PRE-REGISTRATION|"
    r"reproduce:|\baudit_[\w]+\.py|measurement_spec\.json",
    re.I,
)


def _parse_rows(text: str) -> "list[dict]":
    """Extract real failure rows from the ledger markdown. Returns a list of dicts with keys
    id, status, full (the whole row text after the id, for token scanning)."""
    rows: list[dict] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        fid = cells[0]
        if not _ID_RE.match(fid):
            continue
        rows.append({"id": fid, "status": cells[1], "full": " ".join(cells[1:])})
    return rows


def _release_of(fid: str) -> str:
    """Release proxy = the trailing YYYY-MM-DD date suffix of the row id."""
    return fid[-10:]


def classify(row: dict) -> dict:
    """Classify one row and attach evidence signals.

    Returns the row augmented with:
      cls          — 'open' | 'closed-positive' | 'closed-negative'
      hasPath      — cites at least one artifact path
      hasReceipt   — cites a path OR a PASS/GO/VALIDATED marker (a powered positive receipt)
      isNegative   — carries an honest-NEGATIVE / null / decontam-exhausted marker
      checkable    — carries an independently-checkable reason token
      unverifiable — closed-negative with NO checkable token (the farming flag)
    """
    status, full = row["status"], row["full"]
    closed = bool(_CLOSED_RE.match(status))
    negative = bool(_NEGATIVE_RE.search(full))
    has_path = bool(_PATH_RE.search(full))
    has_receipt = has_path or bool(_POSITIVE_RECEIPT_RE.search(full))
    checkable = bool(_CHECKABLE_RE.search(full)) or has_path

    if not closed:
        cls = "open"
    elif negative:
        cls = "closed-negative"
    else:
        cls = "closed-positive"

    unverifiable = cls == "closed-negative" and not checkable
    out = dict(row)
    out.update(
        cls=cls,
        release=_release_of(row["id"]),
        hasPath=has_path,
        hasReceipt=has_receipt,
        isNegative=negative,
        checkable=checkable,
        unverifiable=unverifiable,
    )
    return out


def _per_release_quality(rows: "list[dict]") -> "tuple[list[str], list[float]]":
    """Per-release closure-quality series in [0,1]: for each release that has >=1 CLOSED row,
    quality = (verifiable closed rows) / (all closed rows in that release). A closed row is
    'verifiable' when it is a receipted positive OR a checkable negative; an unverifiable
    negative counts as 0 quality. Releases with no closures are skipped (no closure signal)."""
    by_rel: dict[str, list[dict]] = {}
    for r in rows:
        if r["cls"] in ("closed-positive", "closed-negative"):
            by_rel.setdefault(r["release"], []).append(r)
    releases = sorted(by_rel)
    series: list[float] = []
    for rel in releases:
        closed = by_rel[rel]
        good = 0
        for r in closed:
            if r["cls"] == "closed-positive" and r["hasReceipt"]:
                good += 1
            elif r["cls"] == "closed-negative" and not r["unverifiable"]:
                good += 1
        series.append(good / len(closed))
    return releases, series


def evaluate(text: str, *, negative_ratio_threshold: float = 1.0, alpha: float = 0.05) -> dict:
    """Run the full measurement over ledger `text`. Pure function — no I/O, no exit."""
    rows = [classify(r) for r in _parse_rows(text)]
    counts = {
        "total": len(rows),
        "open": sum(1 for r in rows if r["cls"] == "open"),
        "closedPositive": sum(1 for r in rows if r["cls"] == "closed-positive"),
        "closedNegative": sum(1 for r in rows if r["cls"] == "closed-negative"),
        "receiptedPositive": sum(
            1 for r in rows if r["cls"] == "closed-positive" and r["hasReceipt"]
        ),
        "unverifiableNegative": sum(1 for r in rows if r["unverifiable"]),
    }

    unverifiable_ids = [r["id"] for r in rows if r["unverifiable"]]

    # Guard (b): negative-closures vs RECEIPTED positive-closures.
    neg = counts["closedNegative"]
    recpos = counts["receiptedPositive"]
    ratio = (neg / recpos) if recpos > 0 else (float("inf") if neg > 0 else 0.0)
    ratio_report = None if ratio == float("inf") else round(ratio, 4)

    alarms: list[str] = []
    if unverifiable_ids:
        alarms.append(
            f"unverifiable-negative: {len(unverifiable_ids)} closed-NEGATIVE row(s) carry no "
            f"independently-checkable reason token (assert_decontam / findings.json / "
            f"PRE-REGISTRATION / reproduce command / artifact path)"
        )
    if ratio == float("inf") or ratio > negative_ratio_threshold + 1e-9:
        shown = "inf (no receipted positive closures)" if ratio == float("inf") else round(ratio, 4)
        alarms.append(
            f"negative-closure ratio {shown} exceeds pre-registered threshold "
            f"{negative_ratio_threshold}: honest negatives are outnumbering receipted positives — "
            f"the closed-count is being inflated by the cheap side of the ledger"
        )

    releases, series = _per_release_quality(rows)
    cs = eval_stats.confidence_sequence_mean(series, alpha=alpha) if series else [None, None]
    mean = round(sum(series) / len(series), 4) if series else None

    return {
        "experimentId": "honest-closure-ratchet",
        "measures": "per-release closure quality (receipted/checkable closures over all closures)",
        "counts": counts,
        "honestClosureRate": {
            "perReleaseMean": mean,
            "confidenceSequence": cs,
            "alpha": alpha,
            "nReleases": len(series),
            "series": [round(s, 4) for s in series],
            "releases": releases,
            "method": "Robbins anytime-valid confidence sequence (eval_stats.confidence_sequence_mean)",
            "note": "anytime-valid: the ledger is append-only so a fixed-n CI would be invalid",
        },
        "antiFarming": {
            "negativeRatioThreshold": negative_ratio_threshold,
            "negativeToReceiptedPositiveRatio": ratio_report,
            "unverifiableNegativeIds": unverifiable_ids,
            "guardA": "every closed-NEGATIVE row must cite an independently-checkable reason token",
            "guardB": "negative-closures must not exceed receipted-positive-closures by the threshold",
        },
        "alarms": alarms,
        "pass": len(alarms) == 0,
        "canClaimAGI": False,
        "claimBoundary": (
            "This tool MEASURES closure quality from the ledger text. It does NOT assert the "
            "repo's closure rate is good or bad; a downstream gate + human judge against the "
            "pre-registered thresholds echoed here."
        ),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Honest-closure ratchet gate over the failure ledger.")
    ap.add_argument(
        "--ledger",
        default=str(Path(__file__).resolve().parents[1] / "agi-proof" / "failure-ledger.md"),
        help="path to failure-ledger.md",
    )
    ap.add_argument(
        "--negative-ratio-threshold",
        type=float,
        default=1.0,
        help="pre-registered max ratio of negative-closures to receipted-positive-closures",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="confidence-sequence miscoverage")
    args = ap.parse_args(argv)

    path = Path(args.ledger)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read ledger at {path}: {exc}", file=sys.stderr)
        print(json.dumps({"error": "unreadable-ledger", "path": str(path)}), file=sys.stdout)
        return 2

    result = evaluate(
        text,
        negative_ratio_threshold=args.negative_ratio_threshold,
        alpha=args.alpha,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    c = result["counts"]
    print(
        f"[honest-closure] rows={c['total']} open={c['open']} closed+={c['closedPositive']} "
        f"(receipted {c['receiptedPositive']}) closed-={c['closedNegative']} "
        f"unverifiable-neg={c['unverifiableNegative']}",
        file=sys.stderr,
    )
    print(
        f"[honest-closure] closureRate mean={result['honestClosureRate']['perReleaseMean']} "
        f"CS={result['honestClosureRate']['confidenceSequence']} "
        f"over {result['honestClosureRate']['nReleases']} releases",
        file=sys.stderr,
    )
    if result["alarms"]:
        for a in result["alarms"]:
            print(f"[honest-closure] ALARM: {a}", file=sys.stderr)
        return 1
    print("[honest-closure] PASS: no farming alarms", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
