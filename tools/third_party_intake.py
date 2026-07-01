#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Third-party verifiable-domain intake pipeline (REAL decontam-gated scaffold).

The verifier-as-reward loop is starved of *externally-authored* verifiable domains: the
committed third-party held-out pack is EMPTY by design (``caseCount: 0``), so the honest
count of closed-loop external domains is N=0. This tool is the intake side of closing that
gap: it ingests an externally-authored verifiable-task manifest (a mathlib theorem slice, a
GitHub CI suite, a legal-citation corpus, ...), where each item carries a decontamination
proof, and it ADMITS only the items that survive a real decontamination check.

What is REAL here (and tested):
  * The decontam gate. Each item's prompt is checked against the *committed* eval prompt
    surface using the SAME primitives as ``tools/assert_decontam.py`` (import ``normalize``,
    ``_shingles``, ``_jaccard``). An exact/normalized match OR a near-duplicate at/above the
    Jaccard threshold is REFUSED — the item never enters the admitted set. A gate that always
    passes is worse than no gate; this one fail-closes.
  * The validity gate. An item is only eligible if it has a machine-checkable oracle
    (``scoring.gold`` for math or ``scoring.test`` for code). LLM-judged items are not admitted
    to the headline intake (a judge is a separate, labelled family).
  * The TWO STRICTLY SEPARATE counters:
        admittedCount    = items that passed validity + decontam and entered the pool.
        loopClosedCount  = items for which the verifier ADMITTED the item AND a held-out gain
                           was actually measured on it (verifier-admitted AND held-out gain).
    These are never conflated. ``admittedCount`` measures *intake capacity*; only
    ``loopClosedCount`` can support any generalization claim. Admitting an item does NOT close
    the loop.

What is NOT proven (pre-registered):
  * ``loopClosedCount`` stays 0. Closing the loop requires (a) a real external corpus committed
    under ``agi-proof/third-party-heldout/`` and (b) a powered verifier run measuring held-out
    gain (GPU + external corpora + no in-session network). None exist in-repo, so the honest
    value is 0 — which is NOT a failure, it is the pre-registered starting state. See
    ``agi-proof/third-party-heldout/intake_measurement_spec.json``.

Usage:
    python3 tools/third_party_intake.py --manifest eval/third_party_intake/sample_manifest.jsonl
    python3 tools/third_party_intake.py --manifest <path> --jaccard 0.8 --shingle 5

Exit code: 0 always (intake is a reporting scaffold, not a GO gate — the GO gate lives in the
measurement spec and fires only when a real corpus exists). Exit 2 if the manifest is
unreadable/missing. A JSON receipt is printed to stdout; human prose to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str((ROOT / "tools").resolve()))

# Reuse the EXACT decontam primitives the CI assertion uses, so intake and CI agree.
from assert_decontam import _jaccard, _shingles, normalize  # noqa: E402
from provenance_bench.dataset_guard import (  # noqa: E402
    _load_jsonl, eval_prompt_set, prompt_of)

# A machine-checkable oracle is mandatory for the headline intake. LLM-judged items are a
# separate, labelled family and are never admitted here.
_MACHINE_ORACLES = {
    "sympy": lambda sc: bool(str(sc.get("gold", "")).strip()),
    "exec": lambda sc: bool(str(sc.get("test", "")).strip()),
}


def eval_baseline(*, exclude: Path | None = None) -> set[str]:
    """Normalized eval prompt surface for the decontam check, EXCLUDING one file.

    ``eval_prompt_set`` globs ``eval/**/*.jsonl``; a manifest placed under ``eval/`` would be
    self-ingested and every item would spuriously "match itself". We rebuild the baseline from
    ``eval_prompt_set`` and subtract only the prompts that come *solely* from the excluded file
    — a prompt that also appears in another eval file (a true collision) stays in the baseline,
    so real decontam violations are still caught.
    """
    full = set(eval_prompt_set(root=ROOT))
    if exclude is None or not exclude.exists():
        return full
    excl_norm: set[str] = set()
    for row in _load_jsonl(exclude):
        pr = prompt_of(row)
        if pr:
            excl_norm.add(normalize(pr))
    # Which of the excluded prompts also appear in ANOTHER eval file? Those are true
    # collisions and must remain in the baseline.
    exclude_res = exclude.resolve()
    also_elsewhere: set[str] = set()
    for p in sorted((ROOT / "eval").rglob("*.jsonl")):
        if p.resolve() == exclude_res:
            continue
        for row in _load_jsonl(p):
            pr = prompt_of(row)
            if pr:
                n = normalize(pr)
                if n in excl_norm:
                    also_elsewhere.add(n)
    return (full - excl_norm) | also_elsewhere


def _read_manifest(path: Path) -> list[dict]:
    """Load a JSONL manifest. Raises on unreadable/missing (caller maps to exit 2)."""
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest line {lineno}: bad JSON — {exc}") from exc
    return items


def _validity_reason(item: dict) -> str | None:
    """Return None if the item is structurally eligible, else a human reason string."""
    if not str(item.get("prompt", "")).strip():
        return "empty-prompt"
    if not str(item.get("itemId", "")).strip():
        return "missing-itemId"
    if not item.get("decontamProof"):
        return "missing-decontamProof"
    scoring = item.get("scoring") or {}
    kind = scoring.get("kind")
    checker = _MACHINE_ORACLES.get(kind)
    if checker is None:
        return f"non-machine-checkable-oracle:{kind!r}"
    if not checker(scoring):
        return f"incomplete-{kind}-oracle"
    return None


def _decontam_reason(prompt: str, eval_norm: set[str],
                     eval_shingles: list[tuple[str, set]], *,
                     jaccard: float, shingle: int) -> str | None:
    """Return None if the prompt is decontam-clean, else a human reason string.

    Same two layers as tools/assert_decontam.py: exact/normalized overlap, then a content
    k-shingle Jaccard near-duplicate scan against the committed eval prompt surface.
    """
    npr = normalize(prompt)
    if npr in eval_norm:
        return "exact-eval-overlap"
    tsh = _shingles(prompt, shingle)
    if not tsh:
        return None
    for enorm, esh in eval_shingles:
        if npr == enorm:  # exact handled above; skip self
            continue
        if _jaccard(tsh, esh) >= jaccard:
            return f"near-dup(J>={jaccard})"
    return None


def run_intake(items: list[dict], *, jaccard: float = 0.9, shingle: int = 5,
               eval_norm: set[str] | None = None) -> dict:
    """Core intake logic (importable + testable, no I/O).

    Returns a receipt dict. Enforces the two-counter rule:
      * ``admittedCount`` counts validity+decontam survivors.
      * ``loopClosedCount`` counts verifier-admitted-AND-held-out-gain items. With no real
        verifier/gain signal in-repo it is ALWAYS 0 — never derived from admittedCount.

    The caller is responsible for the eval baseline (``eval_norm``). Use ``eval_baseline(
    exclude=<manifest path>)`` so a manifest that lives under ``eval/`` is not self-ingested
    (see that helper). If ``eval_norm`` is None the full ``eval_prompt_set`` is used verbatim.
    """
    if eval_norm is None:
        eval_norm = set(eval_prompt_set(root=ROOT))
    eval_shingles = [(e, _shingles(e, shingle)) for e in eval_norm]

    admitted: list[str] = []
    rejected: list[dict] = []
    loop_closed: list[str] = []  # stays empty: no verifier+gain data committed.

    for item in items:
        item_id = str(item.get("itemId", "<no-id>"))
        vreason = _validity_reason(item)
        if vreason is not None:
            rejected.append({"itemId": item_id, "stage": "validity", "reason": vreason})
            continue
        dreason = _decontam_reason(str(item["prompt"]), eval_norm, eval_shingles,
                                   jaccard=jaccard, shingle=shingle)
        if dreason is not None:
            rejected.append({"itemId": item_id, "stage": "decontam", "reason": dreason})
            continue
        # Admitted = passed validity + decontam. This is intake capacity, NOT a closed loop.
        admitted.append(item_id)
        # loopClosed is DELIBERATELY NOT incremented here. Closing the loop requires a
        # verifier admission AND a measured held-out gain, which needs a real corpus + a
        # powered run. Neither exists in-repo, so loop_closed stays empty (honest N=0).

    receipt = {
        "tool": "third_party_intake",
        "status": "preregistration_only",
        "canClaimAGI": False,
        "go": False,
        "decontamGate": {"jaccard": jaccard, "shingle": shingle,
                         "primitivesFrom": "tools/assert_decontam.py"},
        "counters": {
            "seenCount": len(items),
            "admittedCount": len(admitted),          # validity + decontam survivors
            "rejectedCount": len(rejected),
            "loopClosedCount": len(loop_closed),     # verifier-admitted AND held-out gain
        },
        "twoCounterRule": (
            "admittedCount != loopClosedCount by construction. Admitting an item measures "
            "intake capacity only. loopClosedCount requires verifier admission AND a measured "
            "held-out gain; with no external corpus + no in-session network it is 0 (honest, "
            "not a failure)."
        ),
        "admitted": admitted,
        "rejected": rejected,
        "honestBound": (
            "loopClosedCount=0 is the PRE-REGISTERED starting state, not a NEGATIVE result. "
            "GO requires real N>=1: an externally-authored verifiable domain that passes "
            "assert_decontam AND on which the verifier loop closes with a measured held-out gain."
        ),
    }
    # Invariant guard: the two counters must never be conflated, and loopClosed can never
    # exceed admitted. Fail-closed if the scaffold is ever wired to violate this.
    assert receipt["counters"]["loopClosedCount"] <= receipt["counters"]["admittedCount"], \
        "invariant violated: loopClosedCount > admittedCount"
    assert receipt["counters"]["loopClosedCount"] == 0, \
        "loopClosedCount must be 0 until a real verifier+gain signal is committed"
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="path to an externally-authored verifiable-task manifest (JSONL)")
    ap.add_argument("--jaccard", type=float, default=0.9,
                    help="near-duplicate Jaccard threshold (matches assert_decontam)")
    ap.add_argument("--shingle", type=int, default=5, help="word k-shingle size")
    args = ap.parse_args()

    path = (ROOT / args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
    if not path.exists():
        print(json.dumps({"tool": "third_party_intake", "error": "manifest-not-found",
                          "path": str(path)}))
        print(f"manifest not found: {path}", file=sys.stderr)
        return 2
    try:
        items = _read_manifest(path)
    except (OSError, ValueError) as exc:
        print(json.dumps({"tool": "third_party_intake", "error": "manifest-unreadable",
                          "detail": str(exc)}))
        print(f"manifest unreadable: {exc}", file=sys.stderr)
        return 2

    baseline = eval_baseline(exclude=path)
    receipt = run_intake(items, jaccard=args.jaccard, shingle=args.shingle, eval_norm=baseline)
    receipt["manifest"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    receipt["decontamGate"]["baselineEvalPrompts"] = len(baseline)

    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    c = receipt["counters"]
    print(f"INTAKE: seen={c['seenCount']} admitted={c['admittedCount']} "
          f"rejected={c['rejectedCount']} loopClosed={c['loopClosedCount']} "
          f"(loopClosed=0 is the honest pre-registered state, not a failure)", file=sys.stderr)
    for r in receipt["rejected"]:
        print(f"  REJECT [{r['stage']}] {r['itemId']}: {r['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
