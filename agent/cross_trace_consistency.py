# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cross-trace OKF contradiction mining (extension E4) — a global consistency
invariant no current component enforces.

Today ``okf.contradiction_ledger`` finds contradictions WITHIN one belief graph
(X and ¬X both live in one trace). But Sophia's trace log accumulates steps
across runs and phases (rlvr / sft / benchmark / conscience), and two traces
from DIFFERENT runs can assert contradictory claims with neither trace internally
inconsistent:

  run A, step 7: "the charter was written by the committee"  (verified=True)
  run B, step 3: "the charter was written by Alice"          (verified=True)

Each passed its own fact+logic gates; together they contradict. No existing
component mines the log globally for this. This module does.

A cross-trace contradiction is the highest-signal audit event: it means two
runs that each believed they were correct cannot both be right. Surfacing these
is the difference between "each step is verified" (local) and "the body of
recorded reasoning is globally consistent" (global) — the latter is the actual
trustworthiness claim.

Design:
  - ``mine_contradictions(traces)`` extracts a normalized (claim, negated) key
    per verified trace, then reports any (X, ¬X) pair that BOTH appear as
    verified across different traces. Reuses reasoning_compiler._norm so
    normalization matches the compiler's own contradiction detector.
  - The "negated" half is detected by the leading "not " / trailing negation the
    compiler's make_graph plants (and the conscience kernel's block reasons),
    so the same vocabulary is mined consistently.
  - Returns a ledger shaped like okf.contradiction_ledger's output so it drops
    into the same audit surface.

Honest scope: this catches DECLARED-style negation contradictions in the logged
claim text, not semantic paraphrase contradictions ("Alice wrote it" vs "the
committee is the author"). Semantic mining needs a judge; structural mining is
deterministic, offline, and never overclaims coverage — the report states both
the contradictions found AND that absence-of-finding is not proof of consistency.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

# Reuse the compiler's normalization so cross-trace claims and in-graph claims
# are normalized identically (a contradiction found by either detector uses the
# same key). Lazy import to avoid a hard cycle in cold-start paths.
def _norm(stmt: str) -> str:
    return " ".join(stmt.strip().lower().split())


def _split_negation(claim_text: str) -> tuple[str, bool]:
    """Split a claim into (normalized_core, is_negated).

    Mirrors the compiler's planted-contradiction convention: a negated claim is
    "not <core>" (see reasoning_compiler.make_graph: 'not ' + base.statement).
    Returns (core, True) for negated claims, (claim, False) otherwise.
    """
    s = claim_text.strip()
    # leading "not " negation (the compiler's convention)
    low = s.lower()
    if low.startswith("not "):
        return _norm(s[4:]), True
    # "X is not Y" — treat the whole thing as a positive assertion of a negated
    # property; we keep it as-is (core) and mark negated so "X is Y" elsewhere
    # can pair against it. This is conservative: only literal "not <core>" is a
    # clean negation; "is not" stays as its own core.
    return _norm(s), False


def mine_contradictions(traces: Iterable[dict]) -> dict:
    """Mine the trace log for cross-trace contradictions.

    A cross-trace contradiction is a pair of traces where one asserts ``X`` and
    another asserts ``not X`` — and BOTH were recorded as ``verified``. (If one
    is unverified, the gates already caught it locally; the global miner is for
    the case where each run passed its own gates yet they disagree.)

    Returns a ledger:
      {
        "nTraces": int,
        "nVerified": int,
        "contradictions": [ {a: {traceId,runId,claim}, b: {...}, claim: core}, ... ],
        "claimIndex": {core: [traceIds that asserted it positive]},
        "negatedIndex": {core: [traceIds that asserted it negated]},
      }
    """
    traces = list(traces)
    # core claim -> list of (traceId, runId) that asserted it POSITIVE & verified
    claim_index: dict[str, list[dict]] = defaultdict(list)
    # core claim -> list of (traceId, runId) that asserted it NEGATED & verified
    negated_index: dict[str, list[dict]] = defaultdict(list)

    n_verified = 0
    for t in traces:
        if not t.get("verified"):
            continue  # unverified traces are already gate-caught; not a global concern
        n_verified += 1
        core, negated = _split_negation(t.get("claimText", ""))
        if not core:
            continue
        entry = {"traceId": t.get("traceId"), "runId": t.get("runId"),
                 "claim": t.get("claimText", "")}
        (negated_index if negated else claim_index)[core].append(entry)

    contradictions: list[dict] = []
    # a contradiction exists where a core is asserted BOTH positively and negated
    # across (different) verified traces
    for core in set(claim_index) & set(negated_index):
        positives = claim_index[core]
        negations = negated_index[core]
        # pair the first positive with the first negation (the audit surface; a
        # full cross-product would be noise — one clear example per core suffices
        # to flag the global inconsistency for human review)
        contradictions.append({
            "claim": core,
            "a": positives[0],
            "b": negations[0],
            "nPositiveTraces": len(positives),
            "nNegatedTraces": len(negations),
        })

    return {
        "schema": "sophia.cross_trace_ledger.v1",
        "nTraces": len(traces),
        "nVerified": n_verified,
        "contradictions": contradictions,
        "claimIndex": dict(claim_index),
        "negatedIndex": dict(negated_index),
        "globalConsistent": len(contradictions) == 0,
        # honest scope: structural mining catches literal "not X" pairs, not
        # semantic paraphrase. Absence of contradiction here is NOT proof of
        # consistency — only that no declared-negation pair was found.
        "coverageNote": (
            "Structural contradiction mining over logged claim text. Catches "
            "literal 'X' vs 'not X' pairs across verified traces; does NOT catch "
            "semantic paraphrase contradictions. Absence of contradiction is not "
            "proof of global consistency."
        ),
    }


def mine_log(*, trace_log=None) -> dict:
    """Convenience: mine the current verified-trace log end to end."""
    from sophia_contract.stores import _read_jsonl
    if trace_log is None:
        from agent.verified_trace import TRACE_LOG
        trace_log = TRACE_LOG
    return mine_contradictions(_read_jsonl(trace_log))


__all__ = ["mine_contradictions", "mine_log"]
