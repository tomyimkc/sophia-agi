# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Datalog port of the ``provenance_faithful`` provenance gate.

This is the symbolic half of the neuro-symbolic split called for by the
Verifiable-Sophia strategic plan (commit cb887e5). The *decision rule* —
"fail-closed when an answer asserts a forbidden attribution in a clause that
is not itself a correction/negation" — is re-expressed as a Datalog program;
the *fact extraction* (which is irreducibly lexical: detecting authorship
assertions and carve-outs in free text) stays in Python and feeds ground facts
to :mod:`agent.datalog_engine`.

The single auditable rule (the whole point of the exercise)::

    violation(Work, Author) :-
        asserted_in_clause(Clause, Work, Author),
        forbidden(Work, Author),
        not carveout(Clause).

Faithfulness contract
---------------------
:func:`check_claim_datalog` MUST return the same ``{passed, reasons,
violations}`` shape as :func:`agent.guarded.check_claim`, and the
byte-identical-verdict experiment (``tools/run_datalog_provenance_audit.py``)
asserts the two agree on every committed provenance case. Any divergence is a
real finding about where the regex gate and the logic port part company.

The ground facts are extracted by reusing the SAME regex patterns and clause
splitter ``provenance_faithful`` uses (imported, not duplicated), so a
divergence cannot come from "we rebuilt the regexes differently" — it can only
come from the rule's logical shape.
"""
from __future__ import annotations

import re
from typing import Any

from agent.datalog_engine import A, Atom, Program, Rule, Var
from agent.guarded import check_claim  # noqa: F401  (re-exported for convenience)


def build_logic_program() -> Program:
    """The pure Datalog kernel: one rule deriving ``violation``.

    Returned unsolved; callers add the ground facts for a given text and call
    ``solve()``. Keeping the rule here (not inline at every call site) means the
    abstention policy is readable and auditable in one place.
    """
    prog = Program()
    # violation(Work, Author) :- asserted_in_clause(C, Work, Author),
    #                            forbidden(Work, Author), not carveout(C).
    prog.rule(
        Rule.clause(
            A("violation", Var("Work"), Var("Author")),
            A("asserted_in_clause", Var("Clause"), Var("Work"), Var("Author")),
            A("forbidden", Var("Work"), Var("Author")),
            A("carveout", Var("Clause")).neg(),
        )
    )
    return prog


def extract_facts(text: str, records: dict | None = None) -> tuple[Program, dict]:
    """Extract the ground facts ``provenance_faithful`` would reason over.

    Reuses the gate's own compiled-pattern builder and clause splitter (imported
    from :mod:`agent.verifiers`) so the lexical side is identical to the Python
    gate — the only thing that differs is how the decision is *derived* from the
    extracted facts (Datalog here, inline ``if/any`` there).

    Returns ``(program_with_facts, meta)`` where ``meta`` carries the per-clause
    and per-record detail needed to render reasons identical to the Python gate.
    """
    from agent.benchmark_checks import DENY_PATTERNS, MYTH_PATTERNS, matches_any
    from agent.verifiers import _carveout_clauses

    # The gate's (rid, author, patterns) specs, recovered via its read-only
    # return_specs hook and cached by records identity. The specs ARE the gate's
    # compiled patterns, returned by reference — so we never rebuild the verifier
    # here (the ~766-pattern compile is the hot path; the verifier object itself
    # is not needed since we match via `specs` directly).
    records = records if records is not None else _default_records()
    specs = _specs_cache(records)

    extra_deny = _gate_extra_deny()

    prog = build_logic_program()

    # forbidden(Work, Author): one fact per (record_id-as-work, forbidden author).
    # The work term is the record id (what the gate keys violations on); the
    # Python gate reports violations as "author -> rid", so we match that exactly.
    forbidden: set[tuple[str, str]] = set()
    for rid, record in records.items():
        for author in record.get("doNotAttributeTo", []) or []:
            forbidden.add((rid, author))
            prog.fact(A("forbidden", rid, author))

    # asserted_in_clause(ClauseId, Work, Author) + carveout(ClauseId):
    # walk sentences -> clauses exactly as the gate does, and for each clause
    # record which forbidden (work, author) pairs its patterns match (assertion)
    # and whether the clause carries a correction/negation marker (carveout).
    clauses_meta: list[dict[str, Any]] = []
    clause_id = 0
    for sentence in re.split(r"[.!?。！？\n]+", text or ""):
        low = sentence.lower()
        if not low.strip():
            continue
        for clause_text in _carveout_clauses(low):
            if not clause_text.strip():
                continue
            cid = f"c{clause_id}"
            clause_id += 1
            is_carveout = (
                matches_any(clause_text, DENY_PATTERNS)
                or matches_any(clause_text, MYTH_PATTERNS)
                or matches_any(clause_text, extra_deny)
            )
            if is_carveout:
                prog.fact(A("carveout", cid))
            for rid, author, patterns in specs:
                if any(p.search(clause_text) for p in patterns):
                    prog.fact(A("asserted_in_clause", cid, rid, author))
                    clauses_meta.append(
                        {"clause": cid, "work": rid, "author": author, "text": clause_text}
                    )
    return prog, {
        "records": records,
        "forbidden": forbidden,
        "clauses": clauses_meta,
        "specs": specs,
    }


def check_claim_datalog(text: str, *, records: dict | None = None) -> dict:
    """Datalog-derived provenance check, same shape as ``check_claim``.

    Returns ``{passed, reasons, violations}`` where ``violations`` is the sorted
    set of ``"author -> rid"`` strings the Python gate emits, and ``passed`` is
    False iff at least one violation was derived. Reasons mirror the gate's
    ``"forbidden attribution asserted: <author> -> <rid>"`` wording.
    """
    prog, _meta = extract_facts(text, records)
    prog.solve()
    # Re-derive the violation set in the gate's "author -> rid" string form.
    violations = sorted({f"{a} -> {w}" for (w, a) in prog.query(A("violation", Var("W"), Var("A")))})
    if violations:
        reasons = [f"forbidden attribution asserted: {v}" for v in violations]
        return {"passed": False, "reasons": reasons, "violations": violations}
    return {"passed": True, "reasons": [], "violations": []}


# --- helpers that read the gate's own internals (no regex reconstruction) ---
_DEFAULT_RECORDS: dict | None = None


def _default_records() -> dict:
    """The frozen default records, cached module-level so its ``id()`` is stable.

    ``_load_provenance_records`` builds a fresh dict every call, which defeated
    the identity-keyed specs cache (each call missed -> ~766 re-compiles -> the
    ~470ms/call profile). The records file set is immutable at runtime, so this
    cache is correct: a different records set always comes through the explicit
    ``records=`` argument and is cached by its own id."""
    global _DEFAULT_RECORDS
    if _DEFAULT_RECORDS is None:
        from agent.verifiers import _load_provenance_records

        _DEFAULT_RECORDS = _load_provenance_records()
    return _DEFAULT_RECORDS


def _gate_specs(records: dict) -> list[tuple[str, str, list[re.Pattern]]]:
    """Recover the (rid, author, patterns) specs ``provenance_faithful`` builds,
    via the gate's OWN read-only ``return_specs`` hook.

    No regex reconstruction, no monkeypatch: the patterns are the production
    gate's compiled patterns, returned by reference. This is the faithfulness
    anchor — a divergence between this port and the Python gate can only come
    from the rule's logical shape, never from "we rebuilt the patterns"."""
    from agent.verifiers import provenance_faithful

    verifier = provenance_faithful(records, return_specs=True)
    return list(verifier.specs)  # type: ignore[attr-defined]


_SPECS_CACHE: dict[int, list[tuple[str, str, list[re.Pattern]]]] = {}


def _specs_cache(records: dict) -> list[tuple[str, str, list[re.Pattern]]]:
    """Memoise ``_gate_specs`` by records identity. The pattern build is the
    hot path (~766 compiles); for a fixed records set it is a pure function."""
    key = id(records)
    cached = _SPECS_CACHE.get(key)
    if cached is None:
        cached = _gate_specs(records)
        _SPECS_CACHE[key] = cached
    return cached


def _gate_extra_deny() -> list[str]:
    """The extra_deny marker list ``provenance_faithful`` applies per-clause.

    Defined as plain strings in the gate (then compiled via ``matches_any`` which
    escapes+compiles on the fly); we replicate the literal list so the carve-out
    predicate matches the gate exactly."""
    return [
        r"do not attribut", r"not\s+\w*\s*attribut", r"never\s+\w*\s*attribut",
        r"misattribut", r"wrongly", r"falsely", r"erroneous", r"並非", r"勿",
        r"summaries say", r'say[s]?\s+["“\']', r"often\s+said", r"commonly\s+(said|believed|attributed)",
        r"\bpopular", r"\bmisread", r"\bmistaken", r"\bloosely\b", r"\bas if\b", r"main speaker",
        r'["“\'](?:wrote|written|authored|composed|penned|attributed)["”\']',
        r"traditionally", r"\bspuri", r"pseudo[\s-]?", r"apocryphal", r"\bforg(?:ed|ery)",
        r"\bdisputed\b", r"\bdoubtful\b", r"\bdebated\b", r"scholars?\s+(?:doubt|reject|dispute|question)",
    ]


# Re-export the Atom type for callers that build their own programs.
__all__ = ["Atom", "build_logic_program", "extract_facts", "check_claim_datalog", "check_claim"]
