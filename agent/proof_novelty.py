# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Semantic novelty assessor for Lean proofs — the explicit guard for THEORY-ISSUES #4.

WHY THIS EXISTS (the honest framing):

The live Lean run verified 8/10 LLM-proposed proofs, but they were RECALL of known
library lemmas, not NOVELTY: cross-proposer char-trigram Jaccard hit 0.948 on `map_map`.
THEORY-ISSUES issue 4 names the gap precisely — the Lean kernel guarantees CORRECTNESS,
not NOVELTY, and the local-Jaccard probe in ``agent.lean_backend.novelty_check`` only
measures SURFACE overlap against a LOCAL corpus. It cannot see the model's training data,
so it cannot detect that a "verified, locally-novel" proof is in fact a memorized
reproduction of a standard named lemma. A recalled proof could therefore FALSELY close the
``verifier_synthesis_over_proof_kernel`` bet.

This module adds a SEMANTIC layer on top of surface Jaccard. It combines three cheap,
honest signals:

  * ``surface_novelty``   — the existing char-trigram Jaccard (max overlap vs a LOCAL
                            corpus). Reused verbatim from the repo convention. It only
                            measures LOCAL surface overlap; it says nothing about training
                            data.
  * ``structural_signal`` — cheap proof-structure features (uses induction? only
                            rfl/simp? cites a named lemma? proof length) as a heuristic
                            for "did real work happen, or is this a one-liner recall?".
  * ``semantic_novelty``  — an INJECTED LLM ``judge_fn`` rates whether the THEOREM is a
                            standard/named textbook lemma vs non-standard, and whether the
                            PROOF reads as recall vs constructed. The verdict is combined
                            honestly.

HONEST SCOPE (non-negotiable, the discipline this repo enforces):
  NONE of these signals can PROVE a proof is absent from the model's training data. There
  is no oracle for "this was not memorized". Surface Jaccard is local; structural features
  are heuristics; an LLM judge is itself a fallible model that may have seen the same
  corpus. Together they REDUCE — they do NOT ELIMINATE — the memorization false-positive
  risk. This module makes the issue-4 guard explicit and measurable; it does not, and
  cannot, certify genuine novelty. ``canClaimAGI`` / ``level3Evidence`` stay false.

Fail-closed throughout: an unusable judge (raises / empty / unparseable) -> the verdict
defaults to ``likely_recall=True`` (assume the worst — a recalled proof is the failure
mode we must NOT miss). ``surface_novel`` likewise defaults conservatively.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from agent.lean_backend import _trigram_best_overlap

# Surface-novelty threshold mirrors ``agent.lean_backend.novelty_check`` (0.92): a best
# char-trigram Jaccard overlap >= this counts as a LOCAL near-duplicate (surface-recalled).
SURFACE_DUP_THRESHOLD = 0.92

__all__ = [
    "surface_novelty",
    "structural_signal",
    "semantic_novelty",
    "make_judge_fn",
    "SURFACE_DUP_THRESHOLD",
]


def surface_novelty(proof: str, corpus: list[str]) -> float:
    """Max char-trigram Jaccard overlap of ``proof`` against the best ``corpus`` match.

    Reuses the repo convention (``agent.lean_backend._trigram_best_overlap``): the cheap,
    dependency-free surface signal. Returns a float in [0, 1]; 0.0 when either side has no
    trigrams (fail-safe, not a verdict).

    IMPORTANT — what this does NOT measure: this is LOCAL SURFACE overlap against the
    supplied ``corpus`` only. It says nothing about the model's training data, so a low
    value does NOT imply the proof is genuinely novel (it may be a memorized standard
    lemma absent from this local corpus). That blind spot is exactly why
    ``semantic_novelty`` exists.
    """
    return _trigram_best_overlap(proof, corpus or [])


# Tactic heads that, used ALONE, indicate a one-shot/closing proof (low "work done").
# A proof that is only these is more likely a recalled one-liner than constructed reasoning.
_CLOSER_TACS = frozenset({"rfl", "simp", "trivial", "decide", "exact", "assumption"})
# Heads that indicate genuine case/recursion work happened.
_INDUCTION_TACS = frozenset({"induction", "cases", "rcases", "obtain", "match"})
# A dotted identifier (e.g. ``Nat.add_comm``, ``List.map_map``) is a cited NAMED lemma —
# a strong recall signal: the proof leans on a specific library result by name.
_NAMED_LEMMA_RE = re.compile(r"\b[A-Z][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)+\b")
_TAC_HEAD_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_']*)")
_TAC_SPLIT_RE = re.compile(r"<;>|[;\n]")


def structural_signal(proof: str) -> dict[str, Any]:
    """Cheap proof-structure features — a HEURISTIC for "did real work happen?".

    Pure stdlib (``re``), no Lean, no model. Splits the proof string into tactic heads and
    reports structural cues that distinguish a constructed multi-step proof from a recalled
    one-liner:

      * ``uses_induction``   — True if any tactic is induction/cases/rcases/obtain/match.
      * ``only_closers``     — True if EVERY tactic head is a one-shot closer (rfl/simp/
        trivial/decide/exact/assumption). A pure closer proof is a weak novelty signal:
        more likely a recalled/automated one-liner than constructed reasoning.
      * ``cites_named_lemma``— True if the proof references a dotted library name
        (``Nat.add_comm``). Citing a named lemma is a RECALL signal.
      * ``named_lemmas``     — the distinct dotted names cited (audit detail).
      * ``proof_len``        — character length (gross "amount written").
      * ``n_tactics``        — number of tactic statements parsed.
      * ``work_score``       — a [0,1] crude composite (higher = more apparent work):
        rewards induction + multiple tactics, penalizes pure-closer + named-lemma recall.

    HONEST CAVEAT: these are surface heuristics on the proof STRING. A short ``simp`` proof
    can be genuinely novel and a long proof can be memorized; ``work_score`` is a nudge, not
    a verdict. It feeds ``semantic_novelty`` as one input among several.
    """
    text = proof or ""
    heads: list[str] = []
    for chunk in _TAC_SPLIT_RE.split(text):
        m = _TAC_HEAD_RE.match(chunk.strip())
        if m:
            heads.append(m.group(1))
    named = sorted(set(_NAMED_LEMMA_RE.findall(text)))
    uses_induction = any(h in _INDUCTION_TACS for h in heads)
    only_closers = bool(heads) and all(h in _CLOSER_TACS for h in heads)
    cites_named = bool(named)

    # Crude composite in [0,1]. Start neutral; reward structure, penalize recall cues.
    score = 0.5
    if uses_induction:
        score += 0.3
    if len(heads) >= 3:
        score += 0.1
    if only_closers:
        score -= 0.3
    if cites_named:
        score -= 0.2
    work_score = max(0.0, min(1.0, score))

    return {
        "uses_induction": uses_induction,
        "only_closers": only_closers,
        "cites_named_lemma": cites_named,
        "named_lemmas": named,
        "proof_len": len(text),
        "n_tactics": len(heads),
        "work_score": round(work_score, 3),
    }


def _judge_prompt(theorem: str, proof: str) -> str:
    """The prompt handed to the injected ``judge_fn``.

    Asks two yes/no questions on separate lines so the reply is cheaply parseable:
      1. Is the THEOREM a standard/named textbook or library lemma?
      2. Does the PROOF look like recall of a memorized proof (vs newly constructed)?
    """
    return (
        "You are assessing whether a Lean theorem and its proof are likely MEMORIZED from "
        "training data rather than genuinely reasoned.\n\n"
        f"THEOREM:\n{theorem}\n\nPROOF:\n{proof}\n\n"
        "Answer with exactly two lines, each 'yes' or 'no':\n"
        "STANDARD_LEMMA: <yes if the theorem is a standard/named textbook or library "
        "lemma, no if it is non-standard/bespoke>\n"
        "PROOF_RECALL: <yes if the proof reads like recall of a memorized proof, no if it "
        "looks freshly constructed>"
    )


_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def _parse_judge_line(reply: str, key: str) -> bool | None:
    """Extract the yes/no for ``key`` (e.g. ``STANDARD_LEMMA``) from a judge ``reply``.

    Returns True (yes), False (no), or None if the line is missing/unparseable. None is a
    FAIL-CLOSED signal the caller treats as "assume the worst" (recall)."""
    for line in (reply or "").splitlines():
        if key.lower() in line.lower():
            tail = line.split(":", 1)[-1] if ":" in line else line
            if _YES_RE.search(tail):
                return True
            if _NO_RE.search(tail):
                return False
    return None


def semantic_novelty(
    theorem: str,
    proof: str,
    judge_fn: Callable[[str], str],
    *,
    corpus: list[str] | None = None,
) -> dict[str, Any]:
    """SEMANTIC novelty verdict combining surface, structural, and LLM-judge signals.

    ``judge_fn`` is INJECTED: a fake in tests (deterministic, no network), an LLM in prod
    (see :func:`make_judge_fn`). It receives :func:`_judge_prompt` and returns a string the
    parser reads for two yes/no answers — is the THEOREM a standard named lemma, and does
    the PROOF read as recall?

    Combined honestly into ``{"surface_novel", "likely_recall", "notes", ...}``:
      * ``surface_novel``  — True iff the best LOCAL char-trigram overlap < threshold (the
        proof is not a surface near-duplicate of the supplied corpus). False on a
        near-duplicate. LOCAL only — see :func:`surface_novelty`.
      * ``likely_recall``  — True iff the evidence points to a MEMORIZED proof: the judge
        calls the theorem a standard lemma OR the judge calls the proof recall OR the
        structural signal is a pure-closer/named-lemma one-liner. Fail-closed: if the judge
        is unusable (raises / empty / unparseable), ``likely_recall`` defaults to True
        (assume the worst — a recalled proof is the failure mode issue 4 says we must NOT
        miss).

    HONEST CAVEAT (the issue-4 guard, restated): this REDUCES, it does not ELIMINATE, the
    memorization false-positive risk. It CANNOT prove the proof is absent from training
    data — surface overlap is local, structural cues are heuristics, and the judge is a
    fallible model that may share the same corpus. Treat ``likely_recall=False`` as
    "no memorization evidence found", NOT as "certified novel".
    """
    corpus = corpus or []
    overlap = surface_novelty(proof, corpus)
    surface_novel = overlap < SURFACE_DUP_THRESHOLD
    struct = structural_signal(proof)

    notes: list[str] = []
    # Fail-closed judge handling: any failure -> treat as a recall signal.
    judge_ok = True
    reply = ""
    try:
        reply = judge_fn(_judge_prompt(theorem, proof)) or ""
    except Exception as exc:  # fail-closed: an unusable judge never clears the proof
        judge_ok = False
        notes.append(f"judge_error:{type(exc).__name__} -> assume recall (fail-closed)")
    if judge_ok and not reply.strip():
        judge_ok = False
        notes.append("judge_empty -> assume recall (fail-closed)")

    theorem_standard = _parse_judge_line(reply, "STANDARD_LEMMA") if judge_ok else None
    proof_recall = _parse_judge_line(reply, "PROOF_RECALL") if judge_ok else None
    if judge_ok and theorem_standard is None and proof_recall is None:
        judge_ok = False
        notes.append("judge_unparseable -> assume recall (fail-closed)")

    if not surface_novel:
        notes.append(
            f"surface near-duplicate (overlap {overlap:.3f} >= {SURFACE_DUP_THRESHOLD})"
        )
    if theorem_standard:
        notes.append("judge: theorem is a STANDARD/named lemma")
    if proof_recall:
        notes.append("judge: proof reads as RECALL")
    if struct["only_closers"]:
        notes.append("structural: proof is only one-shot closers (weak novelty signal)")
    if struct["cites_named_lemma"]:
        notes.append(f"structural: cites named lemma(s) {struct['named_lemmas']}")

    # likely_recall is True if ANY recall signal fires, OR the judge was unusable
    # (fail-closed: an unusable judge must not clear a proof).
    likely_recall = (
        (not judge_ok)
        or bool(theorem_standard)
        or bool(proof_recall)
        or struct["only_closers"]
    )
    if not likely_recall:
        notes.append(
            "no memorization evidence found (NOT a certification of novelty; this probe "
            "cannot see training data)"
        )

    return {
        "surface_novel": surface_novel,
        "likely_recall": likely_recall,
        "best_overlap": round(overlap, 4),
        "judge_ok": judge_ok,
        "theorem_is_standard": theorem_standard,
        "proof_is_recall": proof_recall,
        "structural": struct,
        "notes": notes,
        "method": (
            "semantic: surface char-trigram Jaccard (LOCAL) + structural heuristics + "
            "injected LLM judge. Reduces — does NOT eliminate — memorization false positives; "
            "cannot prove absence from training data."
        ),
    }


def make_judge_fn(spec: str, *, max_tokens: int = 16) -> Callable[[str], str]:
    """Build a real ``(prompt) -> str`` judge that calls ``agent.model.complete`` with ``spec``.

    Lazy-imports ``agent.model`` so the deterministic tests (which inject fake judges) never
    touch network code — mirrors ``agent.llm_debunk_detector.make_llm_judge_fn``.

    Args:
        spec: a model spec understood by ``agent.model.complete``.
        max_tokens: cap on the judge reply (two short lines; small by default).

    Returns:
        A judge callable. On any model error it returns ``""``, which
        :func:`semantic_novelty` treats fail-closed as ``likely_recall=True``.
    """
    def judge(prompt: str) -> str:
        from agent.model import complete  # noqa: PLC0415 — lazy; never imported in tests

        try:
            return complete(
                "You are a strict, terse classifier for proof-memorization risk. Reply with "
                "exactly the two requested lines, each 'yes' or 'no'.",
                prompt,
                spec=spec,
                max_tokens=max_tokens,
            )
        except Exception:
            return ""  # fail closed -> semantic_novelty treats as likely_recall=True

    return judge
