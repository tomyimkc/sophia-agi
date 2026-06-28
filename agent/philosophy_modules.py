# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Philosophy as scoped, source-grounded, tested, retractable modules.

The report's hard finding: "ingest text -> selfextend synthesizes a philosophical
verifier" is fiction (the synthesis engines only fit substring/numeric stumps).
Philosophy works ONLY as **human-authored, sound checkers** validated through the
disjoint-split / canary plumbing — never *synthesized* by it.

This module ships the first such checker (Aristotelian assertoric syllogistic, a
finite/decidable system) plus the **formalizability gradient** as per-module
``maxVerdict`` metadata:

  - Aristotelian term-logic / finite deontic  -> may reach machine-checked ``accepted``
  - virtue / care / contractualist ethics     -> max ``candidate`` / ``quarantine``
  - Wittgenstein (family resemblance)          -> ``polythetic`` (no crisp subClassOf)
  - Nagarjuna (catuṣkoṭi)                       -> ``abstain`` / isolated paraconsistent
                                                  side-channel (never feeds the ABox)

See docs/11-Platform/Ontology-Claim-Boundary.md and
moral_corpus/philosophy_modules/.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- the formalizability gradient (per-module max verdict) ------------------- #
MAX_VERDICTS = ("accepted", "candidate", "quarantine", "polythetic", "abstain")


@dataclass(frozen=True)
class PhilosophyModule:
    """Metadata for one philosophy module. ``max_verdict`` caps how strong a
    verdict this module's claims may ever earn (the gradient)."""

    id: str
    tradition: str
    max_verdict: str
    formalizability: str
    checker: str           # "" if human-judgement / no machine checker yet
    notes: str


MODULES: tuple[PhilosophyModule, ...] = (
    PhilosophyModule(
        id="aristotle_term_logic", tradition="aristotelian", max_verdict="accepted",
        formalizability="finite-decidable (assertoric syllogistic; maps to description logic)",
        checker="aristotelian_syllogism_valid",
        notes="24 valid moods over 4 figures; machine-checkable. Start here.",
    ),
    PhilosophyModule(
        id="kant_universal_law", tradition="kantian", max_verdict="candidate",
        formalizability="defeasible deontic (dyadic); maxim formulation is the hard part",
        checker="",
        notes="Formula of Universal Law is encodable, but maxim formulation is not deduction.",
    ),
    PhilosophyModule(
        id="virtue_care_contractualist", tradition="virtue_ethics", max_verdict="quarantine",
        formalizability="defeasible / argumentation (no crisp deductive core)",
        checker="",
        notes="Candidate-only; route through argumentation, never crisp subClassOf.",
    ),
    PhilosophyModule(
        id="wittgenstein_family_resemblance", tradition="wittgensteinian", max_verdict="polythetic",
        formalizability="polythetic cluster membership only (PI §65–71)",
        checker="",
        notes="No necessary-and-sufficient core -> never a crisp subClassOf edge.",
    ),
    PhilosophyModule(
        id="nagarjuna_catuskoti", tradition="madhyamaka", max_verdict="abstain",
        formalizability="paraconsistent / 4-valued; classical ABox would explode",
        checker="",
        notes="Abstain or isolated side-channel; a dialetheia must never enter the classical ABox.",
    ),
)

MODULES_BY_ID = {m.id: m for m in MODULES}


def load_modules() -> tuple[PhilosophyModule, ...]:
    return MODULES


# --- the first SOUND checker: Aristotelian assertoric syllogistic ------------ #
# Premise/conclusion forms: A (All S are P), E (No S are P), I (Some S are P),
# O (Some S are not P). A "mood" is (major, minor, conclusion) forms; a "figure"
# fixes the middle-term position. The 24 traditionally-valid forms (incl. the 5
# subaltern/weakened forms that assume existential import) are finite and total —
# this is a sound decision procedure, not a heuristic.
_VALID_MOODS: dict[int, frozenset[str]] = {
    1: frozenset({"AAA", "EAE", "AII", "EIO", "AAI", "EAO"}),  # Barbara, Celarent, Darii, Ferio, Barbari, Celaront
    2: frozenset({"EAE", "AEE", "EIO", "AOO", "EAO", "AEO"}),  # Cesare, Camestres, Festino, Baroco, Cesaro, Camestrop
    3: frozenset({"AAI", "IAI", "AII", "EAO", "OAO", "EIO"}),  # Darapti, Disamis, Datisi, Felapton, Bocardo, Ferison
    4: frozenset({"AAI", "AEE", "IAI", "EAO", "EIO", "AEO"}),  # Bramantip, Camenes, Dimaris, Fesapo, Fresison, Camenop
}
_FORMS = frozenset("AEIO")


def aristotelian_syllogism_valid(figure: int, mood: str) -> bool:
    """Sound decision: is the (figure, mood) syllogism valid in classical
    assertoric syllogistic (with existential import)?

    ``figure`` ∈ {1,2,3,4}; ``mood`` is a 3-char string over {A,E,I,O}
    (major, minor, conclusion). Returns False for any malformed input
    (fail-closed: an unparseable form is never declared valid).
    """
    if figure not in _VALID_MOODS:
        return False
    mood = (mood or "").upper().strip()
    if len(mood) != 3 or any(c not in _FORMS for c in mood):
        return False
    return mood in _VALID_MOODS[figure]


def check_syllogism_item(item: dict) -> dict:
    """Grade one eval item carrying structured ``figure`` + ``mood`` + gold
    ``valid``. Returns ``{passed, predicted, gold, maxVerdict}``.

    The checker decides over the STRUCTURED form (sound). Parsing natural language
    into a figure/mood is the brittle extractor step and is deliberately NOT done
    here — eval items carry the structured form alongside their prose.
    """
    figure = item.get("figure")
    mood = item.get("mood")
    gold = bool(item.get("valid"))
    predicted = aristotelian_syllogism_valid(figure, mood) if isinstance(figure, int) else False
    return {
        "passed": predicted == gold,
        "predicted": predicted,
        "gold": gold,
        "maxVerdict": MODULES_BY_ID["aristotle_term_logic"].max_verdict,
    }


__all__ = [
    "PhilosophyModule", "MODULES", "MODULES_BY_ID", "load_modules",
    "aristotelian_syllogism_valid", "check_syllogism_item", "MAX_VERDICTS",
]
