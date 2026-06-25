# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Judge-free visual verifiers — deterministic ground truth over a scene spec.

These are the multimodal analog of ``agent/verifiers.py``'s machine-checkable
checks (``math_sound``, ``code_tests_pass``): they share **no code** with the VLM
or the lexical judge, so the hallucination/grounding delta they label is
non-circular. Each verifier resolves a structured ``check`` against the scene's
ground-truth objects/texts and returns a plain bool/int — no model in the loop.

A scene is::

    {"width": int, "height": int,
     "objects": [{"label": str, "box": [x, y, w, h]}, ...],
     "texts":   [{"value": str, "box": [x, y, w, h]}, ...]}

Boxes are ``[x, y, w, h]`` in a top-left origin plane (y grows downward), so
``above`` means *smaller* center-y. The verifiers assume at most one object per
referenced label in a trap scene (the data is authored that way); ``count`` is
the exception and counts all matching objects.
"""

from __future__ import annotations

import re

# --- geometry primitives --------------------------------------------------- #


def _center(box: "list[float]") -> "tuple[float, float]":
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _find(scene: dict, label: str) -> "dict | None":
    for o in scene.get("objects", []):
        if o.get("label") == label:
            return o
    return None


# --- the four verifier families ------------------------------------------- #


def count(scene: dict, label: str) -> int:
    """How many objects carry ``label``."""
    return sum(1 for o in scene.get("objects", []) if o.get("label") == label)


def present(scene: dict, label: str) -> bool:
    """Whether at least one object carries ``label``."""
    return count(scene, label) > 0


_RELATIONS = ("left_of", "right_of", "above", "below")


def relation(scene: dict, a: str, rel: str, b: str) -> bool:
    """Whether ``a`` stands in spatial ``rel`` to ``b`` (False if either missing)."""
    if rel not in _RELATIONS:
        raise ValueError(f"unknown relation {rel!r}; expected one of {_RELATIONS}")
    oa, ob = _find(scene, a), _find(scene, b)
    if oa is None or ob is None:
        return False
    (ax, ay), (bx, by) = _center(oa["box"]), _center(ob["box"])
    if rel == "left_of":
        return ax < bx
    if rel == "right_of":
        return ax > bx
    if rel == "above":
        return ay < by
    return ay > by  # below


def _norm_text(s: str) -> str:
    """Lowercase, strip currency/punctuation/space for tolerant OCR matching."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def ocr_strings(scene: dict) -> "list[str]":
    """All rendered text values in the scene (the OCR ground truth)."""
    return [t.get("value", "") for t in scene.get("texts", [])]


def ocr_contains(scene: dict, text: str) -> bool:
    """Whether ``text`` matches some rendered string (normalised).

    The sentinel ``"none"`` is True exactly when the scene has **no** text — the
    honest answer to a fabricated-OCR-from-blank trap.
    """
    values = ocr_strings(scene)
    if text.strip().lower() == "none":
        return len([v for v in values if v.strip()]) == 0
    target = _norm_text(text)
    return any(_norm_text(v) == target for v in values)


# --- resolve a trap's declared check (data self-validation) ---------------- #


def resolve_check(scene: dict, check: dict):
    """Run a trap's ``check`` against its scene and return the verifier's value.

    Returns a bool for presence/relation/ocr and an int for count. Tests assert
    this equals the trap's declared ``gold_answer`` — so the dataset's labels are
    machine-derived, never hand-asserted.
    """
    kind = check.get("type")
    if kind == "presence":
        return present(scene, check["label"])
    if kind == "count":
        return count(scene, check["label"])
    if kind == "relation":
        return relation(scene, check["a"], check["rel"], check["b"])
    if kind == "ocr":
        return ocr_contains(scene, check["text"])
    raise ValueError(f"unknown check type {kind!r}")


def gold_matches_check(trap: dict) -> bool:
    """Whether a trap's human-readable ``gold_answer`` agrees with its verifier.

    This is the dataset integrity invariant: the label a reader sees and the label
    the judge-free verifier computes must be the same, for every row.
    """
    scene, check = trap["scene"], trap["check"]
    resolved = resolve_check(scene, check)
    atype = trap["answer_type"]
    gold = str(trap["gold_answer"]).strip().lower()
    if atype == "count":
        return int(resolved) == int(trap["gold_answer"])
    if atype == "yesno":
        # presence/relation: expect flag must equal the verifier, and gold must
        # read 'yes' iff expect is True.
        expect = bool(check["expect"])
        return bool(resolved) == expect and gold == ("yes" if expect else "no")
    if atype == "text":
        # ocr: the declared gold text must actually be what the verifier reads.
        return bool(resolved) is True and bool(check.get("expect", True)) is True
    raise ValueError(f"unknown answer_type {atype!r}")
