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
     "objects": [{"label": str, "box": [x, y, w, h],
                  "z": float, "size": float}, ...],   # z/size optional
     "texts":   [{"value": str, "box": [x, y, w, h]}, ...]}

Boxes are ``[x, y, w, h]`` in a top-left origin plane (y grows downward), so
``above`` means *smaller* center-y. The verifiers assume at most one object per
referenced label in a trap scene (the data is authored that way); ``count`` is
the exception and counts all matching objects.

**Physical / 2.5D dimension (optional fields).** An object may also carry a scalar
``z`` (camera-frame depth; *larger = farther from the camera*) and a scalar
``size`` (real-world size proxy in consistent units, decoupled from apparent box
size — a near small thing can have a *bigger* box than a far large thing). These
give judge-free ground truth for the physical-world axes VLMs are weakest on —
depth ordering, occlusion, real-vs-apparent size, and 3D separation — mirroring
the field's finding that VLMs are semantically strong but metrically blind. Like
``relation``, every physical verifier returns ``False`` (never raises) when a
referenced object is missing or lacks the field it needs, so an ungroundable
physical question fails closed rather than guessing.
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


# --- physical / 2.5D verifiers (depth · occlusion · real size · distance) -- #
# The metric-grounding family: judge-free ground truth over the optional ``z``
# (depth; larger == farther) and ``size`` (real-world size proxy) fields. These
# are the visual twin of the math/code RLVR verifiers for the *physical* axes a
# VLM hallucinates on — it sees co-occurrence priors, not geometry. Each returns
# a plain bool/float and fails closed (False / None) on a missing object/field.

_DEPTH_RELATIONS = ("in_front_of", "behind")


def depth(scene: dict, label: str) -> "float | None":
    """The camera-frame depth ``z`` of ``label`` (None if absent or unset)."""
    o = _find(scene, label)
    return None if o is None else o.get("z")


def depth_order(scene: dict, a: str, rel: str, b: str) -> bool:
    """Whether ``a`` is ``in_front_of`` / ``behind`` ``b`` by depth.

    ``in_front_of`` means *smaller* z (nearer the camera). False if either object
    is missing or has no ``z`` — an ungroundable ordering, never a guess.
    """
    if rel not in _DEPTH_RELATIONS:
        raise ValueError(f"unknown depth relation {rel!r}; expected one of {_DEPTH_RELATIONS}")
    za, zb = depth(scene, a), depth(scene, b)
    if za is None or zb is None:
        return False
    return za < zb if rel == "in_front_of" else za > zb


def _boxes_overlap(b1: "list[float]", b2: "list[float]") -> bool:
    """Whether two ``[x, y, w, h]`` boxes share any 2D area (touching edges = no)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = min(x1 + w1, x2 + w2) - max(x1, x2)
    iy = min(y1 + h1, y2 + h2) - max(y1, y2)
    return ix > 0 and iy > 0


def occludes(scene: dict, a: str, b: str) -> bool:
    """Whether ``a`` occludes ``b``: their boxes overlap AND ``a`` is nearer.

    Occlusion needs *both* image-plane overlap and a depth ordering (the nearer
    object hides the farther). Overlap with the wrong depth order, or no overlap,
    is not occlusion. False if either object is missing or has no ``z``.
    """
    oa, ob = _find(scene, a), _find(scene, b)
    if oa is None or ob is None:
        return False
    za, zb = oa.get("z"), ob.get("z")
    if za is None or zb is None:
        return False
    return _boxes_overlap(oa["box"], ob["box"]) and za < zb


def bigger_than(scene: dict, a: str, b: str) -> bool:
    """Whether ``a`` is larger than ``b`` in *real-world* ``size`` (not apparent box).

    The size-illusion check: a near small object can fill more pixels than a far
    large one, so apparent box area lies. False if either lacks a ``size``.
    """
    oa, ob = _find(scene, a), _find(scene, b)
    if oa is None or ob is None:
        return False
    sa, sb = oa.get("size"), ob.get("size")
    if sa is None or sb is None:
        return False
    return sa > sb


def distance_between(scene: dict, a: str, b: str) -> "float | None":
    """3D Euclidean distance between object centers over (center_x, center_y, z).

    ``z`` defaults to 0 when unset (a flat scene). None if either object missing.
    """
    oa, ob = _find(scene, a), _find(scene, b)
    if oa is None or ob is None:
        return None
    (ax, ay), (bx, by) = _center(oa["box"]), _center(ob["box"])
    az, bz = oa.get("z", 0.0), ob.get("z", 0.0)
    return ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5


_DISTANCE_OPS = {
    ">": lambda d, v: d > v, "<": lambda d, v: d < v,
    ">=": lambda d, v: d >= v, "<=": lambda d, v: d <= v,
}


def distance_cmp(scene: dict, a: str, b: str, op: str, value: float) -> bool:
    """Whether ``distance_between(a, b) <op> value`` — a yes/no metric claim.

    False (fail-closed) if the distance is undeterminable (object missing).
    """
    fn = _DISTANCE_OPS.get(op)
    if fn is None:
        raise ValueError(f"unknown distance op {op!r}; expected one of {sorted(_DISTANCE_OPS)}")
    d = distance_between(scene, a, b)
    return False if d is None else fn(d, value)


def point_in_box(box: "list[float]", x: float, y: float) -> bool:
    """Whether pixel ``(x, y)`` falls inside ``[x, y, w, h]``."""
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def element_at(scene: dict, x: float, y: float) -> "str | None":
    """Label of the topmost object whose box contains ``(x, y)`` (None if none)."""
    for o in reversed(scene.get("objects", [])):  # last drawn == topmost
        if point_in_box(o["box"], x, y):
            return o.get("label")
    return None


def point_hits_label(scene: dict, label: str, x: float, y: float) -> bool:
    """Whether ``(x, y)`` lands on an element carrying ``label``."""
    return element_at(scene, x, y) == label


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


# --- chart / table / document verifiers (synthesis-grounded ground truth) -- #
# These read structured values out of a rendered chart/table/document so a
# synthetic sample's label is machine-derivable, never hand-asserted — the
# roadmap's "verifier-checked synthesis" (workstream D).


def chart_value(scene: dict, label: str) -> "float | None":
    """The numeric value of the bar/slice carrying ``label`` (None if absent)."""
    for bar in (scene.get("chart") or {}).get("bars", []):
        if bar.get("label") == label:
            return bar.get("value")
    return None


def chart_extreme(scene: dict, which: str) -> "str | None":
    """Label of the largest ('max') or smallest ('min') bar."""
    bars = (scene.get("chart") or {}).get("bars", [])
    if not bars:
        return None
    if which == "max":
        return max(bars, key=lambda b: b["value"])["label"]
    if which == "min":
        return min(bars, key=lambda b: b["value"])["label"]
    raise ValueError(f"chart_extreme expects 'max'/'min', got {which!r}")


def table_cell(scene: dict, row_key: str, col: str) -> "str | None":
    """Cell value at the row whose first column == ``row_key``, column ``col``."""
    table = scene.get("table") or {}
    cols = table.get("columns", [])
    if col not in cols:
        return None
    ci = cols.index(col)
    for row in table.get("rows", []):
        if row and str(row[0]) == str(row_key):
            return row[ci] if ci < len(row) else None
    return None


def doc_field(scene: dict, name: str) -> "str | None":
    """Value of a key-value field in a structured document scene."""
    return (scene.get("document") or {}).get("fields", {}).get(name)


# --- resolve a trap's declared check (data self-validation) ---------------- #

_CHECK_FNS = {
    "presence": lambda s, c: present(s, c["label"]),
    "count": lambda s, c: count(s, c["label"]),
    "relation": lambda s, c: relation(s, c["a"], c["rel"], c["b"]),
    "ocr": lambda s, c: ocr_contains(s, c["text"]),
    "chart_value": lambda s, c: chart_value(s, c["label"]),
    "chart_extreme": lambda s, c: chart_extreme(s, c["which"]),
    "table_cell": lambda s, c: table_cell(s, c["row"], c["col"]),
    "doc_field": lambda s, c: doc_field(s, c["name"]),
    # physical / 2.5D checks (all resolve to bool -> yesno traps with `expect`)
    "depth_order": lambda s, c: depth_order(s, c["a"], c["rel"], c["b"]),
    "occlusion": lambda s, c: occludes(s, c["a"], c["b"]),
    "bigger": lambda s, c: bigger_than(s, c["a"], c["b"]),
    "distance_cmp": lambda s, c: distance_cmp(s, c["a"], c["b"], c["op"], c["value"]),
}


def resolve_check(scene: dict, check: dict):
    """Run a trap's ``check`` against its scene and return the verifier's value.

    bool for presence/relation/ocr; int/float for count/chart_value; str for
    chart_extreme/table_cell/doc_field. Tests assert this equals the trap's
    declared ``gold_answer`` — so labels are machine-derived, never hand-asserted.
    """
    fn = _CHECK_FNS.get(check.get("type"))
    if fn is None:
        raise ValueError(f"unknown check type {check.get('type')!r}")
    return fn(scene, check)


def gold_matches_check(trap: dict) -> bool:
    """Whether a trap's human-readable ``gold_answer`` agrees with its verifier.

    The dataset integrity invariant: the label a reader sees and the label the
    judge-free verifier computes must be the same, for every row — across the
    base yesno/count/text traps AND the synthesised chart/table/document traps.
    """
    scene, check = trap["scene"], trap["check"]
    resolved = resolve_check(scene, check)
    atype = trap["answer_type"]
    gold = str(trap["gold_answer"]).strip().lower()
    if atype == "count":
        return resolved is not None and int(resolved) == int(trap["gold_answer"])
    if atype == "yesno":
        # presence/relation: verifier bool must match the declared expect, and the
        # gold must read 'yes' iff expect is True.
        expect = bool(check["expect"])
        return bool(resolved) == expect and gold == ("yes" if expect else "no")
    if atype == "text":
        if isinstance(resolved, bool):
            # ocr: the declared gold text must actually be what the verifier reads.
            return resolved is True and bool(check.get("expect", True)) is True
        # chart_extreme / table_cell / doc_field: string equality (normalised).
        return resolved is not None and _norm_text(str(resolved)) == _norm_text(str(gold))
    raise ValueError(f"unknown answer_type {atype!r}")
