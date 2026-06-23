"""OCEAN substrate + one-way MBTI display veneer (Spec A).

Big Five (OCEAN) is the measured substrate. MBTI is a *display veneer*:
`mbti_to_ocean` translates a user-facing type at the request boundary;
`ocean_to_mbti_letters` renders a code at the display boundary. No gate,
verifier, effect-size, or abstention path may read the MBTI string.

Mapping source: McCrae & Costa (1989), J. Personality 57:17-40, second-letter
convention. Neuroticism has NO MBTI correlate (max |r|~0.16) and is always
left unspecified (None). Scoring/public-domain reference: ipip.ori.org.
"""
from __future__ import annotations

from itertools import product

# Verified r-table (1989 point estimates). 'pole_high' = which second-letter
# pole maps to HIGH on the OCEAN factor.
AXIS_OCEAN: dict[str, dict] = {
    "E/I": {"factor": "E", "r": -0.74, "pole_high": "E", "confidence": "highest"},
    "S/N": {"factor": "O", "r": +0.72, "pole_high": "N", "confidence": "highest",
            "note": "1989 point estimate; pooled replications ~0.60-0.65"},
    "T/F": {"factor": "A", "r": +0.44, "pole_high": "F", "confidence": "lowest",
            "note": "weakest, sex-confounded (0.33-0.44)"},
    "J/P": {"factor": "C", "r": -0.485, "pole_high": "J", "confidence": "moderate-high",
            "note": "cited -0.48 to -0.49"},
}

# letter position -> (axis, the two poles in MBTI order)
_AXES = (
    ("E", "I", "E/I"),
    ("S", "N", "S/N"),
    ("T", "F", "T/F"),
    ("J", "P", "J/P"),
)

SIXTEEN_TYPES: tuple[str, ...] = tuple(
    "".join(p) for p in product("EI", "SN", "TF", "JP")
)


def _letter_to_sign(letter: str, axis: str) -> str:
    """Map a single MBTI letter to 'high'/'low' on its OCEAN factor."""
    pole_high = AXIS_OCEAN[axis]["pole_high"]
    return "high" if letter == pole_high else "low"


def mbti_to_ocean(code: str) -> dict:
    """One-way: an MBTI type -> OCEAN target signs. N is always None.

    Returns {"O","C","E","A","N","_meta"} or {"error","available"} on a bad code.
    """
    norm = (code or "").strip().upper()
    if norm not in SIXTEEN_TYPES:
        return {"error": f"unknown MBTI type: {code!r}", "available": list(SIXTEEN_TYPES)}
    signs: dict = {"N": None}
    meta: dict = {}
    for letter, (pos, neg, axis) in zip(norm, _AXES):
        factor = AXIS_OCEAN[axis]["factor"]
        signs[factor] = _letter_to_sign(letter, axis)
        meta[axis] = {"letter": letter, "factor": factor, "r": AXIS_OCEAN[axis]["r"]}
    signs["_meta"] = {"code": norm, "axes": meta,
                      "neuroticism": "undetermined by MBTI (left unspecified)"}
    return signs


def ocean_to_mbti_letters(ocean: dict) -> str:
    """Display-only: OCEAN signs -> a 4-letter code. Neuroticism is ignored."""
    out = []
    for pos, neg, axis in _AXES:
        factor = AXIS_OCEAN[axis]["factor"]
        sign = ocean.get(factor)
        pole_high = AXIS_OCEAN[axis]["pole_high"]
        pole_low = pos if pole_high == neg else neg
        out.append(pole_high if sign == "high" else pole_low)
    return "".join(out)


def build_type_records() -> dict[str, dict]:
    """All 16 type records derived from the verified map (consumed by the MCP
    resource / portable skill). Display copy is minimal and derived, never
    asserting a Neuroticism value."""
    records: dict[str, dict] = {}
    for code in SIXTEEN_TYPES:
        ocean = mbti_to_ocean(code)
        signs = {k: ocean[k] for k in ("O", "C", "E", "A", "N")}
        records[code] = {
            "code": code,
            "ocean": signs,
            "substrate": "Big Five (OCEAN) is the measured substrate; this MBTI "
                         "code is a display veneer. Neuroticism is undetermined.",
            "axes": ocean["_meta"]["axes"],
        }
    return records
