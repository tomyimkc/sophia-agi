"""Cantonese (粵語 / 廣東話) detection and output scaffolding.

The HK access-to-justice niche the legal-industry assessment flags as underserved:
many HK residents read written Cantonese, not Standard Written Chinese. Cantonese
shares the CJK range with SWC, so a plain "has Chinese?" check cannot tell them
apart — what distinguishes written Cantonese is a set of **distinctive characters
and grammatical particles** (嘅, 喺, 唔, 係, 咗, 喇, 嗰, 啲, 乜嘢, 點解 ...) that do
not appear in Standard Written Chinese.

This module is deliberately small and deterministic: it *detects* written
Cantonese and supplies an output instruction. It does not translate or generate —
that is the model's job, prompted via ``cantonese_instruction()``.
"""

from __future__ import annotations

# Characters/particles distinctive to written Cantonese (rare/absent in SWC).
CANTONESE_MARKERS = (
    "嘅", "喺", "唔", "係", "咗", "喇", "嗰", "啲", "乜", "嘢", "佢", "哋",
    "嚟", "畀", "睇", "諗", "瞓", "攞", "嘥", "冚", "嘞", "㗎", "喎", "咩",
    "點解", "邊度", "幾時", "做乜", "唔該", "多謝",
)

# Particles that, combined with markers, strongly indicate Cantonese register.
_STRONG = ("嘅", "喺", "唔係", "係咪", "咗", "㗎", "喎", "嘞", "點解", "邊度", "唔該")


def is_cantonese(text: str) -> bool:
    """True if the text contains distinctive written-Cantonese markers.

    Conservative: requires at least one strong marker, or two distinct general
    markers, so an incidental character (e.g. a name) does not misfire.
    """
    if not text:
        return False
    if any(s in text for s in _STRONG):
        return True
    hits = {m for m in CANTONESE_MARKERS if m in text}
    return len(hits) >= 2


def cantonese_markers_found(text: str) -> list[str]:
    return [m for m in CANTONESE_MARKERS if m and m in (text or "")]


def cantonese_instruction() -> str:
    """Prompt fragment instructing a Cantonese (written 粵語) response section."""
    return (
        "Provide your summary in written Cantonese (粵語 / 廣東話, e.g. using 嘅、喺、"
        "唔、係 where natural), not Standard Written Chinese, so Hong Kong readers can "
        "follow it in their spoken register. Label it 粵語摘要."
    )
