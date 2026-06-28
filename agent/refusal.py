# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Acceptable-use refusal screen (P3) — input-boundary classifier for the
categories prohibited by ``USAGE-POLICY.md`` (CBRN/weapons uplift, explosives,
CSAM, malware authoring, large-scale fraud/phishing).

This is a HIGH-PRECISION, deterministic pre-screen, not a substitute for the
learned refusal behavior of the model or the conscience classifier. Each rule
requires an *operational intent verb* AND a *harmful target* so that discussing,
defending against, or refusing these topics does not trip the gate. It returns a
recommendation; the conscience kernel / serving layer decides what to do with it.

By design it is OFF in ``conscience_check`` unless the caller opts in
(``context['enforceAcceptableUse']=True``) so existing epistemic behavior is
unchanged; the production serving profile turns it on at the input boundary.
"""

from __future__ import annotations

import re

# (verb-group, target-group) → each rule fires only when BOTH appear and are
# reasonably close. Patterns are intentionally narrow to keep precision high.
_INTENT = r"(?:how (?:do|can|to)|steps? to|give me|provide|write|create|build|make|synthesi[sz]e|produce|manufactur|instructions? (?:for|to)|help me)"

_RULES: list[tuple[str, str]] = [
    ("cbrn_weapons",
     rf"{_INTENT}\b.{{0,40}}\b(nerve agent|sarin|vx gas|mustard gas|anthrax|ricin|botulinum|bioweapon|biological weapon|chemical weapon|dirty bomb|nuclear (?:weapon|device)|weapons?[- ]grade|enrich(?:ing)? uranium)"),
    ("explosives",
     rf"{_INTENT}\b.{{0,40}}\b(bomb|i\.?e\.?d\.?|explosive device|detonator|pipe bomb|pressure cooker bomb|c-?4|tnt|napalm)"),
    ("csam",
     r"(?:child|minor|underage|pre[- ]?teen|toddler|kid)\b.{0,30}\b(?:sexual|porn|nude|naked|explicit|erotica)"
     r"|(?:sexual|porn|nude|naked|explicit|erotica)\b.{0,30}\b(?:child|minor|underage|pre[- ]?teen|toddler)"),
    ("malware",
     rf"{_INTENT}\b.{{0,40}}\b(ransomware|keylogger|spyware|rootkit|botnet|computer virus|worm that|trojan that|malware that|exploit (?:for|targeting)|sql injection payload|reverse shell)"),
    ("fraud_phishing",
     rf"{_INTENT}\b.{{0,40}}\b(phishing (?:email|page|kit|campaign)|steal (?:credentials|passwords|credit cards?)|carding|fake (?:invoice|identity documents?)|launder money)"),
]
_COMPILED = [(name, re.compile(rx, re.IGNORECASE | re.DOTALL)) for name, rx in _RULES]

_REASONS = {
    "cbrn_weapons": "request seeks operational uplift for weapons (CBRN) — prohibited by USAGE-POLICY.md §1.1",
    "explosives": "request seeks instructions to build an explosive device — prohibited by USAGE-POLICY.md §1.1",
    "csam": "request involves sexual content concerning minors — prohibited by USAGE-POLICY.md §1.2",
    "malware": "request seeks malware/intrusion tooling — prohibited by USAGE-POLICY.md §1.5",
    "fraud_phishing": "request seeks fraud/phishing tooling — prohibited by USAGE-POLICY.md §1.4",
}


def refusal_screen(text: str) -> dict:
    """Classify ``text`` against the acceptable-use prohibitions.

    Returns ``{block, category, reason, matched}``. ``block`` is True on the first
    matching rule; ``category``/``reason`` describe it; ``matched`` is the span.
    """
    t = text or ""
    for name, rx in _COMPILED:
        m = rx.search(t)
        if m:
            return {"block": True, "category": name, "reason": _REASONS[name],
                    "matched": m.group(0)[:80]}
    return {"block": False, "category": None, "reason": "no acceptable-use violation detected", "matched": None}


__all__ = ["refusal_screen"]
