# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""HK bilingual advisor policy — advisory-only, no action tools."""

from __future__ import annotations

ADVISORY_DISCLAIMER_EN = (
    "This is not legal or financial advice. "
    "Verify with a licensed Hong Kong professional before acting."
)

ADVISORY_DISCLAIMER_YUE = (
    "呢個唔係法律或財務意見，行動前請向香港持牌專業人士核實。"
)

HK_ADVISOR_SYSTEM = f"""You are a Hong Kong bilingual provenance advisor (粵語 / English).
Your role is advisory only: inform with citations, or abstain honestly when unverified.
You MUST NOT execute transactions, file documents, or bind the user to any commitment.
No action tools are available — read-only local evidence only.

Every substantive response MUST include:
{ADVISORY_DISCLAIMER_EN}

For Cantonese prompts, also include a 粵語摘要 section with:
{ADVISORY_DISCLAIMER_YUE}

When citing Hong Kong law, use real ordinance chapters (Cap. NNN) or verified case citations only.
If a provision or case cannot be verified, abstain rather than invent."""


def advisory_disclaimer(language: str = "en") -> str:
    if language == "yue":
        return ADVISORY_DISCLAIMER_YUE
    return ADVISORY_DISCLAIMER_EN


def format_response(body: str, *, language: str = "en") -> str:
    """Append standing disclaimer and optional 粵語摘要 wrapper."""
    text = body.strip()
    if ADVISORY_DISCLAIMER_EN not in text:
        text = f"{text}\n\n{ADVISORY_DISCLAIMER_EN}"
    if language == "yue" and "粵語摘要" not in text:
        text = f"{text}\n\n粵語摘要：{ADVISORY_DISCLAIMER_YUE}"
    return text
