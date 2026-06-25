# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""HK bilingual advisor training verifier and policy helpers."""

from agent.hk_advisor.policy import (
    ADVISORY_DISCLAIMER_EN,
    ADVISORY_DISCLAIMER_YUE,
    HK_ADVISOR_SYSTEM,
    advisory_disclaimer,
    format_response,
)
from agent.hk_advisor.verifier import (
    Verdict,
    trace_passes,
    verify_abstention,
    verify_advisory_boundary,
    verify_bilingual_fidelity,
    verify_citation,
    verify_no_fabrication,
    verify_substance,
    verify_trace,
)

__all__ = [
    "ADVISORY_DISCLAIMER_EN",
    "ADVISORY_DISCLAIMER_YUE",
    "HK_ADVISOR_SYSTEM",
    "Verdict",
    "advisory_disclaimer",
    "format_response",
    "trace_passes",
    "verify_abstention",
    "verify_advisory_boundary",
    "verify_bilingual_fidelity",
    "verify_citation",
    "verify_no_fabrication",
    "verify_substance",
    "verify_trace",
]
