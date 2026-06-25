# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Request Triage + Intake Contract public helpers."""

from __future__ import annotations

from agent.intake.core import (
    CONTRACT_PROMPT_HEADER,
    INTAKE_PROMPT_HEADER,
    append_intake_audit,
    audit_entry,
    compose_harness_prompt,
    role_for_case,
    run_intake,
    scope_for_role,
    validate_execution_within_contract,
)

__all__ = [
    "CONTRACT_PROMPT_HEADER",
    "INTAKE_PROMPT_HEADER",
    "append_intake_audit",
    "audit_entry",
    "compose_harness_prompt",
    "role_for_case",
    "run_intake",
    "scope_for_role",
    "validate_execution_within_contract",
]
