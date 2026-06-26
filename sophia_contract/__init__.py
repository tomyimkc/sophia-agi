# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia governance contract — the stable, versioned seam aihk-os pins against.

Import surface is intentionally tiny and stable:

    from sophia_contract import SophiaContract, CONTRACT_VERSION

The wire shapes (Claim, Verdict, error, describe) are documented in CONTRACT.md and
machine-checkable against schema/contract-1.0.0.json. Field names are the contract;
they do not change without a MAJOR bump.
"""

from __future__ import annotations

from sophia_contract.errors import ERROR_CODES, ContractError
from sophia_contract.intake import IntakeContract
from sophia_contract.models import HELD_REASONS, VERDICTS
from sophia_contract.service import CONTRACT_VERSION, SCHEMA_URL, SophiaContract
from sophia_contract.blp import BLP_LEVELS
from sophia_contract.scopes import Scope, ScopeRegistry

__all__ = [
    "SophiaContract",
    "CONTRACT_VERSION",
    "SCHEMA_URL",
    "ContractError",
    "ERROR_CODES",
    "VERDICTS",
    "HELD_REASONS",
    "BLP_LEVELS",
    "Scope",
    "ScopeRegistry",
    "IntakeContract",
]
