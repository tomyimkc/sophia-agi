# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Contract error model (stable wire shape).

Every failure crosses the seam as ``{"error": {code, message, retryable}}`` with a
``code`` from a closed set. aihk-os switches on ``code`` and decides retries from
``retryable`` alone — so the set never grows in a MINOR and codes never change
meaning. Fail closed: when in doubt the service returns an error, never a
half-answer.
"""

from __future__ import annotations

# Closed set — additive only in a MINOR, never re-meaning a code without a MAJOR.
ERROR_CODES = (
    "BAD_REQUEST",      # malformed/invalid input; not retryable as-is
    "UNAUTHENTICATED",  # missing/invalid caller identity
    "BLP_VIOLATION",    # Bell-LaPadula breach; never silently downgraded
    "OVER_BUDGET",      # a budget cap was hit; stop-and-report
    "UNAVAILABLE",      # transient dependency failure; retryable
    "INTERNAL",         # unexpected fault; not retryable by the caller
)

# Which codes are safe for the caller to retry unchanged.
_RETRYABLE = {"UNAVAILABLE"}


class ContractError(Exception):
    """Raised internally; serialized to the wire error shape at the boundary."""

    def __init__(self, code: str, message: str, *, retryable: "bool | None" = None):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}; valid: {ERROR_CODES}")
        self.code = code
        self.message = message
        self.retryable = (code in _RETRYABLE) if retryable is None else bool(retryable)
        super().__init__(f"{code}: {message}")

    def to_wire(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "retryable": self.retryable}}


def error(code: str, message: str, *, retryable: "bool | None" = None) -> dict:
    """Build a wire error dict directly."""
    return ContractError(code, message, retryable=retryable).to_wire()
