# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Backward-compat shim. This module was renamed to ``agent.verification_mcts`` for
clarity (the UCT search is real; the simulator it searches is hand-authored). The
implementation and public symbols are unchanged; import from ``agent.verification_mcts``
going forward. This shim re-exports everything so existing imports keep working."""
from __future__ import annotations

from agent.verification_mcts import (  # noqa: F401
    Action,
    Node,
    PlannerState,
    VerificationSimulator,
    __all__,
    initial_state,
    run_mcts,
)
