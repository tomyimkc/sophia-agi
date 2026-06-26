# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Backward-compat shim. This module was renamed to ``agent.tabular_transition_model``
because the old name ("predictive world model") overstated a tabular transition-count
table into a "learned world model". The implementation and public symbols are unchanged;
import from ``agent.tabular_transition_model`` going forward. This shim re-exports
everything so existing imports keep working."""
from __future__ import annotations

from agent.tabular_transition_model import (  # noqa: F401
    PredictiveWorldModel,
    Transition,
    __all__,
    demo_world_model_report,
    write_world_model_report,
)
