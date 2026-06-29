# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cache-stable rollout factory: the RLVR data pump (Phase 1).

Ports DeepSeek-Reasonix's append-only prefix-cache stability + planner/executor
session split into a cheap generator of verifier-scored reasoning traces. See
``docs/06-Roadmap/DeepSeek-Reasonix-Integration-Roadmap.md``.
"""

from pipeline.rollout.cost import CostMeter
from pipeline.rollout.factory import RolloutFactory, ScriptedClient, offline_invariants
from pipeline.rollout.session import Message, Session, count_tokens
from pipeline.rollout.tools import DEFAULT_TOOLS, safe_calc

__all__ = [
    "CostMeter",
    "DEFAULT_TOOLS",
    "Message",
    "RolloutFactory",
    "ScriptedClient",
    "Session",
    "count_tokens",
    "offline_invariants",
    "safe_calc",
]
