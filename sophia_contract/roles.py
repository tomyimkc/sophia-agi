# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The 9 aihk-os service-role scopes — least-privilege capability registry.

Maps the One-Man-AI-Company playbook's nine role pipelines onto the contract's
``ScopeRegistry`` (sophia_contract/scopes.py). Each role may only perform its
listed ops up to its ``max_blp`` ceiling; an unknown role fails closed. This is
the enforcement behind "a public content agent can't read a Confidential client
note" — content_marketing is capped at UNCLASSIFIED, while the agent orchestrator
is the only role cleared to TOP_SECRET.

Pass ``role=`` on every contract call (record_claim / verify_claim / enqueue_task)
to activate enforcement; with no role the call is unrestricted (opt-in).
"""

from __future__ import annotations

from sophia_contract.scopes import Scope, ScopeRegistry

# role -> (ops, max_blp). Ceilings reflect what each pipeline legitimately handles:
# public-facing roles stay low; roles touching client code/data go higher; only the
# agent orchestrator is cleared to the top and may enqueue/admin.
ROLES_9 = ScopeRegistry({
    "role_01_prompting":         Scope(ops={"record_claim", "verify_claim"}, max_blp="CONFIDENTIAL"),
    "role_02_coding":            Scope(ops={"record_claim", "verify_claim"}, max_blp="SECRET"),
    "role_03_design":            Scope(ops={"record_claim", "verify_claim"}, max_blp="CONFIDENTIAL"),
    "role_04_video":             Scope(ops={"record_claim", "verify_claim"}, max_blp="CONFIDENTIAL"),
    "role_05_copywriting":       Scope(ops={"record_claim", "verify_claim"}, max_blp="CONFIDENTIAL"),
    # public-facing: cannot read above UNCLASSIFIED (the playbook's worked example)
    "role_06_content_marketing": Scope(ops={"record_claim", "verify_claim"}, max_blp="UNCLASSIFIED"),
    "role_07_automation":        Scope(ops={"record_claim", "verify_claim", "enqueue_task"}, max_blp="SECRET"),
    "role_08_data_analysis":     Scope(ops={"record_claim", "verify_claim"}, max_blp="SECRET"),
    # the orchestrator: full ops, top clearance
    "role_09_agents":            Scope(ops={"record_claim", "verify_claim", "enqueue_task", "admin"}, max_blp="TOP_SECRET"),
})

ROLE_NAMES = tuple(ROLES_9.roles.keys())
