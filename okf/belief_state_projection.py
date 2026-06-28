# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Thin, HONEST projection from OKF pages to the belief-dynamics ``BeliefState``.

This is the bridge that lets ``okf.decay_okf.plan_decay`` and the
``okf.forgetting_audit.ForgettingAudit`` ledger run over the *real* wiki corpus — the
piece that was missing while the dynamics layer was candidate-only.

==============================================================================
HONESTY CONTRACT — read before trusting any output of this projection.
==============================================================================
The OKF frontmatter today records ONLY provenance (authorConfidence, attributedAuthor,
tradition, sources, …). It records NO temporal signal and NO epistemic-dynamics signal.
Concretely, across all wiki pages:

    written_at            — NOT RECORDED  -> defaulted (see GENESIS_EPOCH)
    last_reinforced_at    — NOT RECORDED  -> defaulted (== GENESIS_EPOCH)
    surprise              — NOT RECORDED  -> defaulted to 0.0
    reinforcement_count   — NOT RECORDED  -> defaulted to 0

These are UNRECORDED PLACEHOLDERS, NOT MEASURED SIGNALS. A default-0 ``surprise`` does
not mean "this belief was unsurprising"; it means "we never measured it." A future reader
(human or agent) MUST NOT read any of these as evidence. Until the real signals are
captured (see RESEARCH_FOLLOWUP), any decay/quarantine output is driven by
``author_confidence`` (the one honest field) plus defaults that, by construction, cannot
themselves move a belief toward suppression — they can only leave it as-is.

What this projection therefore CANNOT do, honestly: produce a meaningful *time-decay*
(the timestamps are all equal, so ``age_days`` is always 0 and nothing decays for
"time"), or a meaningful *surprise-gated consolidation* (surprise is always 0). What it
CAN do honestly: record every consolidation selection in the tamper-evident audit
ledger (the genuine value of wiring this in), and let ``plan_decay`` run without error
over the real corpus, returning a mostly-trivial plan whose audit trail is real.

This is deliberately kept `level3Evidence: false`. It wires the ledger to a real run;
it does NOT earn Level-3 evidence.
==============================================================================

See RESEARCH_FOLLOWUP for what capturing the real signals would take.
"""

from __future__ import annotations

from okf.decay_okf import BeliefState

# Honest fixed timestamp for every belief, because the OKF frontmatter records no
# creation/observation time. Using a single shared GENESIS_EPOCH makes the
# (unrecorded) age of every belief identical, so time-decay is a no-op — which is the
# honest outcome: we cannot claim a belief is "old" when we never recorded when it
# arrived. This constant is intentionally exported so callers can quote it in reports.
GENESIS_EPOCH: float = 0.0

# The honest placeholder values for the unrecorded signals. Exported so the audit
# record can quote them verbatim — no reader can mistake them for measured values.
UNRECORDED_SURPRISE: float = 0.0
UNRECORDED_REINFORCEMENT: int = 0


def project_belief_state(node_id: str, author_confidence: str) -> BeliefState:
    """Project ONE OKF page to its dynamic ``BeliefState`` view, honestly.

    Only ``node_id`` and ``author_confidence`` come from the corpus; the dynamics fields
    (timestamps, surprise, reinforcement) are the documented unrecorded placeholders.
    See module docstring's HONESTY CONTRACT.
    """
    return BeliefState(
        node_id=node_id,
        author_confidence=author_confidence,
        # NOT RECORDED in the corpus — see GENESIS_EPOCH. Both timestamps equal so no
        # belief is ever treated as older than another (we have no arrival-time data).
        written_at=GENESIS_EPOCH,
        last_reinforced_at=GENESIS_EPOCH,
        # NOT RECORDED — unrecorded placeholders, NOT "zero surprise / never used".
        surprise=UNRECORDED_SURPRISE,            # 0.0 == unmeasured, not "unsurprising"
        reinforcement_count=UNRECORDED_REINFORCEMENT,  # 0 == unmeasured, not "never used"
    )


def project_corpus(pages) -> "list[BeliefState]":
    """Project all pages to belief states. ``pages`` is an iterable of OKF pages; the
    page's ``id`` and ``meta["authorConfidence"]`` are the only fields read.
    """
    out: list[BeliefState] = []
    for p in pages:
        conf = (p.meta or {}).get("authorConfidence") or "none_extant"
        out.append(project_belief_state(p.id, conf))
    return out


__all__ = [
    "project_belief_state",
    "project_corpus",
    "GENESIS_EPOCH",
    "UNRECORDED_SURPRISE",
    "UNRECORDED_REINFORCEMENT",
]
