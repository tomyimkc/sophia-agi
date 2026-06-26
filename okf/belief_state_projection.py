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
tradition, sources, …). It records NO temporal signal and (in the frontmatter) NO
epistemic-dynamics signal. Concretely, across all wiki pages:

    written_at            — NOT RECORDED  -> defaulted (see GENESIS_EPOCH)
    last_reinforced_at    — NOT RECORDED  -> defaulted (== GENESIS_EPOCH)
    surprise              — NOT in frontmatter; now MEASURABLE (see below)
    reinforcement_count   — NOT RECORDED  -> defaulted to 0

The default values are UNRECORDED PLACEHOLDERS, NOT MEASURED SIGNALS. A default-0
``surprise`` does not mean "this belief was unsurprising"; it means "we never measured
it." A future reader (human or agent) MUST NOT read any of these placeholders as evidence.

UPDATE — surprise is now a REAL, MEASURED signal. ``okf.surprise_signal`` computes
leave-one-out predictive surprise (``P(belief | the rest of memory)``) over the corpus —
the "retrieval/likelihood over the existing graph" RESEARCH_FOLLOWUP scoped. The MEASURED
path of this projection (``project_corpus_measured`` / passing ``surprise=`` to
``project_belief_state``) injects that real value, so ``surprise`` is no longer a
placeholder when measured. ``written_at``/``last_reinforced_at``/``reinforcement_count``
remain unrecorded — so time-decay and usage-reinforcement are still no-ops until signals
#1 and #2 land. The DEFAULT placeholder path (``project_corpus`` with no surprise) is kept
unchanged for callers that have not adopted measurement yet, and is still loudly honest.

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


def project_belief_state(
    node_id: str, author_confidence: str, *, surprise: float | None = None,
) -> BeliefState:
    """Project ONE OKF page to its dynamic ``BeliefState`` view, honestly.

    ``node_id`` and ``author_confidence`` come from the corpus. ``surprise`` is the
    MEASURED leave-one-out surprise (``okf.surprise_signal``) when supplied; when ``None``
    it falls back to the documented ``UNRECORDED_SURPRISE`` placeholder. The remaining
    dynamics fields (timestamps, reinforcement) are still unrecorded placeholders.
    See module docstring's HONESTY CONTRACT.
    """
    return BeliefState(
        node_id=node_id,
        author_confidence=author_confidence,
        # NOT RECORDED in the corpus — see GENESIS_EPOCH. Both timestamps equal so no
        # belief is ever treated as older than another (we have no arrival-time data).
        written_at=GENESIS_EPOCH,
        last_reinforced_at=GENESIS_EPOCH,
        # MEASURED when provided (okf.surprise_signal); else the unrecorded placeholder
        # (0.0 == unmeasured, NOT "unsurprising").
        surprise=UNRECORDED_SURPRISE if surprise is None else float(surprise),
        # NOT RECORDED — unrecorded placeholder, NOT "never used".
        reinforcement_count=UNRECORDED_REINFORCEMENT,
    )


def project_corpus(pages, *, surprise_by_id: "dict[str, float] | None" = None) -> "list[BeliefState]":
    """Project all pages to belief states. ``pages`` is an iterable of OKF pages; the
    page's ``id`` and ``meta["authorConfidence"]`` are read.

    ``surprise_by_id`` (optional) maps node id -> MEASURED surprise. When omitted, the
    surprise field stays the documented unrecorded placeholder (backward-compatible).
    """
    sbi = surprise_by_id or {}
    out: list[BeliefState] = []
    for p in pages:
        conf = (p.meta or {}).get("authorConfidence") or "none_extant"
        out.append(project_belief_state(p.id, conf, surprise=sbi.get(p.id)))
    return out


def project_corpus_measured(pages, *, graph=None) -> "list[BeliefState]":
    """Project all pages with MEASURED surprise from ``okf.surprise_signal``.

    This is the honest, evidence-grade projection: ``surprise`` is the real leave-one-out
    predictive signal, not a placeholder. Timestamps and reinforcement_count remain
    unrecorded (signals #1/#2 in RESEARCH_FOLLOWUP), so time-decay and usage-driven
    reinforcement are still no-ops — only the surprise channel is live.
    """
    from okf.surprise_signal import surprise_by_id as _measure
    return project_corpus(pages, surprise_by_id=_measure(pages, graph=graph))


__all__ = [
    "project_belief_state",
    "project_corpus",
    "project_corpus_measured",
    "GENESIS_EPOCH",
    "UNRECORDED_SURPRISE",
    "UNRECORDED_REINFORCEMENT",
]
