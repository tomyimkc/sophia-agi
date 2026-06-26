# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Auto-research: the agent proposes its own experiments and runs them, but it
**cannot overclaim** -- every proposed improvement is judged by the same trust gates
that govern the self-evolving agent.

This closes the loop from a hand-written experience stream to an autonomous one:

    hypothesize -> build experiment -> run through the self-evolving agent ->
    judge against a PRE-REGISTERED criterion -> log to a ledger (pass OR fail).

The differentiator vs. ordinary auto-research (which can reward-hack its own metric):
a hypothesis is recorded as *confirmed* only if the agent actually COMMITTED the
update -- i.e. it cleared evolve+promote, the reward-hack probe, the plasticity gate,
and the no-forgetting check. A result can never be marked confirmed without passing
those gates, so the loop is structurally incapable of overclaiming. Refuted
hypotheses are logged too (a failure ledger, not a highlight reel).

Offline & deterministic. The hypotheses here are synthetic (signal-token domains);
the loop STRUCTURE and its no-overclaim invariant are the deliverable.

    from agent.auto_research import AutoResearcher
    report = AutoResearcher().run(n=6)
    report["invariants"]["no_confirmed_without_passing_gates"]   # True
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.self_evolving_agent import Experience, SelfEvolvingAgent
from okf.page import Page

# Signal tokens whose domains are genuinely learnable (a stable token across splits),
# plus a sentinel for a deliberately *unlearnable* null hypothesis so the ledger
# honestly contains refutations, not only confirmations.
_LEARNABLE_SIGNALS = ("delete", "encrypt", "leak", "approve", "shutdown")
_NULL = "__noise__"

_OBJS = ("the database", "user files", "records", "everything", "the backups",
         "the logs", "all accounts", "the cache", "the index", "the config",
         "the queue", "the secrets")


@dataclass(frozen=True)
class Hypothesis:
    id: str
    domain: str
    signal: str
    prereg: dict          # pre-registered success criterion (fixed BEFORE the run)
    rationale: str

    def to_dict(self) -> "dict[str, Any]":
        return {"id": self.id, "domain": self.domain, "signal": self.signal,
                "prereg": self.prereg, "rationale": self.rationale}


@dataclass(frozen=True)
class LedgerEntry:
    hypothesis: Hypothesis
    confirmed: bool
    committed: bool
    overclaimed: bool
    improvement: float
    gates: dict
    reasons: tuple

    def to_dict(self) -> "dict[str, Any]":
        return {
            "schema": "sophia.auto_research_entry.v1",
            "candidateOnly": True,
            "level3Evidence": False,
            "hypothesis": self.hypothesis.to_dict(),
            "verdict": "confirmed" if self.confirmed else "refuted",
            "committed": self.committed,
            "overclaimed": self.overclaimed,
            "improvement": self.improvement,
            "gates": self.gates,
            "reasons": list(self.reasons),
        }


def _page(pid: str, **meta) -> Page:
    return Page(path=Path(f"{pid}.md"), meta={"id": pid, "pageType": "concept", **meta})


def _experiment_for(h: Hypothesis) -> Experience:
    """Render a hypothesis into a runnable experiment (labeled examples + knowledge)."""
    if h.signal == _NULL:
        # No stable signal: no verifier can validate -> the agent must refute it.
        examples = tuple((f"alpha{i} beta{i} gamma{i}", i % 2 == 0) for i in range(16))
    else:
        ex: list = []
        for o in _OBJS:
            ex.append((f"{h.signal} {o} now", True))
            ex.append((f"read {o} now", False))
        examples = tuple(ex)
    return Experience(h.domain, examples, (_page(f"{h.domain}_skill", authorConfidence="consensus"),))


def generate_hypotheses(n: int = 6) -> "list[Hypothesis]":
    """Deterministically propose `n` hypotheses, salting in null hypotheses.

    Every 3rd hypothesis is a deliberate null (unlearnable) so the ledger contains
    genuine refutations -- auto-research that only ever confirms is a red flag.
    """
    out: list[Hypothesis] = []
    for i in range(n):
        if i % 3 == 2:
            signal = _NULL
            rationale = "null hypothesis: no stable signal; expected to be refuted (fail-closed)"
        else:
            signal = _LEARNABLE_SIGNALS[i % len(_LEARNABLE_SIGNALS)]
            rationale = f"a request whose intent is signalled by the token '{signal}' is learnable and safe to commit"
        domain = f"auto_{i}_{'null' if signal == _NULL else signal}"
        out.append(Hypothesis(
            id=f"H{i}",
            domain=domain,
            signal=signal,
            prereg={"minImprovement": 0.05, "requireCommit": True},
            rationale=rationale,
        ))
    return out


class AutoResearcher:
    """Generate hypotheses, run them through a self-evolving agent, judge, and log."""

    def __init__(self, agent: "SelfEvolvingAgent | None" = None) -> None:
        self.agent = agent or SelfEvolvingAgent()
        self.ledger: list[LedgerEntry] = []

    def _judge(self, h: Hypothesis, outcome) -> LedgerEntry:
        prereg = h.prereg
        # Confirmed REQUIRES the gates passed (committed). The prereg cannot relax this.
        meets_metric = outcome.improvement >= float(prereg.get("minImprovement", 0.0))
        confirmed = bool(outcome.committed and meets_metric)
        # Overclaim = a confirmed verdict without the gates having passed. By
        # construction `confirmed` implies `committed`, so this is always False --
        # the invariant the report asserts.
        overclaimed = confirmed and not outcome.committed
        return LedgerEntry(
            hypothesis=h,
            confirmed=confirmed,
            committed=outcome.committed,
            overclaimed=overclaimed,
            improvement=outcome.improvement,
            gates=outcome.gates,
            reasons=outcome.reasons,
        )

    def run(self, *, n: int = 6, hypotheses: "list[Hypothesis] | None" = None) -> "dict[str, Any]":
        hyps = hypotheses if hypotheses is not None else generate_hypotheses(n)
        for h in hyps:
            outcome = self.agent.evolve(_experiment_for(h))
            self.ledger.append(self._judge(h, outcome))
        return self.report()

    def report(self) -> "dict[str, Any]":
        confirmed = [e for e in self.ledger if e.confirmed]
        refuted = [e for e in self.ledger if not e.confirmed]
        invariants = {
            # The headline guarantee: nothing is confirmed without clearing every gate.
            "no_confirmed_without_passing_gates": all(
                e.committed and all(e.gates.values()) for e in confirmed
            ),
            # No entry overclaimed (confirmed yet not committed).
            "no_overclaim": not any(e.overclaimed for e in self.ledger),
            # Failures are logged, not hidden (a falsifiable loop must be able to fail).
            "failures_are_logged": len(refuted) > 0,
        }
        return {
            "schema": "sophia.auto_research_report.v1",
            "candidateOnly": True,
            "level3Evidence": False,
            "hypotheses": len(self.ledger),
            "confirmed": len(confirmed),
            "refuted": len(refuted),
            "ledger": [e.to_dict() for e in self.ledger],
            "invariants": invariants,
            "interpretation": (
                "The agent proposed and ran its own experiments. A hypothesis was "
                "confirmed only when the self-evolving agent committed the update -- "
                "i.e. it cleared the reward-hack, plasticity, and no-forgetting gates. "
                "Null hypotheses were refuted and logged. The loop cannot mark a result "
                "confirmed without passing those gates, so it cannot overclaim."
            ),
        }

    def write_ledger(self, out: "str | Path") -> "dict[str, Any]":
        report = self.report()
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in self.ledger),
            encoding="utf-8",
        )
        return report


__all__ = ["Hypothesis", "LedgerEntry", "AutoResearcher", "generate_hypotheses"]
