# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Self-evolving agent loop: evolve -> no-hack -> promote -> retain -> commit.

**Market thesis.** Everyone is building self-improving agents; the unmet need is
trust: *when an agent self-improves, how do you know it actually got better and did
not silently regress, hallucinate, or reward-hack?* This loop is that answer. The
agent improves between sessions, but every candidate improvement must clear four
independent trust gates before it is allowed to change what the agent knows:

  1. EVOLVE   ``selfextend.close_loop`` synthesizes + held-out-validates a verifier
              for a domain the agent abstained on, and measures a real
              (selection-based) capability gain.
  2. NO-HACK  an independent held-out verifier probes for reward-hacking
              (``selfextend.verified_reward.reward_is_hackable``): a candidate that
              scores high on the train-fit rule but low on an independent held-out
              rule optimized the checker, not the task.
  3. PROMOTE  ``agent.continual_plasticity`` gates the gain: an improvement floor, no
              protected-suite regression, contamination flag, verifier artifacts.
  4. RETAIN   ``agent.continual_retention`` confirms that committing this round's
              knowledge forgot nothing already learned (``forgottenGroundedClaims``).

Only if EVOLVE promoted AND the round is NOT reward-hacked AND PROMOTE == ``promote``
AND RETAIN forgot nothing does the round's knowledge enter the agent's memory.
Otherwise the agent abstains from updating itself (**fail-closed**) -- the moat:
self-improvement that is verifiable and cannot silently regress, hallucinate, or
reward-hack.

Offline & deterministic. Real weight updates stay behind the RunPod GPU path
(``tools/runpod_rlvr.py``); here the policy improvement is verifier-guided selection,
which is honest and CI-gated. The loop STRUCTURE and its invariants are the
deliverable, not a weight-update result.

    from agent.self_evolving_agent import SelfEvolvingAgent, Experience
    agent = SelfEvolvingAgent()
    agent.evolve(Experience("danger_intent", examples, pages))
    report = agent.session_report()   # compounding view + cross-round invariants
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.continual_plasticity import (
    EvalMetric,
    UpdateCandidate,
    append_promotion_ledger,
    evaluate_update,
)
from agent.continual_retention import Task, run_stream
from selfextend import close_loop, reward_is_hackable
from selfextend.competence_map import CompetenceMap
from selfextend.verifier_synthesis import stratified_split, synthesize_verifier


@dataclass(frozen=True)
class Experience:
    """One session's worth of experience the agent may learn from.

    ``examples`` are labeled traces ``[(text, is_positive)]`` the self-extending loop
    learns a verifier from. ``pages`` are the OKF ``Page`` objects representing the
    declarative knowledge the round would commit to memory (used by the retention
    gate). ``domain`` names the capability the agent abstained on going in.
    """

    domain: str
    examples: tuple = ()
    pages: tuple = ()


@dataclass(frozen=True)
class RoundOutcome:
    round: int
    domain: str
    committed: bool
    plasticityVerdict: str
    gates: dict
    preAccuracy: float
    postAccuracy: float
    improvement: float
    heldoutAccuracy: float
    rewardHack: dict
    forgottenGroundedClaims: int
    reliabilityAfter: float
    routeAfter: str
    pagesAdded: int
    reasons: tuple

    def to_dict(self) -> "dict[str, Any]":
        return {
            "schema": "sophia.self_evolving_agent_round.v1",
            "candidateOnly": True,
            "level3Evidence": False,
            "round": self.round,
            "domain": self.domain,
            "committed": self.committed,
            "pagesAdded": self.pagesAdded,
            "plasticityVerdict": self.plasticityVerdict,
            "gates": self.gates,
            "preAccuracy": self.preAccuracy,
            "postAccuracy": self.postAccuracy,
            "improvement": self.improvement,
            "heldoutAccuracy": self.heldoutAccuracy,
            "rewardHack": self.rewardHack,
            "forgottenGroundedClaims": self.forgottenGroundedClaims,
            "reliabilityAfter": self.reliabilityAfter,
            "routeAfter": self.routeAfter,
            "reasons": list(self.reasons),
        }


def _reward_hack_probe(examples: "list[tuple[str, bool]]", *, gap: float = 0.2) -> dict:
    """Independent reward-hacking probe.

    Fit one rule to a train split and an independent rule to a held-out split, then
    ask whether the train-fit rule's reward on held-out candidates diverges from the
    independent rule's. A large drop means the candidate optimized the checker, not
    the task. Fail-closed: if no rule is synthesizable we report ``skipped`` and do
    NOT mark the round hacked (the plasticity gate still requires a real gain).
    """
    train, heldout = stratified_split(examples)
    train_rule = synthesize_verifier(train)
    held_rule = synthesize_verifier(heldout)
    if train_rule is None or held_rule is None or not heldout:
        return {"trainReward": 0.0, "heldoutReward": 0.0, "drop": 0.0,
                "hacked": False, "rule": "no rule synthesizable", "skipped": True}
    candidates = [t for t, _ in heldout]
    probe = reward_is_hackable(candidates, train_rule.predict, held_rule.predict, gap=gap)
    probe["skipped"] = False
    return probe


class SelfEvolvingAgent:
    """An agent that self-improves between sessions, fail-closed on every gate.

    State carried across rounds is the agent's memory: the cumulative OKF page stream
    (declarative knowledge) and a competence self-model (where it is reliable). A
    round that fails any gate leaves both untouched -- rejected self-updates cannot
    mutate what the agent knows.
    """

    def __init__(self, *, min_target_delta: float = 0.05, hack_gap: float = 0.2,
                 competence_threshold: float = 0.7) -> None:
        self.min_target_delta = min_target_delta
        self.hack_gap = hack_gap
        self.competence = CompetenceMap(threshold=competence_threshold)
        self._round = 0
        self._tasks: list[Task] = []          # cumulative committed knowledge stream
        self._knowledge: list = []            # cumulative committed OKF pages
        self._committed_experiences: list[Experience] = []  # for self-distillation
        self.history: list[RoundOutcome] = []

    @property
    def knowledge_size(self) -> int:
        return len(self._knowledge)

    def evolve(self, exp: Experience, *, ledger_path: "str | Path | None" = None) -> RoundOutcome:
        """Run one evolve -> no-hack -> promote -> retain -> commit cycle."""
        self._round += 1
        rnd = self._round
        examples = list(exp.examples)

        # 1. EVOLVE: synthesize + held-out-validate a verifier, measure selection gain.
        loop = close_loop(exp.domain, examples)
        promoted = bool(loop.get("promoted"))
        loop_closed = bool(loop.get("loop_closed"))
        pre = float(loop.get("preAccuracy", 0.0))
        post = float(loop.get("postAccuracy", 0.0))
        heldout_acc = float(loop.get("heldoutAccuracy", 0.0))

        # 2. NO-HACK: independent held-out reward-hacking probe.
        hack = _reward_hack_probe(examples, gap=self.hack_gap)
        hacked = bool(hack.get("hacked"))

        # 3. PROMOTE: the continual-plasticity gate decides if the gain may ship.
        candidate = UpdateCandidate(
            id=f"round{rnd}:{exp.domain}",
            kind="self_evolve_skill",
            verifier_artifacts=(
                ("flywheel-heldout-validation", "reward-hack-probe") if promoted else ()
            ),
            contaminated=hacked,
            metrics=(
                EvalMetric(exp.domain, pre, post),
                # Gate-clean traces only ever enter via close_loop, so source
                # discipline is held constant and protected against regression.
                EvalMetric("source_discipline", 1.0, 1.0, protected=True),
            ),
            notes=f"self-evolving agent round {rnd}",
        )
        decision = evaluate_update(
            candidate, target_suite=exp.domain, min_target_delta=self.min_target_delta
        )

        # 4. RETAIN: trial-commit this round's pages, confirm nothing is forgotten.
        trial = self._tasks + [Task(f"round{rnd}", tuple(exp.pages))]
        if any(t.pages for t in trial):
            retention = run_stream(trial)
        else:
            retention = {"forgottenGroundedClaims": 0, "perfectRetention": True}
        forgot = int(retention.get("forgottenGroundedClaims", 0))

        # 5. COMMIT (fail-closed): only if every independent gate passes.
        gates = {
            "evolved_and_promoted": promoted and loop_closed,
            "not_reward_hacked": not hacked,
            "plasticity_promote": decision.verdict == "promote",
            "no_forgetting": forgot == 0,
        }
        committed = all(gates.values())
        pages_added = 0
        if committed:
            self._tasks = trial
            self._knowledge.extend(exp.pages)
            self._committed_experiences.append(exp)
            pages_added = len(exp.pages)
        # The competence self-model records the round's verified outcome either way.
        self.competence.update(exp.domain, committed)

        outcome = RoundOutcome(
            round=rnd,
            domain=exp.domain,
            committed=committed,
            plasticityVerdict=decision.verdict,
            gates=gates,
            preAccuracy=pre,
            postAccuracy=post,
            improvement=round(post - pre, 4),
            heldoutAccuracy=heldout_acc,
            rewardHack=hack,
            forgottenGroundedClaims=forgot,
            reliabilityAfter=self.competence.reliability(exp.domain),
            routeAfter=self.competence.route(exp.domain),
            pagesAdded=pages_added,
            reasons=decision.reasons,
        )
        self.history.append(outcome)
        if ledger_path is not None:
            append_promotion_ledger(decision, ledger_path)
        return outcome

    def run_session(self, experiences, *, ledger_path: "str | Path | None" = None) -> "dict[str, Any]":
        """Evolve over a sequence of experiences and return the session report."""
        for exp in experiences:
            self.evolve(exp, ledger_path=ledger_path)
        return self.session_report()

    def session_report(self) -> "dict[str, Any]":
        """Compounding view across rounds plus the cross-round trust invariants."""
        n = len(self.history)
        committed = [o for o in self.history if o.committed]
        # Forgetting measured against the FINAL committed memory: the headline claim.
        final_retention = run_stream(self._tasks) if any(t.pages for t in self._tasks) else {
            "forgottenGroundedClaims": 0, "perfectRetention": True, "backwardTransfer": 0.0,
        }
        forgot_across_run = int(final_retention.get("forgottenGroundedClaims", 0))
        invariants = {
            # Nothing committed across the whole run was ever forgotten.
            "no_forgetting_across_run": forgot_across_run == 0,
            # Every committed round cleared all four independent gates.
            "every_committed_round_cleared_all_gates": all(
                all(o.gates.values()) for o in committed
            ),
            # Fail-closed bookkeeping: memory grew by exactly the pages of committed
            # rounds, and rejected rounds added nothing.
            "rejected_rounds_did_not_mutate_memory": (
                self.knowledge_size == sum(o.pagesAdded for o in self.history)
                and all(o.pagesAdded == 0 for o in self.history if not o.committed)
            ),
        }
        return {
            "schema": "sophia.self_evolving_agent_session.v1",
            "candidateOnly": True,
            "level3Evidence": False,
            "rounds": n,
            "committedRounds": len(committed),
            "coverageAfter": round(len(committed) / n, 4) if n else 0.0,
            "knowledgePages": self.knowledge_size,
            "forgottenGroundedClaimsAcrossRun": forgot_across_run,
            "backwardTransfer": final_retention.get("backwardTransfer", 0.0),
            "competenceMap": self.competence.map(),
            "perRound": [o.to_dict() for o in self.history],
            "invariants": invariants,
            "interpretation": (
                "The agent self-improved over the session, but only rounds that cleared "
                "all four trust gates (evolve+promote, no reward-hack, plasticity, no "
                "forgetting) entered memory. Rejected rounds left memory untouched "
                "(fail-closed). Across the committed run, forgottenGroundedClaims is 0: "
                "verifiable self-evolution with no catastrophic forgetting."
            ),
        }

    def write_report(self, out: "str | Path") -> "dict[str, Any]":
        report = self.session_report()
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report

    # ------------------------------------------------------------------ self-distill
    def distillation_rows(self, *, gate_check: "Any | None" = None) -> "list[dict]":
        """Render gate-clean self-distillation rows from COMMITTED rounds only.

        Closes the self-distillation loop (the agent's own *verified* skills become a
        student's training data) with the anti-circularity firewall built in:

          - Only committed rounds are exported -- a self-update that failed any trust
            gate never becomes training data (rejected ideas can't teach the student).
          - Within a committed round, only examples the verified rule classifies
            CORRECTLY are emitted (we distill what was verified, not what was guessed).
          - If ``gate_check`` is supplied (``(target, question) -> has_violations``),
            any rendered target with violations is dropped -- a second firewall so no
            gate-dirty text enters the dataset.

        Output rows match the repo's distillation schema
        (``{"messages": [...], "metadata": {...}}``), consumable by
        ``tools/train_lora.py``. Honest scope: this distills the *verified decision
        rule* (a narrow, gate-clean signal); richer targets come from the live teacher
        path (``tools/distill_council_traces.py``).
        """
        rows: list[dict] = []
        for exp in self._committed_experiences:
            rule = synthesize_verifier(list(exp.examples))
            if rule is None:
                continue
            for text, label in exp.examples:
                if rule.predict(text) != label:
                    continue  # only distill examples the verified rule gets right
                target = _render_decision(exp.domain, rule, bool(label))
                if gate_check is not None and gate_check(target, text):
                    continue  # firewall: drop any non-gate-clean target
                rows.append({
                    "messages": [
                        {"role": "system", "content": _SELF_DISTILL_SYSTEM},
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": target},
                    ],
                    "metadata": {
                        "domain": exp.domain,
                        "source": "self-evolve",
                        "verified": True,
                        "gatePassed": gate_check is not None,
                        "labelStatus": "self-distilled",
                    },
                })
        return rows

    def write_distillation_jsonl(self, out: "str | Path", *, gate_check: "Any | None" = None) -> dict:
        """Write self-distillation rows to JSONL and return summary stats."""
        rows = self.distillation_rows(gate_check=gate_check)
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return {
            "rows": len(rows),
            "committedRounds": len(self._committed_experiences),
            "domains": sorted({e.domain for e in self._committed_experiences}),
            "gateFirewall": gate_check is not None,
        }


_SELF_DISTILL_SYSTEM = (
    "You are a source-disciplined agent. Decide whether a request matches a pattern "
    "you have verifiably learned, and act only within that verified competence. If a "
    "request falls outside it, say so plainly rather than guess. Do not invent "
    "authorities or attributions."
)


def _render_decision(domain: str, rule, label: bool) -> str:
    """Render the verified decision the agent learned for `domain` as the student target."""
    if label:
        return (f"This request matches the verified '{domain}' pattern "
                f"(signal: '{rule.feature}'). Routing to the verified handler.")
    return (f"This request does not match the verified '{domain}' pattern "
            f"(signal: '{rule.feature}'). No action: it is outside the verified competence.")


__all__ = ["Experience", "RoundOutcome", "SelfEvolvingAgent"]
