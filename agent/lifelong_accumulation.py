# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Closed-loop lifelong-accumulation benchmark — the 7 candidate factors as ONE system.

Sophia discipline. This is a CANDIDATE measurement, not a finished general
intelligence and NOT a claim of one. Every emitted report carries
``candidateOnly: True`` and ``validated: False``. The headline is an honest
measurement — *does the system net-accumulate retained capability over a long
stream without catastrophic forgetting, and does it beat a frozen parametric
baseline?* — never a verdict about AGI. Live multi-judge grading is OUT OF SCOPE
for this offline core (CI cannot run LLMs deterministically); a clearly-marked
seam (:data:`LLM_JUDGE_HOOK`) is left for it but never called.

What this milestone turns the seven isolated factors into one measured loop:

  #6 governed_rsi CAGE IN THE LOOP — every fact/skill taught in an episode is first
     proposed to a :class:`agent.governed_rsi.GovernedRSI` via ``.step(Proposal(...))``;
     ONLY proposals whose ``decision == "committed"`` are written into the
     :class:`agent.continual_qa.GraphBackedSystem`. A POISONED proposal (single
     low-trust source / one independence group) and a FORBIDDEN/parametric
     proposal are seeded into the stream and asserted to be REJECTED and to never
     enter the graph (this exercises #4 poison-resistance and #2 verifiability
     THROUGH the cage).
  #7 symbol_identity — every admitted fact gets a ``version_tag`` citation pin.
  #5 belief / retention — per-episode :class:`agent.continual_retention.Snapshot`
     objects feed ``build_report`` for the retention matrix + forgotten split;
     this is the catastrophic-forgetting measurement.
  #3 competence_model — graded ``(domain, correct)`` outcomes feed
     ``build_competence_model`` + ``learning_priorities`` (the "what to learn next"
     measured-weakness ranking).
  #1 skills — SKILL proposals ride the same cage path so skills accumulate
     alongside facts (and never regress).
  control-flow gap + #knowledge_gap_log — ``control_flow_report`` with a
     :class:`agent.continual_qa_controller.LexicalController` measures what the
     LLM-as-router layer costs; abstain/miss queries are logged as gaps and turned
     into a ``gap_worklist`` (misses -> what to enrich next).

THE HONEST METRIC (not gamed). After EACH episode we re-ask the queries for ALL
facts taught so far (a FIXED, GROWING held-out evaluation), not only the
just-taught fact. ``netCapabilityCurve[i].graphCorrectCumulative`` is the count of
distinct facts the graph_backed system answers CORRECTLY (graph-backed assert of
the right id) over that whole cumulative set. It must be monotonically
non-decreasing EXCEPT across a deliberately-retracted fact (counted as deliberate
unlearning, never as forgetting). The frozen :class:`ParametricBaseline` keeps its
t0 facts but cannot learn later ones, so its cumulative-correct stays ~flat at the
t0 count — the contrast that makes "accumulation" meaningful.

    from agent.lifelong_accumulation import make_lifelong_stream, run_accumulation, accumulates_cleanly
    report = run_accumulation(make_lifelong_stream())
    accumulates_cleanly(report)   # True
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.competence_model import build_competence_model, learning_priorities
from agent.continual_qa import (
    Episode as CPQAEpisode,
    ParametricBaseline,
    Query,
    _score_routed,
    build_vocab,
    control_flow_report,
)
from agent.continual_qa import GraphBackedSystem
from agent.continual_qa_controller import LexicalController, OracleController
from agent.continual_retention import Snapshot, build_report
from agent.governed_rsi import (
    GovernedRSI,
    Proposal,
    _good_sources,
    _verifiable_examples,
)
from agent.knowledge_gap_log import gap_worklist
from agent.symbol_identity import version_tag
from okf import build_graph
from okf.page import Page

__all__ = [
    "SCHEMA",
    "LLM_JUDGE_HOOK",
    "FactSpec",
    "Episode",
    "make_lifelong_stream",
    "run_accumulation",
    "accumulates_cleanly",
]

SCHEMA = "sophia.lifelong_accumulation.v1"

# Clearly-marked seam for the live multi-judge grader. The offline core NEVER
# calls it (CI cannot run LLMs deterministically). A deployment may set this to a
# callable ``judge(report) -> dict`` to attach live grading; left None here.
LLM_JUDGE_HOOK = None

# OKF domains the synthetic stream spreads facts across (kept to the real schema
# domains so okf.build_graph / competence_model treat them as known).
_DOMAINS: tuple[str, ...] = ("philosophy", "psychology", "history", "religion")


# --------------------------------------------------------------------------- #
# Stream specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FactSpec:
    """One unit taught (or attempted) in an episode, carrying BOTH its OKF page
    content and the cage-proposal metadata that decides admission.

    Fields
    ------
    id        : stable fact/skill id (the OKF page id and the proposal id).
    domain    : OKF domain label.
    kind      : "fact" | "skill".
    poison    : True => seeded as a poisoned proposal (single low-trust source /
                one independence group) that the cage MUST reject.
    forbidden : True => seeded as a forbidden/parametric proposal the cage MUST
                reject (kind forced parametric so ``weights_frozen`` rejects it).
    """

    id: str
    domain: str
    kind: str = "fact"
    poison: bool = False
    forbidden: bool = False

    # ------------------------------------------------------------------ #
    def proposal(self) -> Proposal:
        """Build the GovernedRSI proposal for this spec.

        Genuine units get the clean ``_verifiable_examples`` (synthesize admits)
        and ``_good_sources`` (two independent trusted sources, poison-clean).
        A poisoned unit keeps verifiable examples but is given a single low-trust
        source in one independence group, so the poison-resistance check rejects
        it. A forbidden unit is a parametric/weight kind the non-parametric cage
        rejects by construction.
        """
        if self.forbidden:
            return Proposal(
                id=self.id,
                kind="weight_update",
                domain=self.domain,
                payload={"text": f"parametric update for {self.id}"},
                examples=_verifiable_examples(),
            )
        if self.poison:
            return Proposal(
                id=self.id,
                kind="fact",
                domain=self.domain,
                payload={"text": f"poisoned claim {self.id}"},
                examples=_verifiable_examples(),
                sources=(
                    {
                        "sourceId": "single.untrusted",
                        "trust": 0.1,
                        "confidence": 0.5,
                        "independenceGroup": "g1",
                    },
                ),
            )
        return Proposal(
            id=self.id,
            kind=self.kind,
            domain=self.domain,
            payload={"text": f"verifiable {self.kind} {self.id}", "name": self.id},
            examples=_verifiable_examples(),
            sources=_good_sources(),
        )

    def page(self) -> Page:
        """The OKF page written into the graph IFF the cage commits this unit.

        Simple, additive, self-grounded concept page (no derivesFrom), so a clean
        admit grounds immediately and an additive stream forgets nothing."""
        return Page(
            path=Path(f"{self.id}.md"),
            meta={
                "id": self.id,
                "pageType": "concept",
                "domain": self.domain,
                "authorConfidence": "attributed",
            },
        )


@dataclass(frozen=True)
class Episode:
    """One step of the lifelong stream.

    ``specs``   : the units PROPOSED this episode (some may be poisoned/forbidden).
    ``retract`` : fact ids deliberately unlearned this episode (counted as
                  deliberate unlearning, never as forgetting).
    The cumulative held-out query set is re-derived per episode in
    :func:`run_accumulation`, so an episode carries no hand-written queries.
    """

    id: str
    specs: tuple = ()
    retract: tuple = ()


# --------------------------------------------------------------------------- #
# Deterministic synthetic long-horizon stream
# --------------------------------------------------------------------------- #
def make_lifelong_stream(
    *,
    seed: int = 0,
    n_episodes: int = 12,
    facts_per_episode: int = 2,
) -> "list[Episode]":
    """A DETERMINISTIC synthetic long-horizon stream across OKF domains.

    Each episode teaches a few new genuine facts (round-robin over domains). The
    cumulative query set (re-asked every episode) grows monotonically. Exactly one
    POISONED proposal and one FORBIDDEN/parametric proposal are seeded mid-stream
    (they must be rejected by the cage and never enter the graph), at least a
    couple of SKILL proposals accumulate, and one mid-stream episode DELIBERATELY
    retracts an earlier fact. Built with a seeded ``random.Random(seed)``; no
    wall-clock anywhere, so two builds with the same seed are byte-identical.
    """
    rng = random.Random(seed)
    episodes: list[Episode] = []

    # Episodes chosen for the special events (deterministic given n_episodes).
    poison_ep = min(2, n_episodes - 1)        # a poisoned proposal lands here
    forbidden_ep = min(3, n_episodes - 1)     # a forbidden/parametric proposal here
    skill_eps = {min(1, n_episodes - 1), min(4, n_episodes - 1)}  # skills accumulate
    retract_ep = max(1, n_episodes - 3)       # deliberate retraction late in stream

    # Track a genuine fact taught at t0 so the deliberate retraction targets a fact
    # the baseline DOES know. The contrast is then maximally honest: the graph
    # deliberately unlearns it and correctly abstains thereafter (its curve stays
    # monotone — the retraction is counted as deliberate unlearning, not a miss),
    # while the frozen baseline CANNOT unlearn and keeps fabricating the retracted
    # fact, so its cumulative-correct can only stay flat or fall, never rise.
    retract_target: str | None = None

    counter = 0
    for e in range(n_episodes):
        specs: list[FactSpec] = []
        for _ in range(facts_per_episode):
            domain = _DOMAINS[counter % len(_DOMAINS)]
            # rng.random() advances the stream's deterministic state (and would
            # drive any future jittered choices); kept so the seam is real.
            rng.random()
            fid = f"fact_{counter:03d}_{domain}"
            specs.append(FactSpec(id=fid, domain=domain, kind="fact"))
            if e == 0 and retract_target is None:
                retract_target = fid   # a t0 fact the baseline knows
            counter += 1

        if e in skill_eps:
            domain = _DOMAINS[counter % len(_DOMAINS)]
            specs.append(FactSpec(id=f"skill_{counter:03d}", domain=domain, kind="skill"))
            counter += 1

        if e == poison_ep:
            specs.append(
                FactSpec(id=f"poisoned_{e:02d}", domain="psychology", kind="fact", poison=True)
            )
        if e == forbidden_ep:
            specs.append(
                FactSpec(id=f"forbidden_{e:02d}", domain="history", kind="fact", forbidden=True)
            )

        retract: tuple = ()
        if e == retract_ep and retract_target is not None:
            retract = (retract_target,)

        episodes.append(Episode(id=f"ep{e:02d}", specs=tuple(specs), retract=retract))

    return episodes


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _query_for(fid: str) -> Query:
    """A held-out recall query for a fact id (expect assert)."""
    return Query(id=f"q_{fid}", target=fid, expect="assert", type="recall",
                 text=f"recall {fid.replace('_', ' ')}")


def _to_cpqa_episodes(
    episodes: "list[Episode]", admitted_pages: "dict[str, list[Page]]",
) -> "list[CPQAEpisode]":
    """Build CPQA episodes for the control-flow report: each episode learns the
    pages ADMITTED that episode (post-cage), retracts its retractions, and asks the
    CUMULATIVE held-out query set for every genuine fact taught (admitted) so far.

    Rejected (poison/forbidden) ids are never learned, so they never get a query
    (the control-flow accuracy is measured over the real, admitted capability)."""
    cpqa: list[CPQAEpisode] = []
    asked: list[str] = []          # cumulative admitted fact ids (query targets)
    retracted: set[str] = set()
    for ep in episodes:
        pages = admitted_pages.get(ep.id, [])
        for p in pages:
            if p.id not in asked:
                asked.append(p.id)
        retracted.update(ep.retract)
        # Cumulative held-out queries: assert for live facts, abstain for retracted.
        queries = []
        for fid in asked:
            if fid in retracted:
                queries.append(Query(id=f"q_{fid}", target=fid, expect="abstain",
                                      type="unlearning", text=f"recall {fid.replace('_', ' ')}"))
            else:
                queries.append(_query_for(fid))
        cpqa.append(CPQAEpisode(id=ep.id, learn=tuple(pages),
                                retract=tuple(ep.retract), queries=tuple(queries)))
    return cpqa


# --------------------------------------------------------------------------- #
# The full closed-loop run
# --------------------------------------------------------------------------- #
def run_accumulation(episodes, *, controller=None, seed: int = 0) -> "dict[str, Any]":
    """Stream the episodes through the FULL closed loop and return the honest report.

    For each episode: (1) propose every unit to ONE shared GovernedRSI cage; only
    ``committed`` units' OKF pages are written into the GraphBackedSystem; rejected
    (poison/forbidden) units never enter. (2) Re-ask the FIXED, GROWING held-out
    query set for ALL genuine facts taught so far (cumulative) — the honest metric.
    (3) Snapshot belief state for the retention measurement, freeze the parametric
    baseline at t0, accumulate graded records + knowledge gaps.

    The ``netCapabilityCurve`` scores the KNOWLEDGE SUBSTRATE under an
    ``OracleController`` (perfect routing), exactly as ``run_benchmark`` does by
    default, so the curve isolates *what the store can answer* — net-accumulated
    retained capability — from routing noise. The COST of the real router is
    measured separately and honestly as ``controlFlowGap`` (oracle accuracy minus
    ``controller``-routed accuracy). This is why a deliberate-retraction dip in the
    curve is attributable to the retraction alone, never to a routing miss.

    The report is byte-identical across two runs with the same seed (no wall-clock,
    no randomness inside the loop; the synthetic stream is the only stochastic input
    and it is seeded upstream).
    """
    controller = controller or LexicalController()
    # The honest net-capability metric isolates the substrate: route with the
    # oracle so a query's correctness reflects the knowledge store, not the router.
    substrate_router = OracleController()

    cage = GovernedRSI()
    gb = GraphBackedSystem()
    baseline: "ParametricBaseline | None" = None

    snapshots: list[Snapshot] = []
    admitted_pages: dict[str, list[Page]] = {}
    committed_ids: list[str] = []
    rejected_ids: list[str] = []
    poison_rejected: list[str] = []
    cage_breaches: list[str] = []        # rejected ids that nonetheless entered the graph
    intentional: set[str] = set()        # ids removed on purpose (retraction + cascade)
    records: list[dict] = []             # competence records: {domain, confidence, correct}
    gaps: list[dict] = []                # knowledge-gap records (in-memory, deterministic)

    asked_targets: list[str] = []        # cumulative genuine fact ids (held-out set)
    target_domain: dict[str, str] = {}
    retracted: set[str] = set()

    curve: list[dict] = []

    for i, ep in enumerate(episodes):
        # 1) CAGE IN THE LOOP — propose every unit; admit only committed.
        ep_pages: list[Page] = []
        for spec in ep.specs:
            verdict = cage.step(spec.proposal())
            decision = verdict.get("decision")
            if decision == "committed":
                committed_ids.append(spec.id)
                if spec.kind == "fact":
                    page = spec.page()
                    ep_pages.append(page)
                    if spec.id not in asked_targets:
                        asked_targets.append(spec.id)
                        target_domain[spec.id] = spec.domain
            else:
                rejected_ids.append(spec.id)
                if verdict.get("reason") == "poison":
                    poison_rejected.append(spec.id)
        admitted_pages[ep.id] = ep_pages

        gb.learn(ep_pages)
        gb.retract(ep.retract)
        retracted.update(ep.retract)
        state = gb.grounded_state()
        intentional |= gb.suppressed_ids()

        # Cage-breach audit: no rejected id may ever be present in the graph state.
        for rid in rejected_ids:
            if rid in state and rid not in cage_breaches:
                cage_breaches.append(rid)

        introduced = tuple(p.id for p in ep_pages if p.id in state)
        snapshots.append(Snapshot(task_id=ep.id, grounded=dict(state), introduced=introduced))
        if i == 0:
            baseline = ParametricBaseline(state.keys())   # freeze the weight model at t0

        # 2) HONEST METRIC — re-ask the FIXED, GROWING held-out query set.
        # The router's namespace is the wiki catalog of everything taught so far.
        vocab = build_vocab([CPQAEpisode(id="v", learn=tuple(gb._active()))])  # noqa: SLF001
        graph_correct = 0
        baseline_correct = 0
        n = len(asked_targets)
        for fid in asked_targets:
            if fid in retracted:
                # deliberately unlearned: expect abstain; correctness measured below
                q = Query(id=f"q_{fid}", target=fid, expect="abstain",
                          type="unlearning", text=f"recall {fid.replace('_', ' ')}")
            else:
                q = _query_for(fid)
            routed = substrate_router.route(q.text, vocab, gold=q.target)
            gb_asserted = routed is not None and routed in state
            bl_asserted = bool(baseline and routed is not None
                               and baseline.answer(routed) == "assert")
            gb_score = _score_routed(routed, q.target, q.expect, gb_asserted)
            bl_score = _score_routed(routed, q.target, q.expect, bl_asserted)
            if gb_score == "correct":
                graph_correct += 1
            if bl_score == "correct":
                baseline_correct += 1
            # competence record: only over assert-expected (live) facts.
            if q.expect == "assert":
                records.append({
                    "domain": target_domain.get(fid, "general"),
                    "confidence": 0.9 if gb_score == "correct" else 0.2,
                    "correct": gb_score == "correct",
                })
            # knowledge gap: an abstain/miss on a live fact is a gap to enrich.
            if q.expect == "assert" and gb_score in ("miss", "wrong"):
                policy = "abstain_no_route" if routed is None else "abstain_no_source"
                gaps.append({"query": q.text, "target": fid, "policy": policy})

        curve.append({
            "episode": ep.id,
            "graphCorrectCumulative": graph_correct,
            "baselineCorrectCumulative": baseline_correct,
            "graphAccuracyCumulative": round(graph_correct / n, 4) if n else 0.0,
            "baselineAccuracyCumulative": round(baseline_correct / n, 4) if n else 0.0,
            "unintendedForgettingSoFar": 0,   # filled after retention pass below
            "cageBreachesSoFar": 0,           # filled after the loop (breaches are 0)
        })

    # 3) RETENTION — separate catastrophic forgetting from deliberate unlearning.
    retention = build_report(snapshots)
    unintended = [d for d in retention["forgottenDetail"] if d["fact"] not in intentional]
    deliberate = [d for d in retention["forgottenDetail"] if d["fact"] in intentional]
    # Anything deliberately retracted that is no longer assertable also counts as
    # deliberate unlearning (retraction may complete after its introducing episode,
    # so it need not appear in forgottenDetail).
    deliberate_ids = {d["fact"] for d in deliberate}
    final_state = snapshots[-1].grounded if snapshots else {}
    deliberate_count = len(deliberate_ids | {r for r in retracted if r not in final_state})

    # Fill the per-episode breach/forgetting columns (both 0 for a clean run).
    for row in curve:
        row["unintendedForgettingSoFar"] = len(unintended)
        row["cageBreachesSoFar"] = len(cage_breaches)

    # COMPETENCE — measured-weakness "what to learn next" ranking.
    model = build_competence_model(records, alpha=0.1, coverage=0.5)
    priorities = learning_priorities(model)

    # KNOWLEDGE-GAP worklist (closing the loop: misses -> enrich next).
    worklist = gap_worklist(gaps)

    # CONTROL-FLOW GAP — LexicalController routing cost over the admitted capability.
    cpqa_episodes = _to_cpqa_episodes(episodes, admitted_pages)
    cf = control_flow_report(cpqa_episodes, controller)

    # SYMBOL IDENTITY — version_tag citations for every admitted fact.
    final_graph = build_graph(gb._active())  # noqa: SLF001
    citations = []
    for fid in asked_targets:
        tag = version_tag(final_graph, fid)
        if tag is not None:
            citations.append({"fact": fid, "versionTag": tag})

    final_graph_correct = curve[-1]["graphCorrectCumulative"] if curve else 0
    final_baseline_correct = curve[-1]["baselineCorrectCumulative"] if curve else 0
    accumulates = bool(
        curve
        and curve[-1]["graphCorrectCumulative"] > curve[0]["graphCorrectCumulative"]
        and final_graph_correct > final_baseline_correct
    )

    return {
        "schema": SCHEMA,
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "netCapabilityCurve": curve,
        "finalGraphCorrect": final_graph_correct,
        "finalBaselineCorrect": final_baseline_correct,
        "accumulates": accumulates,
        "unintendedForgetting": len(unintended),
        "deliberateUnlearning": deliberate_count,
        "cage": {
            "committed": len(committed_ids),
            "rejected": len(rejected_ids),
            "poisonRejected": len(poison_rejected),
            "breaches": len(cage_breaches),
            "committedIds": committed_ids,
            "rejectedIds": rejected_ids,
            "invariantsFinal": cage.check_invariants(),
            "killed": cage.killed,
        },
        "controlFlowGap": cf["controlFlowGap"],
        "controlFlow": {
            "controller": cf["controller"],
            "substrateAccuracy": cf["substrateAccuracy"],
            "endToEndAccuracy": cf["endToEndAccuracy"],
            "controlFlowGap": cf["controlFlowGap"],
        },
        "learningPriorities": priorities,
        "gapWorklist": worklist,
        "retentionMatrix": retention["retentionMatrix"],
        "citations": citations,
        "llmJudge": None,   # seam: LLM_JUDGE_HOOK is never called in the offline core
    }


def accumulates_cleanly(report: "dict[str, Any]") -> bool:
    """True iff the run is an honest, clean net-accumulation.

    Requires: the graph_backed cumulative-correct curve is monotonically
    non-decreasing MODULO deliberate retractions (a dip is allowed only when the
    cumulative-correct drops by no more than the number of deliberately-unlearned
    facts and recovers/holds thereafter); ``unintendedForgetting == 0``; cage
    ``breaches == 0``; and the final graph correct strictly exceeds the final
    (flat) parametric baseline.
    """
    curve = report.get("netCapabilityCurve", [])
    if not curve:
        return False
    if report.get("unintendedForgetting", 1) != 0:
        return False
    if report.get("cage", {}).get("breaches", 1) != 0:
        return False
    if report.get("finalGraphCorrect", 0) <= report.get("finalBaselineCorrect", 0):
        return False

    deliberate = int(report.get("deliberateUnlearning", 0))
    prev = curve[0]["graphCorrectCumulative"]
    allowed_dip = deliberate
    for row in curve[1:]:
        cur = row["graphCorrectCumulative"]
        if cur < prev:
            # A drop is only allowed if it is attributable to deliberate
            # unlearning (bounded by the count of deliberately-retracted facts).
            if prev - cur > allowed_dip:
                return False
        prev = cur
    return True
