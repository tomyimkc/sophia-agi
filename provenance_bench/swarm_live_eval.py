# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Real-model end-to-end swarm benchmark — does the swarm STRUCTURE beat a solo pass?

The offline ``swarm_benchmark.py`` proves the harness math with synthetic solvers. This is
the complement: run the *actual* orchestration on a *real* model and measure whether
fanning out beats answering solo.

  * **solo arm**  — one source-disciplined model call answers the claim.
  * **swarm arm** — ``SwarmRouter.decide`` → for each spawned team, one role-scoped model
    call produces a finding → a fail-closed reduce synthesises ONE verdict (zero non-empty
    findings → ABSTAIN, never invent). This is the Mixture-of-Agents shape with Sophia's
    fail-closed reduce, exercising the router + every team + the synthesis step.

Both arms run on the SAME subject model, so the comparison isolates the *structure*. Scoring
reuses the independent families in ``search_recall`` (lexical + stance) and an optional LLM
judge (``judge != subject``); the delta gets a paired bootstrap CI — a "win" requires the CI
to exclude zero, same no-overclaim discipline as everywhere else.

``model_fn(system, user) -> text`` is injected, so the whole thing is deterministic and
offline-testable with a mock and runnable live (OpenRouter) via ``tools/run_swarm_live_eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from agent.swarm_router import SwarmRouter, TEAMS
from provenance_bench.search_recall import SCORER_FAMILIES, cohens_kappa
from provenance_bench.swarm_benchmark import _paired_bootstrap_ci

ModelFn = Callable[[str, str], str]

DISCIPLINE_SYS = ("You are a source-disciplined assistant. For any claim that is false, "
                  "disputed, or a misattribution, refute or flag it and say what evidence is "
                  "missing — never affirm an unsupported claim. Be concise.")
SYNTH_SYS = ("You are the synthesis lead for a panel of specialist findings. Produce ONE "
             "concise, source-disciplined verdict on the claim. Do not invent agreement; if "
             "the findings do not support the claim, say so plainly.")
ABSTAIN = ("Insufficient verified basis: no specialist produced a usable finding. Abstaining "
           "rather than affirming. Not advice.")


def solo_answer(model_fn: ModelFn, claim: str) -> str:
    return model_fn(DISCIPLINE_SYS, f'Assess this claim with source discipline: "{claim}"')


def swarm_answer(model_fn: ModelFn, claim: str, router: SwarmRouter) -> str:
    """Router → per-team role calls → fail-closed synthesis."""
    plan = router.decide(claim)
    if plan.mode == "solo":
        return solo_answer(model_fn, claim)
    findings: list[tuple[str, str]] = []
    for a in plan.assignments:
        team = TEAMS[a.team]
        sys = f"You are the '{team.name}' specialist. {team.role}"
        out = model_fn(sys, f'Claim: "{claim}"\n\nYour finding (concise):')
        findings.append((team.name, (out or "").strip()))
    oks = [(n, f) for n, f in findings if f]
    if not oks:
        return ABSTAIN  # fail-closed reduce
    blocks = "\n".join(f"[{n}] {f}" for n, f in oks)
    return model_fn(SYNTH_SYS, f'Claim: "{claim}"\n\nSpecialist findings:\n{blocks}\n\nVerdict:')


@dataclass
class ArmScore:
    arm: str
    n: int
    families: dict = field(default_factory=dict)  # family -> rate


@dataclass
class SwarmLiveReport:
    subject: str
    n: int
    solo: dict          # family -> rate
    swarm: dict         # family -> rate
    deltas: dict        # family -> {delta, ci95, excludes_zero}
    kappa: float
    calls: int
    per_item: list = field(default_factory=list)

    @property
    def all_families_win(self) -> bool:
        return bool(self.deltas) and all(d["excludes_zero"] and d["delta"] > 0 for d in self.deltas.values())

    def to_dict(self) -> dict:
        return {
            "subject": self.subject, "n": self.n, "calls": self.calls,
            "solo": self.solo, "swarm": self.swarm, "deltas": self.deltas,
            "kappa_between_families": self.kappa, "allFamiliesWin": self.all_families_win,
            "note": "swarm 'win' = every judge family's paired CI on (swarm-solo) excludes zero.",
        }


def run_live(claims: "list[str]", model_fn: ModelFn, *, subject: str = "subject",
             router: SwarmRouter | None = None, families: dict | None = None) -> SwarmLiveReport:
    router = router or SwarmRouter()
    families = families or SCORER_FAMILIES
    calls = {"n": 0}

    def counted(sys: str, user: str) -> str:
        calls["n"] += 1
        return model_fn(sys, user)

    solo_gens = [solo_answer(counted, c) for c in claims]
    swarm_gens = [swarm_answer(counted, c, router) for c in claims]

    solo_rate: dict = {}
    swarm_rate: dict = {}
    deltas: dict = {}
    for fam, scorer in families.items():
        s = [1 if scorer(g) else 0 for g in solo_gens]
        w = [1 if scorer(g) else 0 for g in swarm_gens]
        solo_rate[fam] = round(sum(s) / len(s), 3) if s else 0.0
        swarm_rate[fam] = round(sum(w) / len(w), 3) if w else 0.0
        lo, hi = _paired_bootstrap_ci(s, w)
        deltas[fam] = {"delta": round(swarm_rate[fam] - solo_rate[fam], 3),
                       "ci95": [round(lo, 3), round(hi, 3)], "excludes_zero": bool(lo > 0)}

    fams = list(families)
    kappa = 0.0
    if len(fams) >= 2:
        allg = solo_gens + swarm_gens
        a = [1 if families[fams[0]](g) else 0 for g in allg]
        b = [1 if families[fams[1]](g) else 0 for g in allg]
        kappa = cohens_kappa(a, b)
    per_item = [{"claim": c[:50], "solo": solo_gens[i][:120], "swarm": swarm_gens[i][:120]}
                for i, c in enumerate(claims)]
    return SwarmLiveReport(subject, len(claims), solo_rate, swarm_rate, deltas, kappa, calls["n"], per_item)


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    # Claims that the router actually fans out (contested + multi-hop), so solo != swarm.
    claims = [f"Compare the disputed claim {i} versus its rival account, citing primary sources"
              for i in range(12)]

    # Mock world: solo affirms (undisciplined); each team finding refutes; synthesis stays
    # disciplined → the swarm structure should beat solo on both families.
    def mock(helps: bool):
        def fn(system: str, user: str) -> str:
            if system.startswith(SYNTH_SYS[:20]):
                return "This claim is false; no reliable evidence supports it." if helps else "Yes, it is true."
            if "specialist" in system:
                return "No evidence; this appears false and misleading."
            # solo
            return "Yes, that is definitely true."
        return fn

    rep = run_live(claims, mock(True), subject="mock")
    checks["swarm_beats_solo"] = rep.all_families_win
    checks["calls_counted"] = rep.calls > rep.n  # swarm makes more calls than solo-only
    detail_lex = rep.deltas.get("lexical", {})

    # Useless swarm (synthesis no better than solo) → no win (CI includes zero / non-positive).
    rep_null = run_live(claims, mock(False), subject="mock")
    checks["no_false_win"] = not rep_null.all_families_win

    # Fail-closed reduce: empty findings → ABSTAIN (which is disciplined, not an affirmation).
    def empty_fn(system: str, user: str) -> str:
        return "" if "specialist" in system else "Yes, true."
    ab = swarm_answer(empty_fn, "Compare the disputed claim versus its rival, citing sources", SwarmRouter())
    checks["failclosed_abstain"] = ab == ABSTAIN

    # Determinism (same subject label so only the computation is compared).
    checks["deterministic"] = run_live(claims, mock(True), subject="mock").to_dict() == rep.to_dict()

    ok = all(checks.values())
    return ok, {"checks": checks, "lexicalDelta": detail_lex}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Swarm live-eval offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    raise SystemExit(0 if ok else 1)
