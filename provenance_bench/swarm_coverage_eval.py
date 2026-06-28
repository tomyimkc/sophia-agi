# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Decomposition/coverage benchmark — the FAIR positive test for the swarm.

The source-discipline live eval found the swarm gives no benefit on *atomic* claims (a single
pass already nails them). The swarm's hypothesised value is on **decomposition-heavy** tasks:
broad questions where one pass misses aspects but complementary specialists, merged, cover
more. This benchmark measures exactly that — **aspect coverage**.

  * **solo arm**  — one "answer completely" pass.
  * **swarm arm** — three GENERIC complementary facet-agents (breadth · evidence · critique),
    each blind to the gold aspects, then a synthesis that merges them.

Each question carries a gold list of important aspects (the model never sees them). A judge
marks, per (answer, aspect), whether the aspect is covered. Coverage = fraction covered; the
swarm-minus-solo delta gets a paired bootstrap CI over all (question, aspect) pairs. A "win"
requires the CI to exclude zero — same no-overclaim discipline.

``model_fn(system, user) -> text`` and ``judge_fn(answer, aspect) -> 0|1`` are injected, so
it is deterministic + offline-testable (mock) and runnable live via tools/run_coverage_eval.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from provenance_bench.swarm_benchmark import _paired_bootstrap_ci

ModelFn = Callable[[str, str], str]
JudgeFn = Callable[[str, str], int]  # (answer, aspect) -> 1 if covered else 0

COVERAGE_SYS = ("You answer questions as completely as possible — cover all the important "
                "aspects, factors, evidence, and caveats. Be thorough but concise.")
SYNTH_SYS = ("You merge specialist notes into ONE complete, well-organised answer that covers "
             "every important aspect raised. Do not drop points; integrate them.")
FACETS = (
    ("breadth", "List the main points, factors, causes, and dimensions relevant to the question."),
    ("evidence", "Give concrete supporting evidence, examples, data, and mechanisms."),
    ("critique", "Note limitations, counterpoints, alternative views, and what is uncertain."),
)


def solo_answer(model_fn: ModelFn, question: str) -> str:
    return model_fn(COVERAGE_SYS, question)


def swarm_answer(model_fn: ModelFn, question: str) -> str:
    notes = []
    for name, lens in FACETS:
        out = model_fn(f"You are the '{name}' specialist. {lens}",
                       f"Question: {question}\n\nYour notes (concise):")
        if (out or "").strip():
            notes.append(f"[{name}] {out.strip()}")
    if not notes:
        return ""  # fail-closed: no facets → empty (scored as zero coverage, never invented)
    return model_fn(SYNTH_SYS, f"Question: {question}\n\nSpecialist notes:\n" + "\n".join(notes) + "\n\nMerged answer:")


@dataclass
class CoverageReport:
    subject: str
    n_questions: int
    n_pairs: int
    solo_coverage: float
    swarm_coverage: float
    delta: float
    ci95: "tuple[float, float]"
    calls: int = 0
    per_question: list = field(default_factory=list)

    @property
    def is_win(self) -> bool:
        return self.ci95[0] > 0.0 and self.delta > 0

    def to_dict(self) -> dict:
        return {
            "subject": self.subject, "nQuestions": self.n_questions, "nPairs": self.n_pairs,
            "soloCoverage": round(self.solo_coverage, 3), "swarmCoverage": round(self.swarm_coverage, 3),
            "delta": round(self.delta, 3), "ci95": [round(self.ci95[0], 3), round(self.ci95[1], 3)],
            "ciExcludesZero": self.is_win, "calls": self.calls,
            "verdict": "swarm_wins_on_coverage" if self.is_win else "no_coverage_advantage",
            "note": "win = paired CI on (swarm-solo) aspect coverage over all (question,aspect) pairs excludes zero.",
        }


@dataclass
class CoverageTask:
    question: str
    aspects: "tuple[str, ...]"


def run_coverage(tasks: "list[CoverageTask]", model_fn: ModelFn, judge_fn: JudgeFn,
                 *, subject: str = "subject") -> CoverageReport:
    calls = {"n": 0}

    def counted(system: str, user: str) -> str:
        calls["n"] += 1
        return model_fn(system, user)

    solo_b: list[int] = []
    swarm_b: list[int] = []
    per_q = []
    for t in tasks:
        sa = solo_answer(counted, t.question)
        wa = swarm_answer(counted, t.question)
        s_hits = [judge_fn(sa, a) for a in t.aspects]
        w_hits = [judge_fn(wa, a) for a in t.aspects]
        solo_b += s_hits
        swarm_b += w_hits
        per_q.append({"q": t.question[:60], "soloCov": round(sum(s_hits) / len(s_hits), 2),
                      "swarmCov": round(sum(w_hits) / len(w_hits), 2), "aspects": len(t.aspects)})
    n = len(solo_b)
    sc = sum(solo_b) / n if n else 0.0
    wc = sum(swarm_b) / n if n else 0.0
    lo, hi = _paired_bootstrap_ci(solo_b, swarm_b)
    return CoverageReport(subject, len(tasks), n, sc, wc, wc - sc, (lo, hi), calls["n"], per_q)


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    tasks = [CoverageTask(f"Question {i} about a broad topic", (f"a{i}", f"b{i}", f"c{i}", f"d{i}"))
             for i in range(10)]
    judge = lambda answer, aspect: 1 if aspect in answer else 0  # keyword coverage

    # Mock where the swarm covers MORE aspects (each facet contributes some; synthesis unions).
    def mock(helps: bool):
        def fn(system: str, user: str) -> str:
            # extract the question index from the user text
            qi = next((tok for tok in user.split() if tok.isdigit()), "0")
            if system.startswith(SYNTH_SYS[:15]):
                return f"a{qi} b{qi} c{qi} d{qi}" if helps else f"a{qi}"        # swarm synth: all 4 / 1
            if "specialist" in system:
                return f"b{qi} c{qi}"                                            # facets contribute extras
            return f"a{qi}"                                                      # solo: 1 aspect
        return fn

    rep = run_coverage(tasks, mock(True), judge, subject="mock")
    checks["swarm_beats_solo"] = rep.is_win and rep.swarm_coverage > rep.solo_coverage
    checks["calls_counted"] = rep.calls > rep.n_questions

    rep_null = run_coverage(tasks, mock(False), judge, subject="mock")
    checks["no_false_win"] = not rep_null.is_win

    # Fail-closed: empty facet notes → empty merged answer → zero coverage (never invented).
    empty = swarm_answer(lambda s, u: "" if "specialist" in s else "x", "Q")
    checks["failclosed_empty"] = empty == ""

    checks["deterministic"] = run_coverage(tasks, mock(True), judge, subject="mock").to_dict() == rep.to_dict()

    ok = all(checks.values())
    return ok, {"checks": checks, "soloCov": round(rep.solo_coverage, 2), "swarmCov": round(rep.swarm_coverage, 2)}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Swarm coverage-eval offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("  mock solo/swarm coverage:", detail.get("soloCov"), "/", detail.get("swarmCov"))
    raise SystemExit(0 if ok else 1)
