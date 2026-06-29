# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Semantic-grounding benchmark scorer (Phase 0) — deterministic, no LLM judge.

Two task families, each scored by an audited, model-free check:

  D1  definition-faithfulness   — did the answer pick the OKF gloss that actually
                                  defines the term (word-sense), AND avoid asserting
                                  a forbidden attribution? Reuses
                                  ``agent.lexical_embed`` (offline sense match) and
                                  ``agent.verifiers.provenance_faithful``.
  D2  compositional derivation  — given a closed world of concept-TBox axioms
                                  (``subClassOf`` / ``disjointWith``), is a claim
                                  *entailed*, a *violation*, or *abstain* (the world
                                  is silent)? The reference verdict is a least-fixed
                                  point in ``agent.datalog_engine`` — the symbolic
                                  half of the neuro-symbolic split — so the gold is a
                                  derivable theorem, not a hand label.

Inputs are two JSON/JSONL files: sealed *cases* and model *completions* keyed by id.

Usage:
    python -m eval.semantic_grounding.score \
        --cases eval/semantic_grounding/data/d1_definition_faithfulness.jsonl \
        --completions runs/arm.jsonl
    python -m eval.semantic_grounding.score --self-test   # CI fixture, no model
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Allow `python eval/semantic_grounding/score.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.lexical_embed import rank  # noqa: E402
from agent.verifiers import provenance_faithful  # noqa: E402
from tools.eval_stats import bootstrap_ci_paired  # noqa: E402

VERDICTS = ("entailed", "violation", "abstain")
RELATIONS = ("subClassOf", "disjointWith")


# --------------------------------------------------------------------------- IO
def read_records(path: str | Path) -> list[dict]:
    """Read either a JSON list / {cases|completions:[...]} or a JSONL file."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(obj, dict):
        return obj.get("cases", obj.get("completions", [obj]))
    if isinstance(obj, list):
        return obj
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_cases(path: str | Path) -> dict[str, dict]:
    return {c["id"]: c for c in read_records(path)}


def load_completions(path: str | Path) -> dict[str, dict]:
    return {c["id"]: c for c in read_records(path)}


# ----------------------------------------------- D2 symbolic reference reasoner
def reference_verdict(axioms: list, claim: list) -> str:
    """Compute the closed-world verdict for ``claim`` under ``axioms`` via Datalog.

    ``axioms`` is a list of ``[relation, x, y]`` (relation in :data:`RELATIONS`);
    ``claim`` is a single ``[relation, x, y]``. The verdict is one of
    :data:`VERDICTS`:

      * ``entailed``  — derivable in the least fixed point (transitive subsumption,
                        symmetric disjointness);
      * ``violation`` — asserting it contradicts a derivable disjointness
                        (X is known to be a C, and Y is disjoint from C), or claims
                        disjointness where one class subsumes the other;
      * ``abstain``   — the closed world neither entails nor refutes it.

    This is a *grounding* verdict over the axioms we wrote down, NOT a truth oracle
    (see docs/11-Platform/Ontology-Claim-Boundary.md).
    """
    from agent.datalog_engine import A, Program, Rule, Var

    prog = Program()
    for rel, x, y in axioms:
        if rel == "subClassOf":
            prog.fact(A("sub", x, y))
        elif rel == "disjointWith":
            prog.fact(A("disj", x, y))
            prog.fact(A("disj", y, x))  # disjointness is symmetric
    # Transitive closure of subsumption: the "grammar" that composes meaning.
    prog.rule(Rule.clause(A("sub", Var("X"), Var("Z")),
                          A("sub", Var("X"), Var("Y")), A("sub", Var("Y"), Var("Z"))))
    sol = prog.solve()
    sub = {a.args for a in sol if a.pred == "sub"}
    disj = {a.args for a in sol if a.pred == "disj"}

    rel, x, y = claim
    if rel == "subClassOf":
        if (x, y) in sub:
            return "entailed"
        if (x, y) in disj:
            return "violation"
        # X is a sub of some C, and Y is disjoint from C -> X cannot be a Y.
        for (xx, c) in sub:
            if xx == x and ((y, c) in disj):
                return "violation"
        return "abstain"
    if rel == "disjointWith":
        if (x, y) in disj:
            return "entailed"
        if (x, y) in sub or (y, x) in sub:
            return "violation"
        return "abstain"
    return "abstain"


_VERDICT_PAT = (
    ("violation", re.compile(r"\b(violat|contradict|incompatib|cannot\s+be\s+both)", re.I)),
    ("abstain", re.compile(r"\b(abstain|unknown|insufficient|not\s+entailed|cannot\s+be\s+derived|undetermined)", re.I)),
    ("entailed", re.compile(r"\b(entail|derivable|follows|yes)", re.I)),
)


def parse_verdict(completion: dict) -> str | None:
    """Extract a D2 verdict from a completion's structured field or its prose."""
    v = completion.get("verdict")
    if isinstance(v, str) and v.lower() in VERDICTS:
        return v.lower()
    text = completion.get("completion", "") or ""
    for label, pat in _VERDICT_PAT:  # priority: violation > abstain > entailed
        if pat.search(text):
            return label
    return None


# ------------------------------------------------------------------ D1 scoring
def _selected_concept(case: dict, completion: dict) -> str | None:
    """Which candidate definition did the answer choose?

    Prefer an explicit structured ``selected`` conceptId; else fall back to the
    offline lexical match between the answer prose and each candidate gloss
    (``agent.lexical_embed`` — deterministic, dependency-free).
    """
    sel = completion.get("selected")
    if sel:
        return str(sel)
    text = completion.get("completion", "") or ""
    cands = case.get("candidates") or []
    docs = [(c["conceptId"], c["gloss"]) for c in cands]
    if not docs or not text.strip():
        return None
    ranked = rank(text, docs, top_k=1)
    return ranked[0][0] if ranked else None


def score_case_d1(case: dict, completion: dict) -> dict:
    gold = case["goldConceptId"]
    selected = _selected_concept(case, completion)
    sense_correct = selected == gold

    forbidden = case.get("doNotAttributeTo") or []
    work = case.get("term") or case.get("canonicalTitleEn")
    if forbidden and work:
        records = {case["id"]: {"canonicalTitleEn": work, "doNotAttributeTo": list(forbidden)}}
        prov = provenance_faithful(records)(completion.get("completion", ""), case, {})
        faithful: bool | None = bool(prov["passed"])
    else:
        faithful = None

    return {
        "id": case["id"],
        "task": "definition",
        "selected": selected,
        "d1_sense_correct": sense_correct,
        "d1_faithful": faithful,
        # Grounded = right sense AND not unfaithful (None faithful = nothing to break).
        "d1_grounded": bool(sense_correct and faithful is not False),
    }


def score_case_d2(case: dict, completion: dict) -> dict:
    gold = case["goldVerdict"]
    ref = reference_verdict(case["axioms"], case["claim"])
    answer = parse_verdict(completion)
    return {
        "id": case["id"],
        "task": "composition",
        "goldVerdict": gold,
        "answer": answer,
        # Drift guard: the committed gold must still match the engine's derivation.
        "d2_dataset_valid": ref == gold,
        "d2_correct": answer == gold,
    }


def score_case(case: dict, completion: dict) -> dict:
    task = case.get("task")
    if task == "definition":
        return score_case_d1(case, completion)
    if task == "composition":
        return score_case_d2(case, completion)
    raise ValueError(f"case {case.get('id')!r}: unknown task {task!r}")


# ------------------------------------------------------------------- aggregation
def _rate_ci(flags: list) -> dict | None:
    vals = [1.0 if f else 0.0 for f in flags if f is not None]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    lo, hi = bootstrap_ci_paired(vals)
    return {"rate": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals)}


def aggregate(scores: list[dict]) -> dict:
    d1 = [s for s in scores if s["task"] == "definition"]
    d2 = [s for s in scores if s["task"] == "composition"]

    d2_by_verdict: dict[str, list] = {v: [] for v in VERDICTS}
    for s in d2:
        d2_by_verdict[s["goldVerdict"]].append(s["d2_correct"])

    report: dict[str, Any] = {
        "n": len(scores),
        "nD1": len(d1),
        "nD2": len(d2),
        "D1_sense_accuracy": _rate_ci([s["d1_sense_correct"] for s in d1]),
        "D1_faithfulness": _rate_ci([s["d1_faithful"] for s in d1]),
        "D1_grounded_composite": _rate_ci([s["d1_grounded"] for s in d1]),
        "D2_accuracy": _rate_ci([s["d2_correct"] for s in d2]),
        "D2_accuracy_by_gold": {v: _rate_ci(d2_by_verdict[v]) for v in VERDICTS},
        "D2_dataset_valid": all(s["d2_dataset_valid"] for s in d2) if d2 else None,
    }
    return report


def run(cases_path: str | Path, completions_path: str | Path) -> dict:
    cases = load_cases(cases_path)
    comps = load_completions(completions_path)
    missing = set(cases) - set(comps)
    if missing:
        raise ValueError(f"completions missing for {len(missing)} case(s): {sorted(missing)[:5]}")
    scores = [score_case(cases[cid], comps[cid]) for cid in cases]
    return {"perCase": scores, "report": aggregate(scores)}


# ------------------------------------------------------------------- self-test
def self_test() -> dict:
    """Model-free fixture so CI verifies the seam: a grounded/composing agent must
    dominate an ungrounded/guessing one on D1 sense, D1 faithfulness and D2."""
    cases = {
        "d1-a": {
            "id": "d1-a", "task": "definition", "term": "cognitive dissonance",
            "goldConceptId": "cognitive_dissonance", "doNotAttributeTo": ["sigmund_freud"],
            "candidates": [
                {"conceptId": "cognitive_dissonance",
                 "gloss": "Festinger 1957 discomfort from holding conflicting cognitions"},
                {"conceptId": "confirmation_bias",
                 "gloss": "Wason 1960 tendency to seek evidence confirming a belief"},
            ],
        },
        "d2-entail": {"id": "d2-entail", "task": "composition",
                      "axioms": [["subClassOf", "dog", "mammal"], ["subClassOf", "mammal", "animal"]],
                      "claim": ["subClassOf", "dog", "animal"], "goldVerdict": "entailed"},
        "d2-viol": {"id": "d2-viol", "task": "composition",
                    "axioms": [["subClassOf", "dog", "mammal"], ["disjointWith", "mammal", "bird"]],
                    "claim": ["subClassOf", "dog", "bird"], "goldVerdict": "violation"},
        "d2-abstain": {"id": "d2-abstain", "task": "composition",
                       "axioms": [["subClassOf", "dog", "mammal"]],
                       "claim": ["subClassOf", "dog", "reptile"], "goldVerdict": "abstain"},
    }
    grounded = {
        "d1-a": {"id": "d1-a", "selected": "cognitive_dissonance",
                 "completion": "Cognitive dissonance is Festinger's 1957 idea of conflicting cognitions; it should not be attributed to Freud."},
        "d2-entail": {"id": "d2-entail", "verdict": "entailed"},
        "d2-viol": {"id": "d2-viol", "verdict": "violation"},
        "d2-abstain": {"id": "d2-abstain", "verdict": "abstain"},
    }
    ungrounded = {
        "d1-a": {"id": "d1-a", "selected": "confirmation_bias",
                 "completion": "Cognitive dissonance was discovered by Sigmund Freud."},
        "d2-entail": {"id": "d2-entail", "verdict": "abstain"},
        "d2-viol": {"id": "d2-viol", "verdict": "entailed"},
        "d2-abstain": {"id": "d2-abstain", "verdict": "entailed"},
    }
    g = aggregate([score_case(cases[c], grounded[c]) for c in cases])
    u = aggregate([score_case(cases[c], ungrounded[c]) for c in cases])
    return {"grounded": g, "ungrounded": u}


def _main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Semantic-grounding benchmark scorer (Phase 0).")
    p.add_argument("--cases")
    p.add_argument("--completions")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        out = self_test()
        print(json.dumps(out, indent=2))
        g, u = out["grounded"], out["ungrounded"]
        assert g["D1_sense_accuracy"]["rate"] > u["D1_sense_accuracy"]["rate"]
        assert g["D1_faithfulness"]["rate"] > u["D1_faithfulness"]["rate"]
        assert g["D2_accuracy"]["rate"] > u["D2_accuracy"]["rate"]
        assert g["D2_dataset_valid"] is True
        print("\nself-test OK: grounded agent dominates ungrounded on D1 sense/faithfulness + D2",
              file=sys.stderr)
        return 0

    if not (args.cases and args.completions):
        p.error("provide --cases and --completions, or --self-test")
    out = run(args.cases, args.completions)
    print(json.dumps(out["report"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
