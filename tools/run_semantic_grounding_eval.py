#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Three-arm semantic-grounding ablation runner (I7).

Arms (all share ONE base model in a real run):
  A0  base only                      — pure distributional guessing
  A1  base + OKF definition retrieval — provenance-carrying grounding context
  A2  base + retrieval + symbolic     — datalog compose/abstain tool

It generates per-arm completions over the sealed D1+D2 datasets, scores each with
``eval.semantic_grounding.score``, and reports paired deltas (A1-A0, A2-A0) with
bootstrap 95% CIs and the MDE at N — the inputs ``tools/claim_gate.py`` later turns
into a GO/NO-GO.

Two modes:
  --mock    deterministic, offline, NO model. A PLUMBING SELF-TEST that exercises the
            metric/CI path end to end. The deltas are illustrative, NOT a capability
            measurement (the report is stamped mock:true, candidateOnly:true).
  --model   real run on the model farm (e.g. 'ollama:qwen2.5:7b-instruct@http://host:11434/v1').
            Not runnable in an air-gapped CI box; this is the dispatch path.

    python tools/run_semantic_grounding_eval.py --mock
    python tools/run_semantic_grounding_eval.py --model <spec> --seeds 3 --write

Exit 0 on a completed run (verdicts are in the JSON); nonzero only on usage/internal error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.semantic_grounding.score import (  # noqa: E402
    aggregate, reference_verdict, score_case)
from tools.eval_stats import bootstrap_ci_paired, mde_at_n  # noqa: E402

DATA = ROOT / "eval" / "semantic_grounding" / "data"
OUT_DIR = ROOT / "agi-proof" / "benchmark-results" / "semantic_grounding"
ARMS = ("A0", "A1", "A2")


def _load(name: str) -> list[dict]:
    return [json.loads(ln) for ln in (DATA / name).read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_cases(fold: str = "all") -> list[dict]:
    cases = _load("d1_definition_faithfulness.jsonl") + _load("d2_compositional_derivation.jsonl")
    if fold != "all":
        cases = [c for c in cases if c.get("fold") == fold]
    return cases


# ----------------------------------------------------------------- offline mock
_CONCEPT_GLOSS: dict[str, str] | None = None


def _okf_gloss_for_term(term: str) -> str | None:
    """Honest 'use the dictionary' grounding: look the term up in the OKF and return
    its canonical gloss (by canonicalTitleEn), without touching the gold id."""
    global _CONCEPT_GLOSS
    if _CONCEPT_GLOSS is None:
        from eval.semantic_grounding.build_dataset import CONCEPT_DIR, _gloss
        from okf.page import load_pages
        _CONCEPT_GLOSS = {
            str(p.meta.get("canonicalTitleEn") or "").lower(): _gloss(p.body)
            for p in load_pages(str(CONCEPT_DIR)) if p.page_type == "concept"
        }
    return _CONCEPT_GLOSS.get(term.lower())


def _grounded_pick(case: dict) -> str:
    """Select the candidate whose gloss matches the term's retrieved OKF definition."""
    from agent.lexical_embed import rank
    query = _okf_gloss_for_term(case["term"]) or case["term"]
    docs = [(c["conceptId"], c["gloss"]) for c in case.get("candidates", [])]
    ranked = rank(query, docs, top_k=1) if docs else []
    return ranked[0][0] if ranked else case["candidates"][0]["conceptId"]


def _blind_pick(case: dict) -> str:
    """Ungrounded guess: a content-free, gold-independent choice over the candidates."""
    import hashlib
    cands = [c["conceptId"] for c in case["candidates"]]
    h = int(hashlib.md5(case["id"].encode()).hexdigest(), 16)
    return cands[h % len(cands)]


def mock_completion(arm: str, case: dict) -> dict:
    """Deterministic, offline stand-in completions that exercise the contrast the
    bench measures. NOT a model; see the module docstring."""
    cid = case["id"]
    if case["task"] == "definition":
        forbidden = case.get("doNotAttributeTo") or []
        merge = f"{case['term']} was coined by {forbidden[0]}." if forbidden else ""
        if arm == "A0":
            # Ungrounded: blind guess + confidently merges a forbidden lineage.
            prose = merge or f"{case['term']} is a concept."
            return {"id": cid, "selected": _blind_pick(case), "completion": prose}
        if arm == "A1":
            # Retrieval grounds the SENSE, but still asserts the forbidden author.
            prose = merge or f"{case['term']} is defined by its retrieved gloss."
            return {"id": cid, "selected": _grounded_pick(case), "completion": prose}
        # A2: retrieval + provenance discipline -> right sense AND faithful.
        note = f" It should not be attributed to {', '.join(forbidden)}." if forbidden else ""
        return {"id": cid, "selected": _grounded_pick(case),
                "completion": f"{case['term']} matches its retrieved gloss.{note}"}

    # D2 composition
    if arm in ("A0", "A1"):
        # No symbolic reasoning: guess the modal verdict.
        return {"id": cid, "verdict": "entailed", "completion": "entailed (guessed)."}
    # A2: call the symbolic engine (tool-use; near-ceiling by construction).
    v = reference_verdict(case["axioms"], case["claim"])
    return {"id": cid, "verdict": v, "completion": f"{v} by closed-world derivation."}


# ----------------------------------------------------------------- real model
def build_prompt(arm: str, case: dict) -> str:
    """Prompt for a real model run. A1 adds retrieved definition context; A2 adds the
    symbolic tool's result (for D2) / provenance constraint (for D1)."""
    base = case["prompt"]
    if arm == "A0":
        return base
    if case["task"] == "definition":
        ctx = ""
        if arm == "A2" and case.get("doNotAttributeTo"):
            ctx = f"\n\n[Provenance constraint] Do NOT attribute to: {', '.join(case['doNotAttributeTo'])}."
        return base + "\n\n[Retrieved OKF definitions are the candidates above.]" + ctx
    # D2
    if arm == "A2":
        v = reference_verdict(case["axioms"], case["claim"])
        return base + f"\n\n[Symbolic checker] A sound closed-world derivation returns: {v}. Use it."
    return base + "\n\n[Reason step by step over only the listed facts.]"


def model_completion(arm: str, case: dict, client) -> dict:
    """Generate one completion from a real model client (agent.model.ModelClient-like)."""
    prompt = build_prompt(arm, case)
    text = client.complete(prompt) if hasattr(client, "complete") else client.generate(prompt)
    return {"id": case["id"], "completion": text}


# ----------------------------------------------------------------- arm scoring
def _primary_flag(score: dict) -> float:
    """Per-case primary correctness used for paired deltas."""
    if score["task"] == "definition":
        return 1.0 if score["d1_grounded"] else 0.0
    return 1.0 if score["d2_correct"] else 0.0


def score_arm(cases: list[dict], comps: dict[str, dict]) -> dict:
    scores = [score_case(c, comps[c["id"]]) for c in cases]
    return {
        "report": aggregate(scores),
        "primary": {s["id"]: _primary_flag(s) for s in scores},
    }


def paired_delta(base: dict[str, float], arm: dict[str, float]) -> dict:
    ids = sorted(base)
    diffs = [arm[i] - base[i] for i in ids]
    mean = sum(diffs) / len(diffs) if diffs else 0.0
    lo, hi = bootstrap_ci_paired(diffs) if diffs else (0.0, 0.0)
    return {
        "n": len(diffs),
        "delta": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "mde_at_n": round(mde_at_n(len(diffs)), 4) if diffs else None,
        "powered": bool(diffs and mde_at_n(len(diffs)) <= 0.10),
    }


def run(*, mock: bool, model: str | None, seeds: int, fold: str = "all") -> dict:
    cases = load_cases(fold)
    client = None
    if not mock:
        from agent.model import ModelClient  # lazy: only needed for a real run
        client = ModelClient(model)

    arms_out: dict[str, dict] = {}
    for arm in ARMS:
        if mock:
            comps = {c["id"]: mock_completion(arm, c) for c in cases}
        else:
            comps = {c["id"]: model_completion(arm, c, client) for c in cases}
        arms_out[arm] = score_arm(cases, comps)

    base = arms_out["A0"]["primary"]
    deltas = {
        "A1_minus_A0": paired_delta(base, arms_out["A1"]["primary"]),
        "A2_minus_A0": paired_delta(base, arms_out["A2"]["primary"]),
        "A2_minus_A1": paired_delta(arms_out["A1"]["primary"], arms_out["A2"]["primary"]),
    }
    return {
        "experimentId": "semantic-grounding-3arm",
        "mock": mock,
        "model": model,
        "seeds": seeds,
        "fold": fold,
        "candidateOnly": True,
        "canClaimAGI": False,
        "note": ("MOCK plumbing self-test — illustrative deltas, NOT a capability measurement"
                 if mock else "real model-farm run"),
        "nCases": len(cases),
        "arms": {a: arms_out[a]["report"] for a in ARMS},
        "deltas": deltas,
    }


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="3-arm semantic-grounding ablation (I7).")
    ap.add_argument("--mock", action="store_true", help="offline deterministic plumbing self-test")
    ap.add_argument("--model", default=None, help="model spec for a real farm run")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--fold", choices=("all", "train", "eval"), default="all",
                    help="restrict eval cases to a fold (use 'eval' to measure a Phase-2 uplift)")
    ap.add_argument("--write", action="store_true", help="write report under agi-proof/benchmark-results/")
    args = ap.parse_args(argv)

    if not args.mock and not args.model:
        ap.error("pass --mock (offline) or --model <spec> (real run)")

    out = run(mock=args.mock, model=args.model, seeds=args.seeds, fold=args.fold)
    print(json.dumps(out, indent=2))

    if args.mock:
        # Seam assertions: grounding must help D1, the symbolic tool must help D2.
        a1 = out["arms"]["A1"]["D1_sense_accuracy"]["rate"]
        a0 = out["arms"]["A0"]["D1_sense_accuracy"]["rate"]
        a2f = out["arms"]["A2"]["D1_faithfulness"]["rate"]
        a0f = out["arms"]["A0"]["D1_faithfulness"]["rate"]
        a2d = out["arms"]["A2"]["D2_accuracy"]["rate"]
        a0d = out["arms"]["A0"]["D2_accuracy"]["rate"]
        assert a1 >= a0, "A1 retrieval should not hurt D1 sense"
        assert a2f > a0f, "A2 provenance discipline should lift D1 faithfulness"
        assert a2d > a0d, "A2 symbolic tool should lift D2 accuracy"
        print("\nmock self-test OK: A1>=A0 (D1 sense), A2>A0 (D1 faithful, D2)", file=sys.stderr)

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = "mock" if args.mock else (args.model or "model").replace("/", "_").replace(":", "_")
        path = OUT_DIR / f"3arm-{tag}.json"
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
