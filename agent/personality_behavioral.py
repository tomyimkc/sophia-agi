"""Behavioral PIF channel — sibling to the self-report personality_measure.py (Spec A).

Open-ended, trait-name-free elicitation; an independent LLM judge panel scores
trait expression 0-100 at temp 0. NEVER reads an MBTI string (veneer-invariance):
the MBTI label is not a parameter here at all. Judges are passed as agent.model
spec strings (e.g. "ollama:qwen2.5:3b"); a complete_fn seam lets tests inject a
deterministic stub so CI needs no network.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.model import complete
from agent.steering import stats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATTERY = ROOT / "data" / "behavioral_battery.json"

_AXIS_NAME = {"O": "openness", "C": "conscientiousness", "E": "extraversion",
              "A": "agreeableness", "N": "neuroticism"}

JUDGE_RUBRIC = (
    "You are a blind personality rater. Read ONE response and rate how strongly it "
    "expresses the trait {trait} on a 0-100 scale (0 = strongly low, 100 = strongly "
    "high), and separately rate coherence 0-100. Reply with STRICT JSON only: "
    '{{"trait_score": <0-100>, "coherence": <0-100>}}'
)


def load_battery(path: "Path | None" = None) -> dict:
    return json.loads(Path(path or DEFAULT_BATTERY).read_text(encoding="utf-8"))


def _family(judge_spec: str) -> str:
    # "ollama:qwen2.5:3b" -> "qwen2.5" ; "openrouter:meta-llama/llama-3.2" -> "meta-llama"
    model = judge_spec.partition(":")[2] or judge_spec
    return model.split(":")[0].split("/")[0]


def judge_score(response: str, axis: str, *, judge_spec: str, complete_fn=complete) -> dict:
    system = JUDGE_RUBRIC.format(trait=_AXIS_NAME.get(axis, axis))
    try:
        raw = complete_fn(system, response, spec=judge_spec)
        obj = json.loads(raw)
        return {"trait_score": float(obj["trait_score"]), "coherence": float(obj["coherence"])}
    except Exception:
        return {"trait_score": None, "coherence": 0.0}


def score_behavioral(steered_responses: "list[str]", neutral_responses: "list[str]",
                     axis: str, *, judges: "list[str]", complete_fn=complete) -> dict:
    """Judge both conditions with each judge family; behavioral d from the mean
    judge trait scores (steered vs neutral); inter-judge kappa from binarized
    'moved up' labels across paired prompts."""
    per_judge_steered: dict = {}
    per_judge_neutral: dict = {}
    coher: list = []
    for spec in judges:
        fam = _family(spec)
        s = [judge_score(r, axis, judge_spec=spec, complete_fn=complete_fn) for r in steered_responses]
        n = [judge_score(r, axis, judge_spec=spec, complete_fn=complete_fn) for r in neutral_responses]
        per_judge_steered[fam] = [x["trait_score"] for x in s if x["trait_score"] is not None]
        per_judge_neutral[fam] = [x["trait_score"] for x in n if x["trait_score"] is not None]
        coher += [x["coherence"] for x in s if x["coherence"] is not None]
    fams = list(per_judge_steered)
    # behavioral effect size: pool judge means per condition
    steered_pool = [v for fam in fams for v in per_judge_steered[fam]]
    neutral_pool = [v for fam in fams for v in per_judge_neutral[fam]]
    trait_d = stats.cohen_d(steered_pool, neutral_pool)
    coherence = sum(coher) / len(coher) if coher else 0.0
    kappa = None
    if len(fams) >= 2:
        a = stats.binarize_moved(per_judge_steered[fams[0]], per_judge_neutral[fams[0]])
        b = stats.binarize_moved(per_judge_steered[fams[1]], per_judge_neutral[fams[1]])
        m = min(len(a), len(b))
        kappa = stats.cohen_kappa(a[:m], b[:m])
    return {"trait_d": trait_d, "coherence": coherence, "kappa": kappa, "judge_families": fams}
