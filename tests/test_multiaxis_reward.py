"""Tests for the multi-axis deterministic reward (Thesis D).

Pins the four design invariants: fail-closed, reward-positive abstention, ordering,
and density-beats-single-axis (the anti-collapse property).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent import multiaxis_reward as mar  # noqa: E402
from agent.gate_reward import REWARD_VIOLATION, reward as single_axis  # noqa: E402


def test_self_check_invariants():
    rep = mar.self_check()
    assert rep["weightsSumToOne"]
    assert rep["distinctMultiAxisValues"] > rep["distinctSingleAxisValues"]


def test_fail_closed_dominates():
    case = {"id": "c", "work": "Dao De Jing", "doNotAttributeTo": ["Confucius"], "split": "answerable"}
    # Even wrapped in perfect-looking hedged prose, a forbidden attribution pins to the floor.
    text = "Reportedly, and with appropriate caution, Confucius wrote the Dao De Jing."
    assert mar.multiaxis_reward(text, case=case) == REWARD_VIOLATION


def test_abstention_reward_positive_but_below_clean():
    forbidden = {"id": "c", "work": "Dao De Jing", "doNotAttributeTo": ["Confucius"], "split": "answerable"}
    unans = {"id": "u", "split": "unanswerable"}
    clean = mar.multiaxis_reward("The Dao De Jing is attributed to Laozi.", case=forbidden)
    abstain = mar.multiaxis_reward("I can't verify that, so I won't assert it.", case=unans)
    assert 0 < abstain < clean


def test_density_beats_single_axis():
    # Across a graded fixture the multi-axis reward must resolve more distinct values than the
    # 3-valued single-axis reward — this is the anti-reward-collapse property.
    forbidden = {"id": "c", "work": "Dao De Jing", "doNotAttributeTo": ["Confucius"], "split": "answerable"}
    texts = [
        "The Dao De Jing is attributed to Laozi.",
        "Likely Laozi, though the attribution is traditional and uncertain.",
        "Absolutely, definitely Laozi, 100% certain, beyond doubt.",
        "The author appears to be Laozi, reportedly.",
    ]
    multi = {round(mar.multiaxis_reward(t, case=forbidden), 4) for t in texts}
    single = {round(single_axis(t), 4) for t in texts}
    assert len(multi) > len(single)


def test_grpo_reward_wrapper_shape():
    fn = mar.make_grpo_reward()
    case = {"id": "c", "split": "answerable"}
    out = fn(prompts=["p1", "p2"], completions=["Laozi, reportedly.", "Absolutely certain."], cases=[case, case])
    assert isinstance(out, list) and len(out) == 2
    assert all(-1.0 <= r <= 1.0 for r in out)


def test_deterministic():
    case = {"id": "c", "work": "Dao De Jing", "doNotAttributeTo": ["Confucius"], "split": "answerable"}
    t = "Likely Laozi, though reportedly uncertain."
    assert mar.multiaxis_reward(t, case=case) == mar.multiaxis_reward(t, case=case)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all multiaxis_reward tests passed")
