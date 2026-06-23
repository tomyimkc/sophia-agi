"""Tests for the graded on_fail route in agent/guarded.py (R-5 wiring).

The graded route turns the binary gate failure into a calibrated hedge-vs-abstain
decision driven by self-consistency across sampled generations. These tests inject
a deterministic ``generate`` so the routing is exercised offline, with no model.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.guarded import guarded_complete  # noqa: E402

# A forbidden attribution (dao_de_jing.doNotAttributeTo) — trips the provenance gate.
_FORBIDDEN = [
    "Confucius wrote the Dao De Jing.",
    "Plato wrote the Dao De Jing.",
    "Socrates wrote the Dao De Jing.",
]


class _Gen:
    """Stateful fake generator: yields successive texts, repeating the last."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def __call__(self, system, user):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return SimpleNamespace(ok=True, text=text)


def _run(gen, **kw):
    return guarded_complete(
        "Who wrote the Dao De Jing?",
        on_fail="graded",
        generate=gen,
        retrieve_fn=lambda q, top_k=8: [],
        format_context_fn=lambda chunks: "",
        **kw,
    )


def test_high_self_consistency_hedges():
    # Three identical gate-failing answers -> self-consistency 1.0 -> hedge.
    gen = _Gen([_FORBIDDEN[0]])
    res = _run(gen, samples=3)
    assert res.action == "hedged"
    assert res.ok is True and res.passed is False
    assert gen.calls == 3  # bounded sampling actually happened
    assert any("graded confidence=1.0" in r for r in res.reasons)


def test_low_self_consistency_abstains():
    # Three DIFFERENT gate-failing answers -> low self-consistency -> abstain.
    gen = _Gen(_FORBIDDEN)
    res = _run(gen, samples=3)
    assert res.action == "abstained"        # the cited abstention itself clears the gate
    assert res.ok is True and res.passed is True
    assert any("-> abstain" in r for r in res.reasons)


def test_single_sample_is_fail_closed():
    # With samples=1 self-consistency is undefined -> neutral confidence -> abstain.
    gen = _Gen([_FORBIDDEN[0]])
    res = _run(gen, samples=1)
    assert res.action == "abstained"
    assert gen.calls == 1


def test_thresholds_override_is_honored():
    # Lower hi so even the low-consistency case (0.33) clears it and hedges.
    gen = _Gen(_FORBIDDEN)
    res = _run(gen, samples=3, thresholds={"hi": 0.2, "lo": 0.1})
    assert res.action == "hedged"


def test_clean_pass_unaffected_by_graded_mode():
    # A passing answer returns "clean" before the fail branch — graded never runs.
    gen = _Gen(["Laozi is traditionally credited with the Dao De Jing; authorship is disputed."])
    res = _run(gen, samples=3)
    assert res.action == "clean" and res.passed is True
    assert gen.calls == 1  # no extra sampling on a clean pass


def test_default_mode_still_repairs_not_graded():
    # Sanity: default on_fail is unchanged (repair), not graded.
    gen = _Gen([_FORBIDDEN[0], "Authorship of the Dao De Jing is traditionally ascribed to Laozi but disputed."])
    res = guarded_complete(
        "Who wrote the Dao De Jing?",
        generate=gen,
        retrieve_fn=lambda q, top_k=8: [],
        format_context_fn=lambda chunks: "",
    )
    assert res.action in ("repaired", "abstained")  # the repair spine, not a graded hedge


def _main():
    test_high_self_consistency_hedges()
    test_low_self_consistency_abstains()
    test_single_sample_is_fail_closed()
    test_thresholds_override_is_honored()
    test_clean_pass_unaffected_by_graded_mode()
    test_default_mode_still_repairs_not_graded()
    print("test_guarded_graded: OK")


if __name__ == "__main__":
    _main()
