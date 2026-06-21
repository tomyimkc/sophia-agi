#!/usr/bin/env python3
"""Tests for the semantic citation-faithfulness tier (offline, stub judge).

A deterministic stub judge stands in for the production LLM judge so this verifies
WIRING ONLY — that the verifier routes supported / contradicted / abstained
correctly and is fail-closed. It is NOT a measurement of semantic accuracy (that
needs a real LLM judge under the no-overclaim gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import verifiers as v  # noqa: E402
from agent.legal_faithfulness import Verdict, assess_text, claim_citation_pairs, register_holdings  # noqa: E402

# Stub judge: "supports" iff a shared keyword links proposition and holding.
def keyword_stub(proposition: str, holding: str) -> Verdict:
    p, h = proposition.lower(), holding.lower()
    for kw in ("prorogation", "same-sex", "marry", "sanction", "fabricat", "ai"):
        if kw in p and kw in h:
            return Verdict(supports=True, abstained=False, reason=f"shared:{kw}", method="stub")
    return Verdict(supports=False, abstained=False, reason="no shared concept", method="stub")


def _holding_for():
    h = register_holdings()
    return lambda c: h.get(c)


def test_register_has_holdings() -> None:
    h = register_holdings()
    assert "[2019] UKSC 41" in h and "prorogation" in h["[2019] UKSC 41"].lower()


def test_pairs_extraction() -> None:
    pairs = claim_citation_pairs("Prorogation was unlawful per [2019] UKSC 41. Unrelated sentence.")
    assert pairs and pairs[0][1] == ["[2019] UKSC 41"]


def test_supported_proposition_passes() -> None:
    ver = v.legal_holding_faithful(holding_for=_holding_for(), judge=keyword_stub)
    r = ver("The prorogation of Parliament was unlawful, per [2019] UKSC 41.", None, {})
    assert r["passed"] is True
    assert "[2019] UKSC 41" in r["detail"]["supported"]


def test_misstated_authority_is_flagged() -> None:
    ver = v.legal_holding_faithful(holding_for=_holding_for(), judge=keyword_stub)
    # cites a real case for a proposition its holding does not establish
    r = ver("Obergefell v. Hodges, 576 U.S. 644, bars all immigration appeals.", None, {})
    assert r["passed"] is False
    assert any("576 U.S. 644" in reason for reason in r["reasons"])
    assert r["detail"]["contradicted"]


def test_no_holding_text_abstains() -> None:
    ver = v.legal_holding_faithful(holding_for=lambda c: None, judge=keyword_stub)
    r = ver("As held in [2019] UKSC 41, the point is settled.", None, {})
    # abstained, not failed (default), and not falsely counted as supported
    assert r["passed"] is True
    assert r["detail"]["abstained"] and not r["detail"]["supported"]


def test_require_support_fails_closed_on_abstain() -> None:
    ver = v.legal_holding_faithful(holding_for=lambda c: None, judge=keyword_stub, require_support=True)
    r = ver("As held in [2019] UKSC 41, the point is settled.", None, {})
    assert r["passed"] is False  # cannot verify support -> fail-closed


def test_broken_judge_fails_closed() -> None:
    def boom(_p, _h):
        raise RuntimeError("judge down")

    r = assess_text("Per [2019] UKSC 41, prorogation was unlawful.", holding_for=_holding_for(), judge=boom)
    assert r["supported"] == [] and r["abstained"]  # broken judge never supports


def main() -> int:
    import inspect

    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_legal_faithfulness: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
