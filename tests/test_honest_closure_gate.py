# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/honest_closure_gate.py — the honest-closure ratchet.

Two synthetic ledger tables exercise the anti-farming guards:
  (i)  a HEALTHY mix (open + receipted-positive closures + checkable honest negatives) PASSES;
  (ii) a FLOOD of bare "honest N=0" closures with no checkable reason trips BOTH the
       unverifiable-negative alarm AND the negative/receipted-positive ratio alarm.

Also checks the parser ignores sub-tables embedded in a cell and that the real repo ledger
parses to a non-trivial row count.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import honest_closure_gate as hcg  # noqa: E402  (tools/ is on sys.path above)

_HEADER = (
    "# Failure Ledger\n\n"
    "| Failure ID | Status | Claim impact | Required response |\n"
    "|---|---|---|---|\n"
)


def _healthy_ledger() -> str:
    rows = [
        # open rows (still in progress)
        "| foo-eval-not-run-2026-06-10 | Open (machinery built) | impact | resp |",
        "| bar-harness-only-2026-06-11 | Partial (scaffold) | impact | resp |",
        # receipted positive closures (cite an artifact path or a PASS/GO marker)
        "| baz-verifier-validated-2026-06-12 | Closed (research result) | ran and PASS "
        "agi-proof/benchmark-results/baz.public-report.json | resp |",
        "| qux-scorer-fixed-2026-06-12 | Closed (M1 GO) | fixed, artifact tools/run_qux.py | resp |",
        "| quux-gate-validated-2026-06-13 | Closed (VALIDATED) | receipt "
        "agi-proof/quux/quux.public-report.json | resp |",
        # checkable honest negative (cites assert_decontam + a findings.json reproduce path)
        "| corge-source-exhausted-2026-06-13 | Closed (honest NEGATIVE - decontam-exhausted) | "
        "largest honest N=0; reproduce: python3 tools/audit_corge.py; findings "
        "agi-proof/corge/findings.json; decontam CLEAN via assert_decontam | resp |",
    ]
    return _HEADER + "\n".join(rows) + "\n"


def _farm_flood_ledger() -> str:
    rows = [
        # exactly ONE receipted positive closure...
        "| real-thing-validated-2026-06-20 | Closed (research result) | ran PASS "
        "agi-proof/benchmark-results/real.public-report.json | resp |",
    ]
    # ...swamped by a FLOOD of bare "honest N=0" negatives with NO checkable token.
    for i in range(8):
        rows.append(
            f"| farm-null-{i:02d}-2026-06-2{i % 10} | Closed (honest NEGATIVE) | "
            f"honest N=0; nothing to see here, trust me | just close it |"
        )
    return _HEADER + "\n".join(rows) + "\n"


def _ledger_with_embedded_subtable() -> str:
    # A "Required response" cell that itself contains a mini markdown table (pipe-delimited).
    # The parser must NOT treat those inner lines as failure rows.
    rows = [
        "| only-real-row-validated-2026-06-15 | Closed (PASS) | receipt "
        "agi-proof/x.public-report.json | see counts below |",
        "| Count | 8 | 1 | note |",  # noise line: id 'Count' is not a valid failure id
        "| 34 | 2 | 6 | more noise |",  # noise line: numeric first cell
    ]
    return _HEADER + "\n".join(rows) + "\n"


def test_healthy_mix_passes():
    res = hcg.evaluate(_healthy_ledger())
    c = res["counts"]
    assert c["open"] == 2, c
    assert c["closedPositive"] == 3, c
    assert c["receiptedPositive"] == 3, c
    assert c["closedNegative"] == 1, c
    assert c["unverifiableNegative"] == 0, c
    assert res["alarms"] == [], res["alarms"]
    assert res["pass"] is True
    # honestClosureRate CS is defined over per-release quality; all closures verifiable -> 1.0.
    hr = res["honestClosureRate"]
    assert hr["perReleaseMean"] == 1.0, hr
    lo, hi = hr["confidenceSequence"]
    assert lo is not None and hi is not None
    assert lo <= 1.0 <= hi + 1e-9, (lo, hi)
    assert res["canClaimAGI"] is False


def test_farm_flood_trips_both_alarms():
    res = hcg.evaluate(_farm_flood_ledger())
    c = res["counts"]
    assert c["closedNegative"] == 8, c
    assert c["receiptedPositive"] == 1, c
    assert c["unverifiableNegative"] == 8, c
    # both anti-farming alarms must fire
    joined = " ".join(res["alarms"])
    assert "unverifiable-negative" in joined, res["alarms"]
    assert "ratio" in joined, res["alarms"]
    assert res["pass"] is False
    # ratio 8 negatives / 1 receipted positive = 8.0 > default threshold 1.0
    assert res["antiFarming"]["negativeToReceiptedPositiveRatio"] == 8.0, res["antiFarming"]
    # every flooded id is reported as unverifiable
    assert len(res["antiFarming"]["unverifiableNegativeIds"]) == 8


def test_infinite_ratio_when_no_receipted_positive():
    # A negative closure with zero receipted positives -> ratio is inf -> ratio alarm fires,
    # and the receipt-less negative is also unverifiable.
    led = _HEADER + (
        "| lonely-null-2026-06-01 | Closed (honest NEGATIVE) | honest N=0, no receipt | resp |\n"
    )
    res = hcg.evaluate(led)
    assert res["counts"]["receiptedPositive"] == 0
    assert res["antiFarming"]["negativeToReceiptedPositiveRatio"] is None  # inf serialized as None
    assert res["pass"] is False
    assert any("exceeds pre-registered threshold" in a for a in res["alarms"])


def test_checkable_negative_is_not_unverifiable():
    # A single honest-negative that cites assert_decontam is verifiable; with a receipted
    # positive present the ratio is 1.0 (== threshold) so NO alarm.
    led = _HEADER + (
        "| pos-validated-2026-06-05 | Closed (PASS) | agi-proof/p.public-report.json | resp |\n"
        "| neg-exhausted-2026-06-05 | Closed (honest NEGATIVE) | honest N=0; decontam CLEAN via "
        "assert_decontam; findings agi-proof/n/findings.json | resp |\n"
    )
    res = hcg.evaluate(led)
    assert res["counts"]["unverifiableNegative"] == 0, res["counts"]
    assert res["antiFarming"]["negativeToReceiptedPositiveRatio"] == 1.0
    assert res["pass"] is True, res["alarms"]


def test_embedded_subtable_ignored():
    res = hcg.evaluate(_ledger_with_embedded_subtable())
    # only the one real failure row is counted; 'Count' and '34' noise lines are dropped
    assert res["counts"]["total"] == 1, res["counts"]
    assert res["counts"]["closedPositive"] == 1, res["counts"]


def test_real_ledger_parses():
    # The actual repo ledger must parse to a non-trivial number of real rows. We assert only
    # structural facts (it parses, counts are internally consistent) — NOT that its closure
    # rate is good or bad (the tool measures; it does not judge here).
    ledger = ROOT / "agi-proof" / "failure-ledger.md"
    res = hcg.evaluate(ledger.read_text(encoding="utf-8"))
    c = res["counts"]
    assert c["total"] > 50, c
    assert c["total"] == c["open"] + c["closedPositive"] + c["closedNegative"], c
    assert c["receiptedPositive"] <= c["closedPositive"], c
    assert c["unverifiableNegative"] <= c["closedNegative"], c
    # CS is defined whenever there is at least one closed row (there is)
    assert res["honestClosureRate"]["nReleases"] >= 1


def _run_all():
    test_healthy_mix_passes()
    test_farm_flood_trips_both_alarms()
    test_infinite_ratio_when_no_receipted_positive()
    test_checkable_negative_is_not_unverifiable()
    test_embedded_subtable_ignored()
    test_real_ledger_parses()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_all()
