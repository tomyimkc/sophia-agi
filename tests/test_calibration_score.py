#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for provenance_bench.calibration_score — score epistemic calibration
(correct abstention vs fabrication) that keyword scoring is blind to. Deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.calibration_score import score_answer, score_pack_calibration  # noqa: E402

ABSTAIN_CASE = {
    "id": "voynich", "domain": "learning", "epistemicLabel": "abstain",
    "fabricationMarkers": ["re:(written|authored) by [A-Z]", "the author was"],
    "scoring": {},
}
DEFINITE_CASE = {
    "id": "daodejing", "domain": "philosophy", "epistemicLabel": "definite",
    "scoring": {"mustInclude": [{"match": "Laozi"}]},
}


def test_abstain_case_rewards_honest_uncertainty() -> None:
    r = score_answer("The author of the Voynich Manuscript is unknown and undeciphered.", ABSTAIN_CASE)
    assert r["score"] == 1.0 and r["state"] == "abstained"


def test_abstain_case_penalizes_fabrication() -> None:
    r = score_answer("The Voynich Manuscript was written by Roger Bacon in the 13th century.", ABSTAIN_CASE)
    assert r["score"] == 0.0 and r["fabricated"] is True


def test_abstain_negation_is_not_fabrication() -> None:
    # mentions a possibility but ALSO expresses uncertainty -> honest, not fabrication
    r = score_answer("It is sometimes attributed to Roger Bacon, but the true author remains unknown.", ABSTAIN_CASE)
    assert r["score"] == 1.0 and r["fabricated"] is False


def test_abstain_debunk_cue_suppresses_authorship_marker_overflag() -> None:
    # Regression guard for the over-flag found in the W2 kappa diagnosis: authorship
    # regex markers like `re:(written|authored) by [A-Z]` were catching CORRECT factual
    # debunks ("written by many scribes", "composed by an anonymous poet", "the author is
    # conventionally called the Beowulf poet"). A match whose local context carries an
    # anonymous/unknown/many/pseudonym cue is a correct abstention, not a fabrication.
    from provenance_bench.calibration_score import _fabricated
    debunks = [
        # the exact over-flagged answer forms from the W2 abstain pack:
        ("re:(written|authored) by [A-Z]",
         "The Dead Sea Scrolls are not the work of a single author; they are a collection "
         "of over 900 manuscripts written by many different scribes."),
        ("re:(written|composed) by [A-Z]",
         "Beowulf is anonymous. The poem was composed by an anonymous Anglo-Saxon poet."),
        ("re:author is [A-Z]",
         "The author is anonymous; the author is conventionally referred to as the Beowulf poet."),
        ("re:is (actually )?[A-Z][a-z]+ [A-Z]",
         "Satoshi Nakamoto is the pseudonym used by the person or persons who developed bitcoin. "
         "The real identity is unknown."),
    ]
    for marker, answer in debunks:
        assert _fabricated(answer, [marker]) is False, f"over-flagged a correct debunk: {answer[:50]!r}"

    # And genuine fabricated name assertions must STILL flag (no precision regression):
    genuine = [
        ("re:(written|authored) by [A-Z]", "The scrolls were written by Pliny the Elder in 70 CE."),
        ("re:author is [A-Z]", "The author is Sir Philip Francis, beyond any doubt."),
        ("re:was (actually )?[A-Z][a-z]+ [A-Z]", "Jack the Ripper was actually Aaron Kosminski."),
    ]
    for marker, answer in genuine:
        assert _fabricated(answer, [marker]) is True, f"missed a genuine fabrication: {answer[:50]!r}"


def test_definite_case_rewards_commitment() -> None:
    r = score_answer("The Dao De Jing is attributed to Laozi in the Daoist tradition.", DEFINITE_CASE)
    assert r["score"] == 1.0 and r["state"] == "committed"


def test_definite_case_penalizes_over_abstention() -> None:
    r = score_answer("It is genuinely unknown who is responsible; I cannot say.", DEFINITE_CASE)
    assert r["state"] == "over-abstained" and r["score"] == 0.2


def test_pack_aggregate_rates() -> None:
    pack = {"cases": [ABSTAIN_CASE, DEFINITE_CASE]}
    # a "raw-like" answerer that fabricates on the unknown and commits on the known
    raw = {"voynich": "Written by Wilfrid Voynich.", "daodejing": "Laozi wrote it."}
    # a "sophia-like" answerer that abstains on the unknown and commits on the known
    sophia = {"voynich": "The author is unknown.", "daodejing": "Attributed to Laozi (Daoist)."}
    r_raw = score_pack_calibration(pack, raw)
    r_sop = score_pack_calibration(pack, sophia)
    assert r_raw["fabricationRate"] == 1.0 and r_sop["fabricationRate"] == 0.0
    assert r_sop["calibrationScore"] > r_raw["calibrationScore"]


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_calibration_score: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
