#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/entity_aliases.py (offline, stdlib-only)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.entity_aliases import (  # noqa: E402
    author_surface_forms,
    is_ambiguous_surname,
)


def test_multitoken_surname() -> None:
    forms = author_surface_forms("Leo Tolstoy")
    assert "tolstoy" in forms, forms
    assert "leo tolstoy" in forms, forms


def test_marcus_aurelius() -> None:
    forms = author_surface_forms("Marcus Aurelius")
    assert "aurelius" in forms, forms
    assert "marcus aurelius" in forms, forms


def test_overcommon_given_stays_full_only() -> None:
    # "Paul" is over-common: a record author "The Apostle Paul" -> the surname
    # token "paul" must NOT be emitted as a bare marker.
    forms = author_surface_forms("Apostle Paul")
    assert "paul" not in forms, forms
    assert "apostle paul" in forms, forms
    # And as a standalone name, no bare-surname duplication beyond the full form.
    assert author_surface_forms("Paul") == ["paul"], author_surface_forms("Paul")
    assert is_ambiguous_surname("Paul") is True


def test_guard_short_and_generic() -> None:
    assert is_ambiguous_surname("Li") is True       # too short
    assert is_ambiguous_surname("the") is True      # generic
    assert is_ambiguous_surname("Tolstoy") is False
    assert is_ambiguous_surname("Aurelius") is False


def test_cjk_alias_resolves() -> None:
    # AUTHOR_ALIASES maps confucius -> ["kongzi", "孔子"]; a CJK alias must surface.
    forms = author_surface_forms("confucius")
    assert "孔子" in forms, forms
    assert "kongzi" in forms, forms


def test_pure_import_no_side_effects() -> None:
    # Re-import must be safe; module exposes only the documented surface.
    import importlib

    import agent.entity_aliases as ea

    importlib.reload(ea)
    assert callable(ea.author_surface_forms)
    assert callable(ea.is_ambiguous_surname)


def test_end_to_end_recall_fix() -> None:
    """Prove the fix: with augmented author_markers, provenance_faithful catches
    'Tolstoy wrote Crime and Punishment' where today it does not.

    We cannot edit benchmark_checks.author_markers (the integrator does that), so
    we monkeypatch it in-test to emulate the one-line wire-in
    ``markers += author_surface_forms(author_id)`` and demonstrate the delta.
    """
    from agent import verifiers as v

    records = {
        "crime_and_punishment": {
            "canonicalTitleEn": "Crime and Punishment",
            "doNotAttributeTo": ["Leo Tolstoy"],
        }
    }
    claim = "Tolstoy wrote Crime and Punishment."

    # author_surface_forms is now wired into benchmark_checks.author_markers, so the
    # LIVE gate catches the surname-only phrasing directly (no monkeypatch needed).
    assert "tolstoy" in author_surface_forms("Leo Tolstoy")
    verify = v.provenance_faithful(records)
    res = verify(claim, None, {})
    assert res["passed"] is False, res
    assert any("Leo Tolstoy" in r for r in res["reasons"]), res

    # Correction phrasing still passes (no over-firing on negation).
    ok = verify("Tolstoy did not write Crime and Punishment; Dostoevsky did.", None, {})
    assert ok["passed"] is True, ok


def main() -> int:
    test_multitoken_surname()
    test_marcus_aurelius()
    test_overcommon_given_stays_full_only()
    test_guard_short_and_generic()
    test_cjk_alias_resolves()
    test_pure_import_no_side_effects()
    test_end_to_end_recall_fix()
    print("test_entity_aliases: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
