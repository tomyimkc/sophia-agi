#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for user-supplied source-discipline records (SOPHIA_DISCIPLINE_RECORDS). Offline.

Lets a small model enforce the user's OWN attribution rules (legal/corporate/code
provenance) with the same machine-checked gate, beyond the 4 seeded domains.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import verifiers as v  # noqa: E402

_ENV = "SOPHIA_DISCIPLINE_RECORDS"


def _write(tmp: str, name: str, obj: dict) -> str:
    p = Path(tmp) / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_user_records_loaded_and_enforced() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "mine.json", {
            "phoenix_charter": {"canonicalTitleEn": "Project Phoenix Charter", "doNotAttributeTo": ["Alice"]},
        })
        os.environ[_ENV] = tmp
        try:
            records = v._load_provenance_records()
            assert "phoenix_charter" in records
            verify = v.provenance_faithful()
            assert verify("Alice wrote the Project Phoenix Charter.", None, {})["passed"] is False
            assert verify("Alice did not write the Project Phoenix Charter.", None, {})["passed"] is True
        finally:
            os.environ.pop(_ENV, None)


def test_seeded_records_still_load_without_env() -> None:
    os.environ.pop(_ENV, None)
    records = v._load_provenance_records()
    assert any("dao" in k.lower() for k in records), "seeded dao_de_jing record should still load"


def test_user_record_without_donotattribute_is_skipped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "bad.json", {"oops": {"canonicalTitleEn": "Some Title"}})  # no doNotAttributeTo
        os.environ[_ENV] = tmp
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                records = v._load_provenance_records()
            assert "oops" not in records
            assert any("oops" in str(w.message) for w in caught), "skipped user record should warn"
        finally:
            os.environ.pop(_ENV, None)


def test_single_file_spec_works() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "one.json", {"r": {"canonicalTitleEn": "The Vault Spec", "doNotAttributeTo": ["Bob"]}})
        os.environ[_ENV] = path
        try:
            records = v._load_provenance_records()
            assert "r" in records
        finally:
            os.environ.pop(_ENV, None)


def main() -> int:
    test_user_records_loaded_and_enforced()
    test_seeded_records_still_load_without_env()
    test_user_record_without_donotattribute_is_skipped()
    test_single_file_spec_works()
    print("test_discipline_records: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
