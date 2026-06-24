#!/usr/bin/env python3
"""Local-Sophia dataset: train/eval contamination guard + build invariants (offline)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.dataset_guard import check_contamination, normalize, prompt_of  # noqa: E402
from tools import build_local_sophia_dataset as build  # noqa: E402


def test_prompt_extraction_sft_and_dpo() -> None:
    assert prompt_of({"messages": [{"role": "user", "content": "Hi?"}]}) == "Hi?"
    assert prompt_of({"prompt": "Who wrote X?"}) == "Who wrote X?"
    assert prompt_of({"question": "Q?"}) == "Q?"


def test_guard_flags_injected_overlap() -> None:
    evalset = {normalize("Who wrote the Republic?")}
    dirty = [{"prompt": "Who wrote the Republic?"}, {"prompt": "totally unrelated prompt"}]
    r = check_contamination(dirty, evalset)
    assert r["clean"] is False and r["overlapCount"] == 1


def test_guard_passes_clean_set() -> None:
    evalset = {normalize("Who wrote the Republic?")}
    clean = [{"prompt": "A novel, non-eval question about provenance."}]
    assert check_contamination(clean, evalset)["clean"] is True


def test_build_check_is_fail_closed_and_clean() -> None:
    # build --check decontaminates then asserts disjoint; must return 0 (clean).
    assert build.build(check_only=True) == 0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} local-sophia-dataset tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
