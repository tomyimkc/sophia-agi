#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Parity gate for the sophia-lex accelerator vs the pure-Python oracle.

The deliverable of sophia-lex is NOT just speed — it is a deterministic scanner
whose verdicts AGREE with the reference Python tools. This test proves that:

  * overclaim: the Rust scanner and the lint_claims FORBIDDEN regexes produce the
    SAME (line, why) verdicts on (a) a shared fixture suite of positive/negative
    vectors and (b) every committed file the linter actually scans.
  * decontam: the Rust near-dup scanner and the assert_decontam Python scan agree
    on a synthetic corpus where the Python `--max-eval-shingle` cap does not bite.

Safety: this test SKIPS unless the compiled binary is already available, or
SOPHIA_LEX_BUILD=1 is set (then it builds via cargo, skipping if cargo is absent
or the build fails). So the default `pytest -q` gate stays green on a machine
with no Rust toolchain — acceleration is opt-in, the Python oracle is the default.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import _lex_accel  # noqa: E402
from lint_claims import ALLOW_MARKER, FORBIDDEN, _files  # noqa: E402

FIXTURES = ROOT / "tools" / "sophia-lex" / "fixtures" / "overclaim_vectors.jsonl"


@pytest.fixture(scope="module")
def lexbin():
    if _lex_accel.available():
        return _lex_accel.binary_path()
    if os.environ.get("SOPHIA_LEX_BUILD") == "1":
        try:
            return _lex_accel.build()
        except _lex_accel.LexUnavailable as exc:
            pytest.skip(f"sophia-lex build unavailable: {exc}")
    pytest.skip("sophia-lex binary not built (set SOPHIA_LEX_BUILD=1 or build the crate to run)")


def _python_overclaim(text: str) -> set[tuple[int, str]]:
    """Reference (line, why) verdicts from the lint_claims FORBIDDEN regexes."""
    out: set[tuple[int, str]] = set()
    for i, line in enumerate(text.split("\n"), 1):
        if ALLOW_MARKER in line:
            continue
        low = line.lower()
        for pat, why in FORBIDDEN:
            if re.search(pat, low):
                out.add((i, why))
    return out


def _rust_overclaim(path: Path) -> set[tuple[int, str]]:
    return {(line, why) for _, line, why in _lex_accel.overclaim_scan([path])}


def _load_fixtures() -> list[dict]:
    import json
    rows = []
    for raw in FIXTURES.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


def test_overclaim_parity_on_fixtures(lexbin, tmp_path):
    rows = _load_fixtures()
    assert rows, "fixture suite is empty"
    mismatches = []
    for idx, row in enumerate(rows):
        text = row["text"]
        py = _python_overclaim(text)
        f = tmp_path / f"vec_{idx}.md"
        f.write_text(text, encoding="utf-8")
        rust = _rust_overclaim(f)
        if py != rust:
            mismatches.append((row.get("note", ""), text, sorted(py), sorted(rust)))
        # the fixture's own expectation must agree with the oracle (catches a stale fixture)
        assert bool(py) == bool(row["violation"]), (
            f"fixture expectation wrong for {row.get('note')!r}: "
            f"expected violation={row['violation']} but oracle found {sorted(py)}"
        )
    assert not mismatches, "Rust/Python overclaim divergence:\n" + "\n".join(
        f"  [{n}] {t!r}\n    python={p}\n    rust={r}" for n, t, p, r in mismatches
    )


def test_overclaim_parity_on_committed_corpus(lexbin):
    """Every file the linter actually scans must get identical verdicts."""
    files = _files()
    assert files, "no scanned files found"
    diverged = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        py = _python_overclaim(text)
        rust = _rust_overclaim(path)
        if py != rust:
            diverged.append((str(path.relative_to(ROOT)), sorted(py), sorted(rust)))
    assert not diverged, "corpus parity divergence:\n" + "\n".join(
        f"  {f}\n    python={p}\n    rust={r}" for f, p, r in diverged
    )


def test_decontam_parity_uncapped_window(lexbin):
    """On a synthetic corpus (eval well under the cap), the Rust full-coverage
    near-dup scan and the Python capped scan flag the SAME train prompts."""
    from assert_decontam import _jaccard, _shingles  # noqa: E402
    from provenance_bench.dataset_guard import normalize  # noqa: E402

    base = " ".join(f"w{i}" for i in range(20))
    eval_prompts = [
        base,
        "completely different unrelated content about felines and astronomy here",
        " ".join(f"x{i}" for i in range(20)),
    ]
    train = [
        base + " w99",          # near-dup of eval[0] (not exact): J = 16/17 ≈ 0.94
        "a totally distinct training prompt with no overlap at all really",
    ]
    k, thr = 5, 0.9

    # Python capped scan (cap large enough to be a no-op here).
    eval_sh = [(e, _shingles(e, k)) for e in eval_prompts]
    py_flagged = set()
    seen = set()
    for pr in train:
        npr = normalize(pr)
        if npr in seen:
            continue
        seen.add(npr)
        tsh = _shingles(pr, k)
        if not tsh:
            continue
        for e, esh in eval_sh:
            if _jaccard(tsh, esh) >= thr and npr != e:
                py_flagged.add(pr[:80])
                break

    rust = _lex_accel.decontam_near(train, eval_prompts, k=k, jaccard=thr)
    rust_flagged = {t for _, t, _ in rust}

    assert py_flagged == rust_flagged, f"python={py_flagged} rust={rust_flagged}"
    assert py_flagged, "expected the near-duplicate train prompt to be flagged"
