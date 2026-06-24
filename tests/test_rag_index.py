#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""RAG index and retrieval tests (no API keys required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rag_sources import collect_curated_chunks, is_holdout_example, load_benchmark_ids  # noqa: E402
from agent.vector_store import IndexedChunk, save_index, load_index, search  # noqa: E402


def test_holdout_examples_excluded() -> None:
    bench_ids, bench_questions = load_benchmark_ids()
    holdout = 0
    included = 0
    for path in (ROOT / "training" / "examples").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if is_holdout_example(payload, bench_ids, bench_questions):
            holdout += 1
        else:
            included += 1
    chunks = collect_curated_chunks()
    example_chunks = [c for c in chunks if c.kind == "teacher_example"]
    assert len(example_chunks) == included
    assert holdout > 0


def test_keyword_search_roundtrip(tmp_path: Path) -> None:
    chunks = [
        IndexedChunk(
            chunk_id="t1",
            path="data/attributions.json#dao_de_jing",
            title="dao_de_jing",
            text="Laozi attribution legendary; Confucius did not write Dao De Jing",
            domain="philosophy",
            kind="data",
        )
    ]
    save_index(chunks, tmp_path)
    loaded = load_index(tmp_path)
    hits = search("Did Confucius write the Dao De Jing?", loaded, top_k=3)
    assert hits
    assert "dao_de_jing" in hits[0].path or "Laozi" in hits[0].excerpt


def test_built_index_retrieve() -> None:
    from agent.retrieval import retrieve

    index_path = ROOT / "rag" / "index" / "chunks.jsonl"
    if not index_path.exists():
        return
    hits = retrieve("Did Confucius write the Dao De Jing?", top_k=3)
    assert hits
    assert any("dao" in h.path.lower() or "confucius" in h.excerpt.lower() or "laozi" in h.excerpt.lower() for h in hits)


def main() -> int:
    import tempfile

    test_holdout_examples_excluded()
    with tempfile.TemporaryDirectory() as tmp:
        test_keyword_search_roundtrip(Path(tmp))
    test_built_index_retrieve()
    print("test_rag_index: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())