#!/usr/bin/env python3
"""Tests for memory consolidation + plan-time recall (offline)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import harness, memory_consolidation as mc, wiki_store  # noqa: E402


def _redirect(tmp: Path) -> None:
    wiki_store.CANONICAL_DIR = tmp / "wiki"
    wiki_store.MEMORY_DIR = tmp / "memory"
    wiki_store.DRAFT_DIR = tmp / "wiki" / "drafts"
    (tmp / "wiki").mkdir(parents=True, exist_ok=True)


def test_consolidate_clean_answer() -> None:
    with tempfile.TemporaryDirectory() as t:
        _redirect(Path(t))
        res = mc.consolidate_result(
            "Who wrote the Dao De Jing?",
            "The Dao De Jing is attributed to Laozi (legendary); Confucius did not write it. Decision: separate the lineages. 中文摘要：道家文本。",
            task_id="q-dao",
        )
        assert res["ok"] is True, res
        page = wiki_store.read_page("mem_q_dao")
        assert page is not None and page.page_type == "memory"


def test_consolidate_rejects_lineage_merge_answer() -> None:
    with tempfile.TemporaryDirectory() as t:
        _redirect(Path(t))
        res = mc.consolidate_result(
            "dao authorship",
            "Confucius wrote the Dao De Jing himself.",
            task_id="bad-dao",
        )
        assert res["ok"] is False  # contaminated memory is never stored
        assert any("forbidden attribution" in r for r in res["reasons"])


def test_consolidate_runs_folder() -> None:
    with tempfile.TemporaryDirectory() as t:
        _redirect(Path(t))
        runs = Path(t) / "runs"
        runs.mkdir()
        # a verified run log (task_start + a passing step_output)
        (runs / "r1.jsonl").write_text(
            '{"type": "task_start", "goal": "explain ren", "mode": "advisor", "taskId": "r1"}\n'
            '{"type": "step_output", "step": "s1", "passed": true, "output": "Ren is the Confucian virtue of benevolence. Decision: ok. 中文。"}\n',
            encoding="utf-8",
        )
        result = mc.consolidate_runs(runs)
        assert result["consolidated"] == 1, result
        assert wiki_store.read_page("mem_r1") is not None  # page id derives from the run's taskId


def test_plan_recall_block_safe() -> None:
    # restore the real wiki dirs (earlier tests redirected them to deleted tmp dirs)
    from agent.config import WIKI_DIR, WIKI_MEMORY_DIR

    wiki_store.CANONICAL_DIR = WIKI_DIR
    wiki_store.MEMORY_DIR = WIKI_MEMORY_DIR
    wiki_store.DRAFT_DIR = WIKI_DIR / "drafts"
    # recall over the real wiki returns a memory block and never raises
    block = harness._memory_recall("Dao De Jing Laozi attribution")
    assert "Memory" in block and "[[dao_de_jing]]" in block
    # a nonsense goal yields an empty (not crashing) block
    assert isinstance(harness._memory_recall("zzzzqqq nonexistent topic"), str)


def main() -> int:
    test_consolidate_clean_answer()
    test_consolidate_rejects_lineage_merge_answer()
    test_consolidate_runs_folder()
    test_plan_recall_block_safe()
    print("test_memory_consolidation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
