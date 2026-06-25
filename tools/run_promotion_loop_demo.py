#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Demo the projection promotion loop: submit → approve → commit (default-deny)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import wiki_store  # noqa: E402
from okf.bulk_graph import BulkGraph  # noqa: E402
from okf.graph import build as build_graph  # noqa: E402
from okf.page import Page  # noqa: E402
from okf.projection import project_to_boundary  # noqa: E402
from okf.promotion_loop import (  # noqa: E402
    approve_projection_candidate,
    commit_approved_candidate,
    promotion_loop_report,
    submit_projection_candidates,
)

DEFAULT_OUT = ROOT / "agi-proof" / "promotion-loop" / "promotion-loop.public-report.json"


def run_demo(*, wiki_memory_dir: Path | None = None) -> dict:
    if wiki_memory_dir is not None:
        wiki_store.MEMORY_DIR = wiki_memory_dir
        wiki_store.DRAFT_DIR = wiki_memory_dir.parent / "drafts"

    boundary = build_graph([
        Page(
            path="wiki/dao_de_jing.md",
            meta={"id": "dao_de_jing", "pageType": "text", "tradition": "daoist"},
            body="body",
        ),
    ])
    bulk = BulkGraph(boundary=boundary)
    node_id = "bulk_promo_demo"
    bulk.add_node(
        node_id,
        meta={"id": node_id, "pageType": "concept", "tradition": "daoist", "authorConfidence": "disputed"},
        body=(
            "Candidate bulk note only (candidateOnly). "
            "Sophia is an AGI-candidate verifier-gated epistemic framework; "
            "this bulk note is candidate infrastructure only. 中文摘要。"
        ),
    )
    projection = project_to_boundary(bulk, skip_provenance=True, skip_conscience=True)

    with tempfile.TemporaryDirectory() as tmp:
        pending = Path(tmp) / "pending_projection_candidates.jsonl"
        submit = submit_projection_candidates(projection, path=pending)

        denied = commit_approved_candidate(node_id, path=pending, tier="draft")
        approve = approve_projection_candidate(node_id, path=pending, reviewer="demo", note="smoke approve")
        commit1 = commit_approved_candidate(node_id, path=pending, tier="draft", reviewer="demo")
        commit2 = commit_approved_candidate(node_id, path=pending, tier="draft", reviewer="demo")

        report = promotion_loop_report(path=pending)
        report.update({
            "schema": "sophia.promotion_loop_demo.v1",
            "candidateOnly": True,
            "level3Evidence": False,
            "submit": submit,
            "defaultDenyAttempt": denied,
            "approve": approve,
            "firstCommit": commit1,
            "secondCommit": commit2,
            "invariants": {
                "defaultDenyBlocksUnapproved": denied.get("defaultDeny") is True and not denied.get("ok"),
                "approveThenCommit": commit1.get("ok") is True,
                "idempotentSecondCommit": commit2.get("idempotent") is True,
            },
            "ok": (
                denied.get("defaultDeny") is True
                and commit1.get("ok") is True
                and commit2.get("idempotent") is True
            ),
        })
        return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Projection promotion loop demo")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as wiki_tmp:
        mem = Path(wiki_tmp) / "wiki"
        mem.mkdir(parents=True)
        report = run_demo(wiki_memory_dir=mem)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "out": str(out)}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
