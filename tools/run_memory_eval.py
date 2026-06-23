#!/usr/bin/env python3
"""Evaluate Sophia's memory substrate for append-only learning safety.

This is a harness-level memory test, not a capability claim. It verifies:
1. a safe verified result can be consolidated into memory;
2. the new memory is searchable/recallable;
3. a forbidden provenance merge is rejected;
4. protected canonical wiki files are not mutated.

By default it writes to a temporary memory tier, so it is safe for CI/local runs.
Use --live-memory only when you intentionally want to write to agent/memory/wiki.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import memory_consolidation, wiki_store  # noqa: E402

OUT = ROOT / "eval" / "results" / "memory_eval.json"
CANONICAL_PROBE = ROOT / "wiki" / "text" / "dao_de_jing.md"


def _hash_tree(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return "missing"
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run(*, out: Path, live_memory: bool = False) -> dict:
    old_memory_dir = wiki_store.MEMORY_DIR
    canonical_before = _hash_tree(ROOT / "wiki")
    with tempfile.TemporaryDirectory(prefix="sophia-memory-eval-") as tmp:
        if not live_memory:
            wiki_store.MEMORY_DIR = Path(tmp) / "memory"
        safe = memory_consolidation.consolidate_result(
            "Remember the Project Phoenix Charter authorship",
            "Project Phoenix Charter was written by the founding committee.",
            task_id="memory-eval-safe",
            mode="repo",
            tier="memory",
        )
        hits = wiki_store.search("Phoenix founding committee", top_k=5)
        recall_ok = any("founding committee" in p.body.lower() for p in hits)
        rejected = memory_consolidation.consolidate_result(
            "Unsafe lineage merge",
            "Confucius wrote the Dao De Jing.",
            task_id="memory-eval-unsafe",
            mode="repo",
            tier="memory",
        )
        canonical_after = _hash_tree(ROOT / "wiki")
        if not live_memory:
            wiki_store.MEMORY_DIR = old_memory_dir
        report = {
            "benchmark": "memory-append-only-safety",
            "mode": "live-memory" if live_memory else "temp-memory",
            "claimStatus": "Harness invariant evidence; not a long-horizon memory capability claim.",
            "checks": {
                "safeConsolidated": bool(safe.get("ok")),
                "recallFindsSafeMemory": recall_ok,
                "unsafeRejected": bool(rejected.get("rejected")) and not rejected.get("ok"),
                "canonicalWikiUnchanged": canonical_before == canonical_after,
            },
            "safe": safe,
            "unsafe": rejected,
            "canonicalHashBefore": canonical_before,
            "canonicalHashAfter": canonical_after,
        }
    report["passed"] = all(report["checks"].values())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("MEMORY EVAL PASS ✓" if report["passed"] else "MEMORY EVAL FAIL ✗")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--live-memory", action="store_true")
    args = ap.parse_args(argv)
    report = run(out=args.out, live_memory=args.live_memory)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
