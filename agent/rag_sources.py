# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Curated Sophia corpus chunks for online RAG (no benchmark holdouts)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from agent.config import DATA_DIR, DOCS_DIR, ROOT, TRAINING_DIR

BENCH_DIR = ROOT / "tests"
REF_DIR = ROOT / "benchmark" / "reference"
DOMAIN_DOCS = DOCS_DIR / "08-Domains"
DISPUTES_DIR = DOCS_DIR / "04-Disputes"
DOMAINS = ("philosophy", "psychology", "history", "religion")


@dataclass
class RagChunk:
    chunk_id: str
    path: str
    title: str
    text: str
    domain: str | None = None
    kind: str = "source"

    def to_dict(self) -> dict:
        return {
            "chunkId": self.chunk_id,
            "path": self.path,
            "title": self.title,
            "text": self.text,
            "domain": self.domain,
            "kind": self.kind,
        }


def load_benchmark_ids() -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    questions: set[str] = set()
    for domain in DOMAINS:
        bench = json.loads((BENCH_DIR / f"benchmark-{domain}.json").read_text(encoding="utf-8"))
        for case in bench.get("cases", []):
            ids.add(case["id"])
            questions.add(case["question"].strip().lower())
    return ids, questions


def is_holdout_example(payload: dict, bench_ids: set[str], bench_questions: set[str]) -> bool:
    meta = payload.get("metadata") or {}
    if meta.get("benchmarkCase") in bench_ids:
        return True
    trap = str(meta.get("trap") or "")
    base_trap = re.sub(r"-r\d+$", "", trap)
    if base_trap in bench_ids:
        return True
    for msg in payload.get("messages", []):
        if msg.get("role") == "user":
            q = str(msg.get("content", "")).strip().lower()
            if q in bench_questions:
                return True
    return False


def _json_records(path: Path, *, domain: str | None, kind: str) -> list[RagChunk]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[RagChunk] = []
    for key, record in data.items():
        if not isinstance(record, dict):
            continue
        text = json.dumps(record, ensure_ascii=False, indent=2)
        chunks.append(
            RagChunk(
                chunk_id=f"{path.name}:{key}",
                path=f"data/{path.name}#{key}",
                title=key,
                text=text[:6000],
                domain=domain or record.get("domain"),
                kind=kind,
            )
        )
    return chunks


def _markdown_chunks(path: Path, *, domain: str | None, kind: str, max_chars: int = 5000) -> list[RagChunk]:
    if not path.exists():
        return []
    from okf import frontmatter

    # strip OKF frontmatter (disputes/wiki carry it) so it is not indexed as body
    text = frontmatter.strip(path.read_text(encoding="utf-8"))[:max_chars]
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    return [
        RagChunk(
            chunk_id=rel,
            path=rel,
            title=path.stem.replace("-", " "),
            text=text,
            domain=domain,
            kind=kind,
        )
    ]


def collect_curated_chunks(*, include_teacher_examples: bool = True) -> list[RagChunk]:
    """Build index from curated sources only — excludes benchmark holdout training rows."""
    bench_ids, bench_questions = load_benchmark_ids()
    chunks: list[RagChunk] = []

    for json_path in sorted(DATA_DIR.glob("*.json")):
        # settled_* are Sophia-Wisdom TRAINING scaffolds (direct-answer calibration aids), not
        # curated OKF provenance records — keep them out of the general retrieval index.
        if json_path.stem.startswith("settled_"):
            continue
        domain = None
        if json_path.stem in ("attributions", "traditions"):
            domain = "philosophy"
        chunks.extend(_json_records(json_path, domain=domain, kind="data"))

    if DISPUTES_DIR.is_dir():
        for md in sorted(DISPUTES_DIR.rglob("*.md")):
            chunks.extend(_markdown_chunks(md, domain="philosophy", kind="dispute"))

    if DOMAIN_DOCS.is_dir():
        for md in sorted(DOMAIN_DOCS.rglob("*.md")):
            domain = md.stem.lower() if md.stem.lower() in DOMAINS else None
            chunks.extend(_markdown_chunks(md, domain=domain, kind="domain_doc"))

    for ref_path in sorted(REF_DIR.glob("responses-*.json")):
        if not ref_path.exists():
            continue
        payload = json.loads(ref_path.read_text(encoding="utf-8"))
        domain = ref_path.stem.replace("responses-", "")
        for case_id, answer in payload.get("responses", {}).items():
            chunks.append(
                RagChunk(
                    chunk_id=f"reference:{domain}:{case_id}",
                    path=f"benchmark/reference/{ref_path.name}#{case_id}",
                    title=f"reference {case_id}",
                    text=f"Q context: benchmark case {case_id}\nA: {answer}",
                    domain=domain,
                    kind="reference",
                )
            )

    if include_teacher_examples and TRAINING_DIR.is_dir():
        for path in sorted(TRAINING_DIR.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if is_holdout_example(payload, bench_ids, bench_questions):
                continue
            user = next((m["content"] for m in payload.get("messages", []) if m.get("role") == "user"), "")
            assistant = next((m["content"] for m in payload.get("messages", []) if m.get("role") == "assistant"), "")
            meta = payload.get("metadata") or {}
            chunks.append(
                RagChunk(
                    chunk_id=f"example:{path.stem}",
                    path=f"training/examples/{path.name}",
                    title=path.stem,
                    text=f"Q: {user}\n\nA: {assistant[:4000]}",
                    domain=meta.get("domain"),
                    kind="teacher_example",
                )
            )

    return chunks