#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Project data/*.json provenance records into OKF wiki/ pages, and check drift.

data/*.json stays the single source of truth for provenance fields; this emits one
OKF Markdown page per record (frontmatter = the provenance record, body = prose).
`check` fails if a page's provenance frontmatter has drifted from its JSON source,
so the wiki and the structured corpus can never silently disagree (which would be
exactly the lineage-merge Sophia exists to prevent, at the repository level).

    python tools/wiki_sync.py emit       # (re)write wiki/ pages from data/
    python tools/wiki_sync.py check      # exit 1 if any page drifted/missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import frontmatter  # noqa: E402

DATA_DIR = ROOT / "data"
WIKI_DIR = ROOT / "wiki"

# domain -> data file (mirrors data/domains.json dataFile)
DOMAIN_FILES = {
    "philosophy": "attributions.json",
    "psychology": "psychology_concepts.json",
    "history": "history_events.json",
    "religion": "religion_concepts.json",
}

# Canonical frontmatter key order for generated pages.
KEY_ORDER = (
    "id",
    "pageType",
    "domain",
    "recordType",
    "tradition",
    "attributedAuthor",
    "authorConfidence",
    "period",
    "subfield",
    "canonicalTitleEn",
    "canonicalTitleZh",
    "doNotAttributeTo",
    "doNotMergeWith",
    "sources",
    "links",
)

# Provenance keys compared by `check` (body is human/agent-editable; these are not).
PROVENANCE_KEYS = (
    "id",
    "pageType",
    "domain",
    "tradition",
    "attributedAuthor",
    "authorConfidence",
    "doNotAttributeTo",
    "doNotMergeWith",
)


def _ordered(meta: dict) -> dict:
    out: dict = {}
    for key in KEY_ORDER:
        if key in meta:
            out[key] = meta[key]
    for key, value in meta.items():  # any extra keys, stable
        if key not in out:
            out[key] = value
    return out


def _period(record: dict):
    return record.get("compilationPeriod") or record.get("period") or record.get("dateConsensus")


def _record_meta(rid: str, record: dict, *, domain: str, source_ref: str) -> dict:
    page_type = record.get("recordType") or "concept"
    meta = {
        "id": rid,
        "pageType": page_type,
        "domain": record.get("domain") or domain,
        "recordType": page_type,
        "tradition": record.get("tradition"),
        "attributedAuthor": record.get("attributedAuthor"),
        "authorConfidence": record.get("authorConfidence"),
        "period": _period(record),
        "subfield": record.get("subfield"),
        "canonicalTitleEn": record.get("canonicalTitleEn"),
        "canonicalTitleZh": record.get("canonicalTitleZh"),
        "doNotAttributeTo": list(record.get("doNotAttributeTo", [])),
        "doNotMergeWith": list(record.get("doNotMergeWith", [])),
        "sources": [source_ref],
        "links": [],
    }
    return _ordered({k: v for k, v in meta.items() if v is not None or k in ("doNotAttributeTo", "doNotMergeWith", "links")})


def _tradition_meta(tid: str, record: dict, *, source_ref: str) -> dict:
    meta = {
        "id": tid,
        "pageType": "tradition",
        "canonicalTitleEn": record.get("labelEn"),
        "canonicalTitleZh": record.get("labelZh"),
        "doNotMergeWith": list(record.get("doNotMergeWith", [])),
        "sources": [source_ref],
        "links": [],
    }
    return _ordered({k: v for k, v in meta.items() if v is not None or k in ("doNotMergeWith", "links")})


def _summary_sentence(meta: dict) -> str:
    """A single answer-bearing prose sentence composed ONLY from already-sourced provenance
    fields (Step 5 enrichment). Authors no new facts — it just surfaces title / domain /
    recordType / subfield / tradition / attributedAuthor+confidence / period as prose the
    grounded model can answer who/when/what-domain questions from, instead of leaving them
    as bullet scaffold a reader (and the source-sufficiency audit) skips."""
    title = meta.get("canonicalTitleEn") or meta.get("id")
    zh = meta.get("canonicalTitleZh")
    name = f"{title}" + (f" ({zh})" if zh else "")
    record_type = meta.get("recordType") or meta.get("pageType") or "record"
    domain = meta.get("domain")
    kind = f"{domain} {record_type}" if domain else str(record_type)
    clauses = [f"{name} is a {kind}"]
    if meta.get("subfield"):
        clauses.append(f"in the {meta['subfield']} subfield")
    if meta.get("tradition"):
        clauses.append(f"in the {meta['tradition']} tradition")
    if meta.get("attributedAuthor"):
        clauses.append(f"attributed to {meta['attributedAuthor']} "
                       f"(authorship confidence: {meta.get('authorConfidence', 'unknown')})")
    if meta.get("period"):
        clauses.append(f"dated to {meta['period']}")
    sentence = ", ".join(clauses) + "."
    return sentence[0].upper() + sentence[1:]


def _render_body(meta: dict, notes: str, summary: str = "") -> str:
    title = meta.get("canonicalTitleEn") or meta.get("id")
    zh = meta.get("canonicalTitleZh")
    head = f"# {title}" + (f" ({zh})" if zh else "")
    # Lead with an answer-bearing prose summary (sourced fields only), then the bullets.
    lines = [head, "", _summary_sentence(meta)]
    # Step 5 enrichment hook: a record may carry an authored `summary` with richer,
    # human-vetted sourced prose; surface it so grounded answers can draw on it. Absent
    # (the default today), the field-derived sentence above stands on its own.
    if summary and summary.strip():
        lines += ["", summary.strip()]
    lines += [""]
    if meta.get("attributedAuthor"):
        lines.append(f"- **Attributed author:** {meta['attributedAuthor']} "
                     f"(confidence: `{meta.get('authorConfidence', 'unknown')}`)")
    if meta.get("tradition"):
        lines.append(f"- **Tradition:** {meta['tradition']}")
    if meta.get("domain"):
        lines.append(f"- **Domain:** {meta['domain']}")
    if notes:
        lines += ["", notes]
    dna = meta.get("doNotAttributeTo") or []
    dnm = meta.get("doNotMergeWith") or []
    if dna or dnm:
        lines.append("")
        if dna:
            lines.append(f"> **Do not attribute to:** {', '.join(dna)}.")
        if dnm:
            lines.append(f"> **Do not merge with:** {', '.join(dnm)}.")
    lines += ["", f"_Provenance frontmatter is authoritative; generated from `{meta['sources'][0]}` "
                  "by `tools/wiki_sync.py`._"]
    return "\n".join(lines) + "\n"


def _tradition_ids() -> "set":
    path = DATA_DIR / "traditions.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {record.get("id") or key for key, record in data.items() if isinstance(record, dict)}


def build_pages(wiki_dir: Path = WIKI_DIR) -> "list[dict]":
    """Return [{path, meta, body, key}] for every record across the domain files."""
    pages: list[dict] = []
    seen: dict = {}
    tradition_ids = _tradition_ids()

    for domain, filename in DOMAIN_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, record in data.items():
            if not isinstance(record, dict):
                continue
            rid = record.get("recordId") or record.get("textId") or key
            meta = _record_meta(rid, record, domain=domain, source_ref=f"data/{filename}#{key}")
            # connect each entity to its tradition page when that page exists (no dangling links)
            tradition = meta.get("tradition")
            if tradition and tradition in tradition_ids:
                meta["links"] = [tradition]
            page_path = wiki_dir / str(meta["pageType"]) / f"{rid}.md"
            if page_path in seen:
                raise ValueError(f"id collision: {page_path} ({rid} from {filename} vs {seen[page_path]})")
            seen[page_path] = filename
            pages.append({"path": page_path, "meta": meta,
                          "body": _render_body(meta, record.get("notes", ""), record.get("summary", "")),
                          "key": rid})

    traditions_path = DATA_DIR / "traditions.json"
    if traditions_path.exists():
        data = json.loads(traditions_path.read_text(encoding="utf-8"))
        for key, record in data.items():
            if not isinstance(record, dict):
                continue
            tid = record.get("id") or key
            meta = _tradition_meta(tid, record, source_ref=f"data/traditions.json#{key}")
            page_path = wiki_dir / "tradition" / f"{tid}.md"
            pages.append({"path": page_path, "meta": meta, "body": _render_body(meta, ""), "key": tid})

    return pages


def emit(wiki_dir: Path = WIKI_DIR) -> dict:
    pages = build_pages(wiki_dir)
    for page in pages:
        page["path"].parent.mkdir(parents=True, exist_ok=True)
        page["path"].write_text(frontmatter.serialize(page["meta"], page["body"]), encoding="utf-8")
    rel = wiki_dir.relative_to(ROOT) if wiki_dir.is_relative_to(ROOT) else wiki_dir
    return {"ok": True, "written": len(pages), "wikiDir": str(rel)}


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def check(wiki_dir: Path = WIKI_DIR) -> dict:
    pages = build_pages(wiki_dir)
    missing: list[str] = []
    drift: list[dict] = []
    for page in pages:
        path = page["path"]
        if not path.exists():
            missing.append(_rel(path))
            continue
        existing, _ = frontmatter.parse(path.read_text(encoding="utf-8"))
        for key in PROVENANCE_KEYS:
            if key in page["meta"] and existing.get(key) != page["meta"][key]:
                drift.append({"page": _rel(path), "key": key,
                              "data": page["meta"][key], "wiki": existing.get(key)})
    return {"ok": not missing and not drift, "pages": len(pages), "missing": missing, "drift": drift}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync data/*.json <-> OKF wiki/ pages")
    parser.add_argument("command", choices=["emit", "check"], help="emit pages or check for drift")
    args = parser.parse_args()
    result = emit() if args.command == "emit" else check()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
