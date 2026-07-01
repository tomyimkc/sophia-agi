# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Ignorance-as-a-node: build 'gap' pseudo-nodes from OPEN failure-ledger items.

Sophia's charter treats what it does NOT know as first-class graph structure, not
absence. This module parses the open items out of ``agi-proof/failure-ledger.md``
(a Markdown table) and/or ``agi-proof/evidence-manifest.json`` openItems[], and
materialises each as a ``gap`` pseudo-node:

    {id: 'gap-<slug>', pageType: 'gap', title, ledgerId, concerns: [wiki ids],
     status: 'open'}

``concerns`` is filled by ``link_gaps_to_concepts`` using a deterministic keyword
overlap between the ledger id's tokens and each concept page's id / title tokens.
These gaps are NEVER written into the wiki — they are an overlay the coupling gate
reports on (grounded-ignorance coverage), never a page mutation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _import_as_list():
    """Import okf.schema.as_list, tolerating a broken sibling in okf/__init__.

    Importing ``okf.schema`` normally executes ``okf/__init__.py``, which (in some
    environments) transitively imports agent modules using Python-3.11+ regex
    syntax that fails to compile under older interpreters. Try the normal import,
    then fall back to loading ``okf/schema.py`` directly (it is pure stdlib).
    """
    try:
        from okf.schema import as_list as _al
        return _al
    except Exception:  # pragma: no cover - only on a broken sibling / old runtime
        import importlib.util
        schema_path = Path(__file__).resolve().parent / "schema.py"
        spec = importlib.util.spec_from_file_location("okf_schema_direct", schema_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.as_list


as_list = _import_as_list()

# Tokens too generic to carry a gap<->concept keyword match on their own.
_STOP_TOKENS = frozenset({
    "the", "a", "an", "of", "in", "on", "and", "or", "to", "by", "for",
    "not", "yet", "run", "built", "only", "gate", "eval", "harness", "pack",
    "runner", "report", "test", "v1", "v2", "v3", "v4", "v5", "proof",
    "open", "closed", "resolved", "2026", "06", "candidate", "instrument",
})

# A trailing -YYYY-MM-DD date suffix on a kebab ledger id.
_DATE_SUFFIX = re.compile(r"-20\d\d-\d\d-\d\d$")


def _slugify(value: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return s.strip("-")


def _id_tokens(ledger_id: str) -> "set[str]":
    """Content tokens from a ledger id (date suffix stripped, stopwords removed)."""
    base = _DATE_SUFFIX.sub("", str(ledger_id).lower())
    raw = re.split(r"[^a-z0-9]+", base)
    return {t for t in raw if t and t not in _STOP_TOKENS and len(t) > 2}


def _title_from_id(ledger_id: str) -> str:
    base = _DATE_SUFFIX.sub("", str(ledger_id))
    return base.replace("-", " ").strip()


def _is_open_status(status: str) -> bool:
    """A ledger row is a live gap iff its Status cell CONTAINS 'open'.

    Matches 'Open', 'Open (...)', 'OpenAI-compatible ...' — the required response
    column sometimes bleeds 'OpenAI' into the status cell on wrapped rows, so we
    accept any leading-word 'open' but reject 'Closed'/'Resolved' cells.
    """
    s = str(status).strip().lower()
    if s.startswith("closed") or s.startswith("resolved"):
        return False
    return s.startswith("open")


def parse_ledger_open_items(ledger_path) -> "list[dict]":
    """Parse the failure-ledger Markdown table; return open rows as dicts.

    Each dict: {ledgerId, statusRaw}. Robust to the huge free-text cells in the
    real ledger (we only read the first two pipe-delimited columns of each row).
    """
    path = Path(ledger_path)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        ledger_id, status = cells[0], cells[1]
        # Skip the header row and its separator.
        if ledger_id.lower() in ("failure id", "") or set(ledger_id) <= set("-: "):
            continue
        if not _is_open_status(status):
            continue
        out.append({"ledgerId": ledger_id, "statusRaw": status})
    return out


def parse_manifest_open_items(manifest_path) -> "list[str]":
    """Return evidence-manifest.json failureLedgerSummary.openItems[] (or [])."""
    path = Path(manifest_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    summary = data.get("failureLedgerSummary", {}) if isinstance(data, dict) else {}
    return [str(x) for x in as_list(summary.get("openItems"))]


def _gap_from_id(ledger_id: str, status: str = "open") -> dict:
    return {
        "id": "gap-" + _slugify(ledger_id),
        "pageType": "gap",
        "title": _title_from_id(ledger_id),
        "ledgerId": ledger_id,
        "concerns": [],
        "status": "open",
        "statusRaw": status,
    }


def load_gaps(ledger_path, manifest_path=None) -> "list[dict]":
    """Load gap pseudo-nodes from the ledger table and/or manifest openItems.

    The union of both sources is taken, de-duplicated by ledgerId, and returned
    sorted by gap id (deterministic). Manifest items that also appear in the
    ledger are merged (the ledger's status text wins).
    """
    by_id: dict = {}
    for row in parse_ledger_open_items(ledger_path):
        gap = _gap_from_id(row["ledgerId"], row["statusRaw"])
        by_id[row["ledgerId"]] = gap
    if manifest_path is not None:
        for item in parse_manifest_open_items(manifest_path):
            if item not in by_id:
                by_id[item] = _gap_from_id(item, "open (manifest)")
    return sorted(by_id.values(), key=lambda g: g["id"])


def _concept_tokens(page) -> "set[str]":
    meta = page.meta if isinstance(getattr(page, "meta", None), dict) else {}
    tokens: set[str] = set()
    for source in (page.id, meta.get("canonicalTitleEn"), meta.get("subfield")):
        if not source:
            continue
        raw = re.split(r"[^a-z0-9]+", str(source).lower())
        tokens |= {t for t in raw if t and t not in _STOP_TOKENS and len(t) > 2}
    return tokens


def link_gaps_to_concepts(gaps, pages, *, min_overlap: int = 1) -> "list[dict]":
    """Fill each gap's ``concerns`` with wiki ids it plausibly bears on.

    Deterministic keyword match: a concept is a concern of a gap iff the gap id's
    content tokens overlap the concept's id/title/subfield tokens by at least
    ``min_overlap`` tokens. Mutates the gap dicts in place AND returns them.
    ``concerns`` is sorted for stable output.
    """
    page_tokens = [(str(p.id), _concept_tokens(p)) for p in pages]
    for gap in gaps:
        gtok = _id_tokens(gap["ledgerId"])
        concerns = sorted(
            pid for pid, ptok in page_tokens
            if pid and len(gtok & ptok) >= min_overlap
        )
        gap["concerns"] = concerns
    return gaps


def coverage(gaps) -> dict:
    """Grounded-ignorance coverage: fraction of gaps linked to >=1 concept."""
    total = len(gaps)
    linked = sum(1 for g in gaps if g.get("concerns"))
    frac = round(linked / total, 4) if total else 0.0
    return {"gapCount": total, "linkedGapCount": linked, "coverage": frac}
