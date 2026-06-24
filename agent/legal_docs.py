# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Document ingestion for legal citation checking (build-item 4).

Extract text from real work product (TXT/MD, DOCX, HTML; PDF if a parser is
installed) and run the fail-closed citation verifier over it — so contract /
filing review checks citations in actual documents, not typed strings.

Dependency-light: TXT/MD/HTML/DOCX use only the standard library (DOCX is a zip of
XML). PDF needs an optional parser (``pypdf`` or ``pdfminer.six``), lazily imported;
if neither is present a clear ``DocIngestError`` is raised rather than a silent miss.
"""

from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

from agent.legal_citations import extract_citations, load_known_authorities

TEXT_SUFFIXES = {".txt", ".md", ".text"}
HTML_SUFFIXES = {".html", ".htm", ".xhtml"}


class DocIngestError(RuntimeError):
    """Raised when a document cannot be read (unsupported type / missing parser)."""


def extract_text(path: "str | Path") -> str:
    """Return the plain text of a document, dispatched by file extension."""
    p = Path(path)
    if not p.exists():
        raise DocIngestError(f"file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return p.read_text(encoding="utf-8", errors="replace")
    if suffix in HTML_SUFFIXES:
        return _strip_markup(p.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".docx":
        return _docx_text(p)
    if suffix == ".pdf":
        return _pdf_text(p)
    raise DocIngestError(f"unsupported document type: {suffix or '(none)'}")


def _strip_markup(markup: str) -> str:
    """Drop tags, keep block breaks, unescape entities — enough for citations."""
    markup = re.sub(r"(?i)</(p|div|br|li|h[1-6]|tr)\s*>", "\n", markup)
    markup = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", markup)
    text = re.sub(r"<[^>]+>", "", markup)
    return html.unescape(text)


def _docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", "replace")
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise DocIngestError(f"could not read DOCX {path.name}: {exc}") from exc
    # Paragraph end -> newline; runs within a paragraph concatenate (no spurious
    # spaces, so "[2025] HKCFI 808" split across runs still reassembles).
    xml = re.sub(r"(?i)</w:p>", "\n", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


def _pdf_text(path: Path) -> str:
    try:
        import pypdf  # type: ignore
    except ImportError:
        pypdf = None
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001 - parser internals vary
            raise DocIngestError(f"could not read PDF {path.name}: {exc}") from exc
    try:
        from pdfminer.high_level import extract_text as _pm_extract  # type: ignore
    except ImportError as exc:
        raise DocIngestError(
            "PDF support needs an optional parser — install 'pypdf' or 'pdfminer.six'"
        ) from exc
    try:
        return _pm_extract(str(path))
    except Exception as exc:  # noqa: BLE001
        raise DocIngestError(f"could not read PDF {path.name}: {exc}") from exc


def scan_text(text: str, *, known: "set | None" = None, resolver=None) -> dict:
    """Classify every legal citation in ``text`` as verified or unverified.

    Mirrors ``agent.verifiers.legal_citation_exists``: a citation passes if it is
    in the trusted register, else (optionally) if the live ``resolver`` verifies
    it. Fail-closed — anything else is flagged. ``passed`` is True iff no citation
    is unverified.
    """
    known = known if known is not None else load_known_authorities()
    citations = extract_citations(text or "")
    verified: list[str] = []
    unverified: list[dict] = []
    for c in citations:
        if c in known:
            verified.append(c)
            continue
        res = _try_resolve(resolver, c)
        if res is not None and getattr(res, "verified", False):
            verified.append(c)
        else:
            unverified.append({"citation": c, "status": getattr(res, "status", "not_in_register")})
    return {
        "citationsFound": len(citations),
        "verified": verified,
        "unverified": unverified,
        "passed": not unverified,
    }


def _try_resolve(resolver, citation: str):
    if resolver is None:
        return None
    try:
        return resolver(citation)
    except Exception:  # noqa: BLE001 - fail-closed: a broken resolver verifies nothing
        return None


def scan_document(path: "str | Path", *, known: "set | None" = None, resolver=None) -> dict:
    """Extract a document's text and scan its citations. Adds source metadata."""
    text = extract_text(path)
    report = scan_text(text, known=known, resolver=resolver)
    report["source"] = str(path)
    report["charsExtracted"] = len(text)
    return report
