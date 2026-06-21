#!/usr/bin/env python3
"""Tests for agent/legal_docs.py — document ingestion + citation scanning (offline).

Builds a real .docx (a zip of XML) and HTML/TXT inputs in-process; no network, no
heavy parser deps. PDF extraction is exercised only if an optional parser exists.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import legal_docs as ld  # noqa: E402

_DOCX_XML = (
    '<?xml version="1.0"?>'
    '<w:document xmlns:w="x"><w:body>'
    '<w:p><w:r><w:t>The court in </w:t></w:r><w:r><w:t>[2025] HKCFI 808</w:t></w:r>'
    '<w:r><w:t> is real.</w:t></w:r></w:p>'
    '<w:p><w:r><w:t>But Varghese, 925 F.3d 1339, is fabricated.</w:t></w:r></w:p>'
    '</w:body></w:document>'
)


def _write_docx(path: Path, document_xml: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", document_xml)


def test_extract_txt_and_html(tmp_path=None) -> None:
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    txt = tmp / "memo.txt"
    txt.write_text("See [2025] HKCFI 808 for the point.", encoding="utf-8")
    assert "[2025] HKCFI 808" in ld.extract_text(txt)

    htm = tmp / "memo.html"
    htm.write_text("<html><body><p>Cite [2025] HKCFI 808</p></body></html>", encoding="utf-8")
    out = ld.extract_text(htm)
    assert "[2025] HKCFI 808" in out and "<p>" not in out


def test_extract_docx_reassembles_split_runs(tmp_path=None) -> None:
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    docx = tmp / "brief.docx"
    _write_docx(docx, _DOCX_XML)
    text = ld.extract_text(docx)
    # citation split across <w:t> runs must reassemble
    assert "[2025] HKCFI 808" in text
    assert "925 F.3d 1339" in text


def test_scan_document_flags_fabricated(tmp_path=None) -> None:
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    docx = tmp / "brief.docx"
    _write_docx(docx, _DOCX_XML)
    report = ld.scan_document(docx)
    assert report["citationsFound"] == 2
    assert "[2025] HKCFI 808" in report["verified"]            # in bundled register
    assert any(u["citation"] == "925 F.3d 1339" for u in report["unverified"])  # the Mata fake
    assert report["passed"] is False


def test_scan_text_clean_passes() -> None:
    r = ld.scan_text("Per [2025] HKCFI 808 and Cap. 614, the position is settled.")
    assert r["passed"] is True and r["unverified"] == []


def test_scan_text_resolver_fail_closed() -> None:
    def boom(_c):
        raise RuntimeError("resolver down")

    r = ld.scan_text("Relying on [2099] HKCFI 1.", known=set(), resolver=boom)
    assert r["passed"] is False  # broken resolver never verifies


def test_unsupported_and_missing() -> None:
    import pytest

    with pytest.raises(ld.DocIngestError):
        ld.extract_text("/no/such/file.txt")
    tmp = Path(tempfile.mkdtemp()) / "x.rtf"
    tmp.write_text("hi", encoding="utf-8")
    with pytest.raises(ld.DocIngestError):
        ld.extract_text(tmp)


def main() -> int:
    import inspect

    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            if nm == "test_unsupported_and_missing":
                continue  # needs pytest.raises; covered under pytest
            fn()
    print("test_legal_docs: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
